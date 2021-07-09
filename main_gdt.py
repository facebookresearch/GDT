#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import datetime
from logging import getLogger
import os
import sys
import time


import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from datasets.GDTPretrainDataset import GDTPretrainDataset
from datasets.HT100MDataset import HT100M_Dataset
from gdt_helper import compute_feats, collate_feats, get_pos_neg, get_losses
from model import GDT, TextVid_GDT
from log_utils import MetricLoggerGDT, SmoothedValue
from src.scheduler import GradualWarmupScheduler
from utils import (
    init_distributed_mode,
    init_signal_handler,
    makedir,
    save_checkpoint,
    trigger_job_requeue
)

try:
    from apex import amp
except ImportError:
    amp = None


logger = getLogger()


def main(args):
    # Set up mixed precision training
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError(
                "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                "to enable mixed-precision training."
            )

    # Make output dir
    if args.output_dir:
        makedir(args.output_dir)

    # Init distributed mode
    if torch.cuda.is_available():
        init_distributed_mode(args)

    # init signal handler
    init_signal_handler()

    # Set up logger
    if args.distributed:
        filename = str(args.job_id) + '_' + str(args.rank) + '_log.out'

    # Set up tensorboard
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    global_rank = args.rank if args.distributed else 0
    is_master = True if global_rank == 0 else False
    if is_master:
        writer = SummaryWriter(tbx_path)
        writer.add_text(
            'args',
            " \n".join(['%s : %s' % (arg, getattr(args, arg)) for arg in vars(args)]), 
            0
        )
    else:
        writer = None

    # Log version information
    logger.info(args)
    logger.info(f"torch version: {torch.__version__}")

    # Set distributed mode
    device = torch.device(args.device)

    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True

    # Create model
    logger.info("Creating model")
    if args.model == 'av_gdt':
        model = GDT(
            vid_base_arch=args.vid_base_arch, 
            aud_base_arch=args.aud_base_arch,
            pretrained=False, 
            norm_feat=args.norm_feat, 
            use_mlp=args.use_mlp,
            num_classes=256, 
        )
    else:
        # Video-Text GDT encoder for pretraining
        model = TextVid_GDT(
            vid_base_arch=args.vid_base_arch,
            text_base_arch='word2vec',
            pretrained=False,
            norm_feat=args.norm_feat,
            use_mlp=args.use_mlp,
            num_classes=256,
        )
    model.to(device)
    if args.distributed and args.sync_bn:
        logger.info("Sync BN on model")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )
        model_without_ddp = model.module

    if args.aug_audio:
        if args.audio_augtype == 'mild':
            args.aug_audio = [1, 1, 2, 5]
        elif args.audio_augtype == 'medium':
            args.aug_audio = [1, 1, 3, 6]
        elif args.audio_augtype == 'heavy':
            args.aug_audio = [2, 2, 3, 6]

    # Set up training optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # For Mixed Precision training
    if args.apex:
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=args.apex_opt_level
        )

    # Set up LR scheduler
    milestones = [int(lr) - args.lr_warmup_epochs for lr in args.lr_milestones.split(',')]
    lr_scheduler = None
    if args.use_scheduler:
        if args.lr_warmup_epochs > 0:
            if args.scheduler_type == 'multi_step':
                logger.info(f'Using Multi-Step LR scheduler')
                scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                logger.info(f'Using Cosine Annealing LR scheduler')
                scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            lr_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=args.world_size,
                total_epoch=args.lr_warmup_epochs,
                after_scheduler=scheduler_step
            )
        else:
            if args.scheduler_type == 'multi_step':
                logger.info(f'Using Multi-Step LR scheduler')
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                logger.info(f'Using Cosine Annealing LR scheduler')
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Checkpointing restart
    ckp_path = os.path.join(args.output_dir, 'checkpoints', 'checkpoint.pth')
    if os.path.isfile(ckp_path):
        logger.info(f'Loading checkpoint')
        checkpoint = torch.load(ckp_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        logger.info(f'Restrating at epoch {args.start_epoch}')

    # Create dataloader
    if args.dataset == "ht100m":
        ds = HT100M_Dataset(
            csv_file='data/howto.csv',
            video_root=args.root_dir,
            caption_root=args.ht100m_caption_root,
            token_to_word_path='data/dict.npy',
            fps=32/int(args.sample_rate),
            num_frames=args.clip_len,
            size=args.train_crop_size,
            center_crop=args.center_crop, # True
        )
    else:
        # Audio-Visual datasets: Kinetics-400/600, Audioset, VGG-Sound
        ds = GDTPretrainDataset(
            ds_name=args.dataset,
            root_dir=args.root_dir,
            mode='train',
            args=args
        )

    print("Creating data loaders", flush=True)
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None,
        drop_last=True
    )

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if writer:
            writer.add_scalar('train/epoch', epoch, epoch)
        logger.info(f'Start training epoch: {epoch}')
        loss = train_one_epoch(
            args,
            data_loader,
            model,
            optimizer,
            device,
            epoch,
            args.print_freq,
            lr_scheduler,
            args.apex,
            writer=writer,
        )
        if lr_scheduler:
            lr_scheduler.step()
        if args.output_dir:
            save_checkpoint(args, epoch, model, optimizer, lr_scheduler)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')


def train_one_epoch(
        args,
        data_loader,
        model,
        optimizer,
        device,
        epoch,
        print_freq,
        lr_scheduler,
        apex=False,
        logger=None,
        writer=None,
):

    model.train()
    metric_logger = MetricLoggerGDT(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for batch_idx, batch in metric_logger.log_every(
        data_loader, print_freq, header, logger, writer, 'train', epoch=epoch
    ):
        video, spec, label, vid_idx, idx = batch
        video, videoR, video2, video2R  = torch.split(video, [3, 3, 3, 3], dim=1)
        if args.decode_audio:
            audio, audioR, audio2, audio2R = torch.split(spec, [1, 1, 1, 1], dim=1)
        else: # text
            audio, audioR, audio2, audio2R = torch.split(spec, [20, 20, 20, 20], dim=1)
        start_time = time.time()

        video, audio = video.to(device), audio.to(device)

        # form positive and negative pairs dependent on hypothesis
        if args.hypothesis == 1:
            hyp = 'basecase'

            # compute features
            feats1 = compute_feats(model, video, audio)
            feat_v, feat_a = feats1

            # collation across GPUs
            feat_v_col, feat_a_col = collate_feats([feat_v, feat_a])
            feats1_col = (feat_v_col, feat_a_col)

            # basecase cross-modal loss #########################################
            pairs1 = get_pos_neg(hyp, feats1, feats1_col)
            # (V, A)
            loss1, loss_dict1 = get_losses(pairs1, pairs2=None, nce_t=args.nce_t)
            loss = loss1
            #####################################################################
            loss_dict2 = None
        if args.hypothesis in [2, 3]:
            
            # Add Reversal pairs to GPU
            videoR, audioR = videoR.to(device), audioR.to(device)

            # compute features
            feats1, feats2 = compute_feats(model, video, audio, videoR, audioR)
            feat_v, feat_a = feats1
            feat_vR, feat_aR = feats2

            # collation across GPUs
            feat_v_col, feat_a_col, feat_vR_col, feat_aR_col = collate_feats(
                [feat_v,feat_a,feat_vR,feat_aR])
            feats1_col = (feat_v_col, feat_a_col)
            feats2_col = (feat_vR_col, feat_aR_col)

            if args.hypothesis == 2:
                hyp = 'vtime'
                loss_b = None
            if args.hypothesis == 3:
                hyp = 'itime'
                # base loss
                # Pos: (V, A), Neg: Other (V, A) + (V, AT) except from local
                pairs1 = get_pos_neg(
                    'basecase', feats1, feats1_col, feats2, feats2_col
                )
                # Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A) except from local
                pairs2 = get_pos_neg(
                    'basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1]
                )
                loss_b, loss_dict_b = get_losses(pairs1, pairs2, nce_t=args.nce_t)

            # time-reversal ##########################################################
            # Inv: Pos: (V, AT) Var: Pos: (V, A), Neg: Other (V, A) + (V, AT)
            pairs1 = get_pos_neg(
                hyp, feats1, feats1_col, feats2, feats2_col
            )
            # Inv: Pos: (VT, A) Var: Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A)
            pairs2 = get_pos_neg(
                hyp, feats2, feats2_col, feats1, feats1_col, pairs1[-1]
            )
            loss1, loss_dict1 = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            loss = 0.5 * (loss1 + loss_b) if loss_b is not None else loss1
            ###########################################################################
            loss_dict2 = None
        if args.hypothesis in [4, 5]:
            
            # Add time shift pairs to GPU
            video2, audio2 = video2.to(device), audio2.to(device)

            # compute features
            feats1, feats2 = compute_feats(model, video, audio, video2, audio2)
            feat_v, feat_a = feats1
            feat_v2, feat_a2 = feats2

            # collation across GPUs
            feat_v_col, feat_a_col, feat_v2_col, feat_a2_col = collate_feats(
                [feat_v, feat_a, feat_v2, feat_a2])
            feats1_col = (feat_v_col, feat_a_col)
            feats2_col = (feat_v2_col, feat_a2_col)

            if args.hypothesis == 4:
                hyp = 'vasync'
                loss_b = None
            if args.hypothesis == 5:
                hyp = 'iasync'
                # base loss
                # Pos: (V, A), Neg: Other (V, A) + (V, AT) except from local
                pairs1 = get_pos_neg(
                    'basecase', feats1, feats1_col, feats2, feats2_col
                )
                # Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A) except from local
                pairs2 = get_pos_neg(
                    'basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1]
                )
                loss_b, loss_dict_b = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            
            # time-shift ##########################################################
            # Inv: Pos: (V, AT) Var: Pos: (V, A), Neg: Other (V, A) + (V, AT)
            pairs1 = get_pos_neg(hyp, feats1, feats1_col, feats2, feats2_col)
            # Inv: Pos: (VT, A) Var: Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A)
            pairs2 = get_pos_neg(hyp, feats2, feats2_col, feats1, feats1_col, pairs1[-1])
            loss1, loss_dict1 = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            loss = 0.5 * (loss1 + loss_b) if loss_b is not None else loss1
            ########################################################################
            loss_dict2 = None

        elif args.hypothesis >= 6:

            # Add time reversal and shift pairs to GPU
            videoR, audioR = videoR.to(device), audioR.to(device)
            video2, audio2 = video2.to(device), audio2.to(device)
            video2R, audio2R = video2R.to(device), audio2R.to(device)

            # compute features
            feats1 = compute_feats(model, video, audio)
            feats2 = compute_feats(model, video2, audio2)
            feats3 = compute_feats(model, videoR, audioR)
            feats4 = compute_feats(model, video2R, audio2R)
            feat_v, feat_a = feats1
            feat_v2, feat_a2 = feats2
            feat_vR, feat_aR = feats3
            feat_v2R, feat_a2R = feats4

            # collation on GPUs
            outs = collate_feats([feat_v, feat_a, feat_v2, feat_a2,
                                  feat_vR, feat_aR, feat_v2R, feat_a2R])
            feats1_col, feats2_col = tuple(outs[0:2]), tuple(outs[2:4])
            feats3_col, feats4_col = tuple(outs[4:6]), tuple(outs[6:8])

            if args.hypothesis == 6:
                hyp1 = 'vtime'
                hyp2 = 'vasynced'
            elif args.hypothesis == 7:
                hyp1 = 'itime'
                hyp2 = 'vasynced'
            elif args.hypothesis == 8:
                hyp1 = 'vtime'
                hyp2 = 'iasynced'
            elif args.hypothesis == 9:
                hyp1 = 'itime'
                hyp2 = 'iasynced'

            if hyp1 == 'itime':
                # base loss time
                # (V, A)
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats3, feats3_col)
                # (VR, AR)
                pairs2 = get_pos_neg('basecase', feats3, feats3_col, feats1, feats1_col, pairs1[-1])
                loss_b1, loss_dict_b = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            else:
                loss_b1 = None

            if hyp2 == 'iasynced':
                # basecase iasync
                # (V, A)
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats2, feats2_col)
                # (VR, AR)
                pairs2 = get_pos_neg('basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1])
                loss_b2, loss_dict_b = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            else:
                loss_b2 = None

            # time-reversal ##########################################################
            # on normal
            # (V, AR)
            pairs1 = get_pos_neg(hyp1, feats1, feats1_col, feats3, feats3_col)
            # (VR, A)
            pairs2 = get_pos_neg(hyp1, feats3, feats3_col, feats1, feats1_col, pairs1[-1])
            loss1, loss_dict1 = get_losses(pairs1, pairs2, nce_t=args.nce_t)

            ## on time-shifted:
            # (V2, A2R)
            pairs1 = get_pos_neg(hyp1, feats2, feats2_col, feats4, feats4_col)
            # (V2R, A2)
            pairs2 = get_pos_neg(hyp1, feats4, feats4_col, feats2, feats2_col, pairs1[-1])
            loss2, loss_dict2 = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            ##########################################################################

            # time-shift #############################################################
            ## on normal
            # (V, A2)
            pairs1 = get_pos_neg(hyp2, feats1, feats1_col, feats2, feats2_col)
            # (V2, A)
            pairs2 = get_pos_neg(hyp2, feats2, feats2_col, feats1, feats1_col, pairs1[-1])
            loss3, loss_dict3 = get_losses(pairs1, pairs2, nce_t=args.nce_t)

            ## on time-reversed
            # (VR, A2R)
            pairs1 = get_pos_neg(hyp2, feats3, feats3_col, feats4, feats4_col)
            # (V2R, AR)
            pairs2 = get_pos_neg(hyp2, feats4, feats4_col, feats3, feats3_col, pairs1[-1])
            loss4, loss_dict4 = get_losses(pairs1, pairs2, nce_t=args.nce_t)
            ##########################################################################

            # combine losses
            if loss_b1 is not None and loss_b2 is not None:
                loss = (1./6) * (loss1 + loss2 + loss3 + loss4 + loss_b1 + loss_b2)
            elif loss_b1 is not None:
                loss = 0.2 * (loss1 + loss2 + loss3 + loss4 + loss_b1)
            elif loss_b2 is not None:
                loss = 0.2 * (loss1 + loss2 + loss3 + loss4 + loss_b2)
            else:
                loss = 0.25 * (loss1 + loss2 + loss3 + loss4 )

        # Backward pass
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # signal received, relaunch experiment
        if os.environ['SIGNAL_RECEIVED'] == 'True':
            args.resume = 'True'
            if args.rank == 0:
                logger.info("Beginning reqeue")
                trigger_job_requeue(os.path.join(
                    args.output_dir, 'checkpoints', 'checkpoint.pth'))

        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['batch_t/s'].update((time.time() - start_time))
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
    if args.distributed:
        dist.barrier()
    torch.cuda.empty_cache()
    return metric_logger.loss.avg


def parse_args():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
                         'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Video Representation Learning')
    parser.register('type', 'bool', str2bool)

    # Data
    parser.add_argument('--root_dir', type=str, default=None,
        help='path to dataset train directory e.g. /path/to/kinetics/train')
    parser.add_argument('--ht100m_caption_root', type=str, default='/private/home/mandelapatrick/data/howto100m_csv',
        help='path to ht100m caption root directory')
    parser.add_argument('--dataset', default='kinetics', type=str,
        help='name of dataset')
    parser.add_argument('--dualdata', default='True', type='bool',
        help='use dataloader that returns two samples per video')
    parser.add_argument('--num_data_samples', default=None, type=int,
        help='number of samples in dataset')
    parser.add_argument('--fold', default=1, type=str,
        help='fold of dataset (ucf101/ hmdb51)')
    parser.add_argument('--workers', default=0, type=int,
        help='number of data loading workers (default: 0)')

    # GDT NCE loss
    parser.add_argument('--hypothesis', default=1, type=int,
        help="use it for encoding what learning hypothesis we're using")
    parser.add_argument('--nce_t', default=0.07, type=float, 
        help='softmax weighting')
    parser.add_argument('--num_negatives', default=-1, type=int,
        help='number of negatives in contrastive loss')

    # Video Augmentations
    parser.add_argument('--clip_len', default=30, type=int,
        help='number of frames per clip')
    parser.add_argument('--target_fps', default=30, type=int,
        help='target fps')
    parser.add_argument('--sample_rate', default=1, type=int,
        help='Subsampling rate: num frames between clips')
    parser.add_argument('--clips_per_video', default=1, type=int,
        help='number of clips to sample from video')
    parser.add_argument('--train_crop_size', default=112, type=int,
        help='Size of spatial crops')
    parser.add_argument('--colorjitter', default='False', type='bool',
        help='Apply random color jitter')
    parser.add_argument('--use_scale_jittering', default='False', type='bool',
        help='scale jittering as augmentations')
    parser.add_argument('--augtype', default=1, type=int,
        help='augmentation type (default: 1)')
    parser.add_argument('--use_temp_jitter', default='True', type='bool',
        help='Get clips from random timestamps each epoch')
    parser.add_argument('--center_crop', default='False', type='bool',
        help='Use center cropping instead of random cropping')
    
    # Audio Augmentation
    parser.add_argument('--aud_sample_rate', default=24000, type=int,
        help='audio sample rate')
    parser.add_argument('--aud_spec_type', default=1, type=int,
        help='audio spec type') # 1 : (40, 99), (257, 199)
    parser.add_argument('--use_volume_jittering', default='True', type='bool',
        help='use volume jittering')
    parser.add_argument('--use_temporal_jittering', default='False', type='bool',
        help='use temporal jittering')
    parser.add_argument('--num_sec', default=1, type=int,
        help='Number of seconds')
    parser.add_argument('--z_normalize', default='True', type='bool',
        help='normalize audio')
    parser.add_argument('--aug_audio', default='True', type='bool',
        help='whether to augment audio')
    parser.add_argument('--audio_augtype', default='medium', type=str,
        choices=['na', 'mild', 'medium', 'heavy'],
        help='type of audio-augment default: mild')
    parser.add_argument('--decode_audio', default='True', type='bool',
        help='whether to deocde audio')

    # Model
    parser.add_argument('--model', default='av_gdt', help='model',
        choices=['av_gdt', 'vid_text_gdt'])
    parser.add_argument('--vid_base_arch', default='r2plus1d_18',
        help='Video Base Arch for A-V model',
        choices=['r2plus1d_18', 'r2plus1d_34'])
    parser.add_argument('--aud_base_arch', default='resnet9',
        help='Audio Base Arch for A-V model',
        choices=['resnet9', 'resnet18'])
    parser.add_argument('--pretrained', default='False', type='bool',
        help='Use pre-trained models from the modelzoo')
    parser.add_argument('--headcount', default=1, type=int,
        help='how many heads each modality has')
    parser.add_argument('--use_mlp', default='True', type='bool',
        help='Use MLP projection head')
    parser.add_argument('--use_max_pool', default='False', type='bool',
        help='Use max pool instead of GAP')
    parser.add_argument('--mlptype', default=0, type=int,
        help='MLP type (default: 0)')

    # Training
    parser.add_argument('--batch_size', default=16, type=int,
        help='batch-size / GPU')
    parser.add_argument('--epochs', default=200, type=int,
        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='weight decay (default: 1e-5)')
    parser.add_argument('--use_scheduler', default='True', type='bool',
        help='Use LR scheduler')
    parser.add_argument('--scheduler_type', default='multi_step', type=str,
        choices=['multi_step', 'cosine'],
        help='Type of LR scheduler')
    parser.add_argument('--lr_milestones', default='150,175', type=str,
        help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int,
        help='number of warmup epochs')
    parser.add_argument('--sync_bn', default='True', type='bool',
        help='Use sync batch norm')
    parser.add_argument('--warmup_bn', default='False', type='bool',
        help='Warmup batchnorm')
    parser.add_argument('--norm_feat', default='True', type='bool',
        help='Normalize embeddings')

    # Logging
    parser.add_argument('--print_freq', default=10, type=int,
        help='print frequency')
    parser.add_argument('--output_dir', default='.',
        help='path where to save')

    # Checkpointing
    parser.add_argument('--resume', default='False', type='bool',
        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
        help='start epoch')

    # Mixed precision training parameters
    parser.add_argument('--apex', default='False', type='bool', 
        help='Use apex for mixed precision training'
    )
    parser.add_argument('--apex_opt_level', default='O1', type=str,
        help='For apex mixed precision training'
             'O0 for FP32 training, O1 for mixed precision training.'
             'For further detail, see' 
             'https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )

    # distributed training parameters
    parser.add_argument('--device', default='cuda', 
        help='device')
    parser.add_argument('--distributed', default='False', type='bool',
        help='ddp mode')
    parser.add_argument('--dist_backend', default='nccl', type=str,
        help='distributed backend')
    parser.add_argument('--dist_url', default='env://',
        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
        help='number of distributed processes')
    parser.add_argument('--debug_slurm', default='False', type='bool',
        help="Debug SLURM")
    parser.add_argument('--local_rank', default=-1, type=int,
        help='Local rank of node')
    parser.add_argument('--master_port', default=-1, type=int,
        help='Master port of Job')
    parser.add_argument('--bash', default='False', type='bool',
        help='if in bash')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # set multi-processing start method
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass

    main(args)
