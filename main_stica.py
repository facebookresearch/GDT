#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import os
import shutil
import time
from logging import getLogger

# Import torch and other dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from opt import parse_arguments
from utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    init_signal_handler,
    trigger_job_requeue,
    dist_collect_other,
)
from model import Stica_TransformerFMCrop
from datasets.AVideoDataset import AVideoDataset
from src.scheduler import GradualWarmupScheduler

logger = getLogger()

def main():

    # parse arguments
    global args
    parser = parse_arguments()
    args = parser.parse_args()

    # exp setup: logger, distributed mode and seeds
    init_distributed_mode(args)
    init_signal_handler()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    if args.rank == 0:
        writer = SummaryWriter(args.dump_path)
        writer.add_text(
            'args',
            " \n".join(['%s : %s' % (arg, getattr(args, arg)) for arg in vars(args)]), 
            0
        )
    else:
        writer = None

    # Spec Augment params: []
    if args.audio_augtype == 'mild':
        aug_audio = [1, 1, 2, 5]
    elif args.audio_augtype == 'medium':
        aug_audio = [1, 1, 3, 6]
    elif args.audio_augtype == 'heavy':
        aug_audio = [2, 2, 3, 6]
    else:
        aug_audio = []

    train_dataset = AVideoDataset(
        ds_name='kinetics',
        mode='train',
        path_to_data_dir='datasets/data',
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        sample_rate=args.sample_rate,
        num_train_clips=args.num_train_clips,
        train_crop_size=args.train_crop_size,
        test_crop_size=args.test_crop_size,
        num_data_samples=None,
        colorjitter=args.colorjitter,
        use_grayscale=args.use_grayscale,
        use_gaussian=args.use_gaussian,
        temp_jitter=True,
        decode_audio=True,
        aug_audio=aug_audio,
        num_sec=args.num_sec_aud,
        aud_sample_rate=args.aud_sample_rate,
        aud_spec_type=args.aud_spec_type,
        use_volume_jittering=args.use_volume_jittering,
        use_temporal_jittering=args.use_audio_temp_jittering,
        z_normalize=args.z_normalize,
        dual_data=args.dual_data,
        multi_crop=args.multi_crop,
        use_random_resize_crop=args.use_random_resize_crop,
        constant_scale=False,
        num_large_crops=2,
        num_small_crops=0
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(
        len(train_dataset)))

    # build model
    model = Stica_TransformerFMCrop(
        vid_base_arch='r2plus1d_18',
        aud_base_arch='resnet9',
        pretrained=False,
        norm_feat=True,
        use_mlp=True,
        num_classes=256,
        args=args
    )

    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    if args.use_warmup_scheduler:
        warmup_lr_schedule = np.linspace(
            args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
        if args.use_lr_scheduler:
            cosine_lr_schedule = np.array(
                [args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                    math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) 
                    for t in iters
                ])
            lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        else:
            constant_schedule = np.array([args.base_lr for t in iters])
            lr_schedule = np.concatenate((warmup_lr_schedule, constant_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp if args.use_fp16 else None,
    )
    start_epoch = to_restore["epoch"]

    # Set CuDNN benhcmark
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores = train(
            train_loader, model, optimizer, epoch, lr_schedule, writer)
        training_stats.update(scores)
        if args.rank == 0 and writer:
            writer.add_scalar('pretrain/epoch', epoch, epoch)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(
                    args.dump_path, 
                    "checkpoint.pth.tar"
                ),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(
                        args.dump_path, 
                        "checkpoint.pth.tar"
                    ),
                    os.path.join(
                        args.dump_checkpoints, 
                        "ckp-" + str(epoch) + ".pth"
                    ),
                )


def train(train_loader, model, optimizer, epoch, lr_schedule, writer):
    # Put model in train mode
    model.train()
    XE = torch.nn.CrossEntropyLoss()
    # Init Logger meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    croplosses_meter = AverageMeter()
    avlosses = AverageMeter()

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ Get inputs ... ============
        video, audio, _, _, _, _ = inputs
        audio = audio.cuda()

        # ============ forward passes and GDT-Loss ... ============
        feat_v_nce_lst = []
        crop_feat_v_nces_lst = []
        feat_a_nce_lst = []

        # FORWARD PASS
        for i in range(len(video)):
            # get video
            video_input = torch.cat(video[i: i+1]).cuda(non_blocking=True)

            # get crop params
            params = fmcrop_params(
                duration=model.module.duration,
                s_large_crops=args.num_large_crops,  
                s_small_crops=args.num_small_crops,
                t_large_crops=args.num_large_tcrops, 
                t_small_crops=args.num_small_tcrops
            )

            # Forward pass
            feat_v_nce, crop_feat_v_nces, feat_a_nce = model(
                video_input, audio, params=params)

            # Save features
            feat_v_nce_lst.append(feat_v_nce)
            crop_feat_v_nces_lst.append(crop_feat_v_nces)
            feat_a_nce_lst.append(feat_a_nce)

        # LOSS
        crop_losses, counters = nce_crop_losses_dual(
            feats_v=crop_feat_v_nces_lst[0],
            feats_v2=crop_feat_v_nces_lst[1], 
            XE=XE,
            s_large_crops=args.num_large_crops,  
            s_small_crops=args.num_small_crops,
            t_large_crops=args.num_large_tcrops, 
            t_small_crops=args.num_small_tcrops, 
            temp=args.temp
        )
        loss_crops = sum(crop_losses) / sum(counters)
        if args.cross_modal_alpha > 0:
            loss_av = 0.5 * (
                gdt_loss(feat_v_nce_lst[0], feat_a_nce_lst[0], XE) +
                gdt_loss(feat_v_nce_lst[1], feat_a_nce_lst[1], XE)
            )
        else:
            loss_av = torch.tensor(0)

        if args.cross_modal_alpha > 0:
            loss = (
                (1. - args.cross_modal_alpha) * loss_crops + 
                args.cross_modal_alpha * loss_av
            )
        else:
            loss = (1. - args.cross_modal_alpha) * loss_crops

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ============ misc ... ============
        bs = audio.size(0)
        losses.update(loss.item(), bs)
        avlosses.update(loss_av.item(), bs)
        croplosses_meter.update(loss_crops.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.2f} ({loss.avg:.2f})\t"
                "AVLoss {avloss.val:.2f} ({avloss.avg:.2f})\t"
                "CropLoss {closs.val:.2f} ({closs.avg:.2f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    avloss=avlosses,
                    closs=croplosses_meter,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )


            # Log onto tensorboard
            if writer:
                writer.add_scalar(
                    f'pretrain/loss/iter',
                    loss.item(),
                    iteration
                )
                writer.add_scalar(
                    f'pretrain/crop_loss/iter',
                    loss_crops.item(),
                    iteration
                )
                if counters[0] > 0:
                    writer.add_scalar(
                        f"pretrain/crop_loss/L-L/iter",
                        crop_losses[0].item()/counters[0],
                        iteration
                    )
                if counters[1] > 0:
                    writer.add_scalar(
                        f"pretrain/crop_loss/S-L/iter",
                        crop_losses[1].item() / counters[1],
                        iteration
                    )
                if counters[2] > 0:
                    writer.add_scalar(
                        f"pretrain/crop_loss/TL-TL/iter",
                        crop_losses[2].item() / counters[2],
                        iteration
                    )
                if counters[3] > 0:
                    writer.add_scalar(
                        f"pretrain/crop_loss/TS-TL/iter",
                        crop_losses[3].item() / counters[3],
                        iteration
                    )
                writer.add_scalar(
                    f'pretrain/av_loss/iter',
                    loss_av.item(),
                    iteration
                )
                writer.add_scalar(
                    f'pretrain/lr/iter',
                    optimizer.param_groups[0]["lr"],
                    iteration
                )
                writer.add_scalar(
                    f'pretrain/batch_time/iter',
                    batch_time.avg,
                    iteration
                )
                writer.add_scalar(
                    f'pretrain/data_time/iter',
                    data_time.avg,
                    iteration
                )

        # ============ signal handling ... ============
        if os.environ['SIGNAL_RECEIVED'] == 'True':
            if args.rank == 0:
                logger.info("Beginning reqeue")
                trigger_job_requeue(os.path.join(
                    args.dump_path, 
                    "checkpoint.pth.tar"
                ))

    return (epoch, losses.avg)


def gdt_loss(
    feat_v, 
    feat_a, 
    XE, 
    symmetric=True, 
    temp=0.1
):
    # Collate features from other GPUs
    feat_v_other = dist_collect_other(feat_v, 
        return_before_cat=False)

    # Concat positives and negatives
    v_other = torch.cat((feat_v, feat_v_other), 0).detach()

    # Audio-Video NCE loss
    logits_av = torch.einsum('bc,mc->bm', feat_a, v_other)
    labels_av = torch.arange(0, len(logits_av),
        dtype=torch.long).cuda()
    loss_nce_av = XE(logits_av / temp, labels_av)

    # loss is constructed from both
    if symmetric:
        # Video-Audio NCE loss
        feat_a_other = dist_collect_other(feat_a,
            return_before_cat=False)
        a_other = torch.cat((feat_a, feat_a_other), 0).detach()
        logits_va = torch.einsum('bc,mc->bm', feat_v, a_other)
        labels_va = torch.arange(0, len(logits_va),
            dtype=torch.long).cuda()
        loss_nce_va = XE(logits_va / temp, labels_va)
        loss_gdt = 0.5 * (loss_nce_av + loss_nce_va)
    else:
        loss_gdt = loss_nce_av
    return loss_gdt


def nce_crop_losses_dual(
    feats_v, 
    feats_v2, 
    XE, 
    s_large_crops=1, 
    s_small_crops=0, 
    t_large_crops=1, 
    t_small_crops=0, 
    temp=0.1
):
    assert (s_large_crops <= 2) and (t_large_crops <= 2)
    loss_big = torch.tensor(0.0).cuda()
    loss_small = torch.tensor(0.0).cuda()
    t_loss_big = torch.tensor(0.0).cuda()
    t_loss_small = torch.tensor(0.0).cuda()
    counter = [0] * 4
    if (s_small_crops > 0) or (s_large_crops > 1):
        for i in range(s_large_crops):
            large_crop = feats_v[0][0][i]
            if i == 1:
                loss_big += gdt_loss(
                    feats_v2[0][0][i], 
                    large_crop, 
                    XE=XE, 
                    temp=temp
                )
                counter[0] += 1
            for j in range(s_small_crops):
                small_crop = feats_v2[0][1][j]
                loss_small += gdt_loss(
                    large_crop.detach(), 
                    small_crop, XE=XE, 
                    symmetric=False, 
                    temp=temp
                )
                counter[1] += 1

    if (t_small_crops > 0) or (t_large_crops > 1):
        for ti in range(t_large_crops):
            large_tcrop = feats_v[1][0][ti]
            if ti == 1:
                t_loss_big += gdt_loss(
                    feats_v2[1][0][ti], 
                    large_tcrop, 
                    XE=XE, 
                    temp=temp
                )
                counter[2] += 1
            for tj in range(t_small_crops):
                small_tcrop = feats_v2[1][1][tj]
                t_loss_small +=  gdt_loss(
                    large_tcrop.detach(), 
                    small_tcrop, 
                    XE=XE, 
                    symmetric=False, 
                    temp=temp
                )
                counter[3] += 1
    return (
        [loss_big, loss_small, t_loss_big, t_loss_small], 
        counter
    )


def fmcrop_params(
    duration=4, 
    s_large_crops=1, 
    s_small_crops=0, 
    t_large_crops=1, 
    t_small_crops=0
):
    assert (s_large_crops <= 2) and (t_large_crops <= 2)
    crop_locs = [[],[]]
    tcrop_locs = [[],[]]
    if (s_small_crops > 0) or (s_large_crops > 1):
        for i in range(s_large_crops):
            large_crop = get_fm_crop(
                spatial=True, large=True, duration=duration)
            crop_locs[0].append(large_crop)
            for j in range(s_small_crops):
                small_crop = get_fm_crop(
                    spatial=True, large=False, duration=duration)
                crop_locs[1].append(small_crop)

    if (t_small_crops > 0) or (t_large_crops > 1):
        for ti in range(t_large_crops):
            large_tcrop = get_fm_crop(
                spatial=False, large=True, duration=duration)
            tcrop_locs[0].append(large_tcrop)
            for tj in range(t_small_crops):
                small_tcrop = get_fm_crop(
                    spatial=False, large=False, duration=duration)
                tcrop_locs[1].append(small_tcrop)
    return [crop_locs, tcrop_locs]


def get_fm_crop(spatial, large, duration=4):
    if spatial:
        if large:
            _x_window = 6
            _y_window = 6
        else:
            _x_window = 4
            _y_window = 4
        xmin = np.random.randint(0, 7 - _x_window)
        xmax = xmin + _x_window
        ymin = np.random.randint(0, 7 - _y_window)
        ymax = ymin + _y_window
        return torch.tensor([xmin,xmax,ymin,ymax])
    else:
        if large:
            _window = 3 if duration == 4 else 6
        else:
            _window = 2 if duration == 4 else 4
        tmin = np.random.randint(0, duration - _window)
        tmax = tmin + _window
        return torch.tensor([tmin,tmax])


if __name__ == "__main__":
    main()
