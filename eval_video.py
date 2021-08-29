#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
import datetime
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Custom imports
from src.scheduler import GradualWarmupScheduler
from utils import (
    AverageMeter,
    accuracy,
    aggregrate_video_accuracy,
    initialize_exp,
    getLogger,
    accuracy,
    save_checkpoint,
    load_model_parameters
)
from datasets.AVideoDataset import AVideoDataset
from model import load_model, Identity

logger = getLogger()


# DICT with number of classes for each  dataset
NUM_CLASSES = {
    'hmdb51': 51,
    'ucf101': 101,
    'kinetics400': 400
}


# Create Finetune Model
class Finetune_Model(torch.nn.Module):
    def __init__(
        self, 
        base_arch, 
        num_ftrs=512, 
        num_classes=101, 
        use_dropout=False, 
        use_bn=False, 
        use_l2_norm=False, 
        dropout=0.9
    ):
        super(Finetune_Model, self).__init__()
        self.base = base_arch
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm

        message = 'Classifier to %d classes;' % (num_classes)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_bn: message += ' + final BN'
        print(message)

        if self.use_bn:
            print("Adding BN to Classifier")
            self.final_bn = nn.BatchNorm1d(num_ftrs)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.linear_layer = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.linear_layer)
            self.classifier = torch.nn.Sequential(
                self.final_bn,
                self.linear_layer
            )
        else:
            self.classifier = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.classifier)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
    
    def forward(self, x):
        x = self.base(x).squeeze()
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


class Finetune_Model_Agg(torch.nn.Module):
    def __init__(
        self, 
        base_arch, 
        pooling_arch, 
        num_ftrs=512, 
        num_classes=101, 
        use_dropout=False, 
        use_bn=False, 
        use_l2_norm=False, 
        dropout=0.9,
    ):
        super(Finetune_Model_Agg, self).__init__()
        self.base = base_arch
        self.pooling_arch = pooling_arch
        self.num_chunk = 2
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm


        message = 'Classifier to %d classes;' % (num_classes)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_bn: message += ' + final BN'
        print(message)

        if self.use_bn:
            print("Adding BN to Classifier")
            self.final_bn = nn.BatchNorm1d(num_ftrs)
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
            self.linear_layer = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.linear_layer)
            self.classifier = torch.nn.Sequential(
                self.final_bn,
                self.linear_layer
            )
        else:
            self.classifier = torch.nn.Linear(num_ftrs, num_classes)
            self._initialize_weights(self.classifier)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
    
    def forward(self, x):
        # Encode
        x = self.base(x).squeeze()

        # Pooling
        x = self.pooling_arch(x)

        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


# Load finetune model and training params
def load_model_finetune(
    args, model, num_ftrs, num_classes, agg_model=False, 
    pooling_arch=None, use_dropout=False, use_bn=False, 
    use_l2_norm=False, dropout=0.9,
):
    if agg_model:
        print('Using Stica model')
        new_model = Finetune_Model_Agg(
            model, 
            pooling_arch, 
            num_ftrs, 
            num_classes, 
            use_dropout=use_dropout, 
            use_bn=use_bn, 
            use_l2_norm=use_l2_norm, 
            dropout=dropout,
        )
    else:
        print('Using non-agg GDT model')
        new_model = Finetune_Model(
            model, 
            num_ftrs, 
            num_classes, 
            use_dropout=use_dropout, 
            use_bn=use_bn, 
            use_l2_norm=use_l2_norm, 
            dropout=dropout
        )
    return new_model


def main(args, writer):

    # Create Logger
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True

    # Load model
    logger.info("Loading model")
    model = load_model(
        model_type=args.model,
        vid_base_arch=args.vid_base_arch,
        aud_base_arch=args.aud_base_arch,
        pretrained=args.pretrained,
        norm_feat=False,
        use_mlp=args.use_mlp,
        num_classes=256,
        args=args,
    )

    # Load model weights
    weight_path_type = type(args.weights_path)
    if weight_path_type == str:
        weight_path_not_none = args.weights_path != 'None'
    else:
        weight_path_not_none = args.weights_path is not None
    if not args.pretrained and weight_path_not_none:
        logger.info("Loading model weights")
        if os.path.exists(args.weights_path):
            ckpt_dict = torch.load(args.weights_path)
            try:
                model_weights = ckpt_dict["state_dict"]
            except:
                model_weights = ckpt_dict["model"]
            epoch = ckpt_dict["epoch"]
            logger.info(f"Epoch checkpoint: {epoch}")
            load_model_parameters(model, model_weights)
    logger.info(f"Loading model done")

    # Add FC layer to model for fine-tuning or feature extracting
    model = load_model_finetune(
        args,
        model.video_network.base,
        pooling_arch=model.video_pooling if args.agg_model else None,
        num_ftrs=model.encoder_dim,
        num_classes=NUM_CLASSES[args.dataset],
        use_dropout=args.use_dropout, 
        use_bn=args.use_bn,
        use_l2_norm=args.use_l2_norm,
        dropout=0.9,
        agg_model=args.agg_model,
    )

    # Create DataParallel model
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    # Get params for optimization
    params = []
    if args.feature_extract: # feature_extract only classifer
        logger.info("Getting params for feature-extracting")
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {
                    'params': param, 
                    'lr': args.head_lr, 
                    'weight_decay': args.weight_decay
                })
    else: # finetune
        logger.info("Getting params for finetuning")
        for name, param in model_without_ddp.classifier.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {
                    'params': param, 
                    'lr': args.head_lr, 
                    'weight_decay': args.weight_decay
                })
        for name, param in model_without_ddp.base.named_parameters():
            logger.info((name, param.shape))
            params.append(
                {   
                    'params': param, 
                    'lr': args.base_lr, 
                    'weight_decay': args.wd_base
                })
        if args.agg_model:
            logger.info("Adding pooling arch params to be optimized")
            for name, param in model_without_ddp.pooling_arch.named_parameters():
                if param.requires_grad and param.dim() >= 1:
                    logger.info(f"Adding {name}({param.shape}), wd: {args.wd_tsf}")
                    params.append(
                        {
                            'params': param, 
                            'lr': args.tsf_lr, 
                            'weight_decay': args.wd_tsf
                        })
                else:
                    logger.info(f"Not adding {name} to be optimized")


    logger.info('\n===========Check Grad============')
    for name, param in model_without_ddp.named_parameters():
        logger.info((name, param.requires_grad))
    logger.info('=================================\n')

    logger.info("Creating AV Datasets")
    dataset = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='train',
        num_train_clips=args.train_clips_per_video,
        decode_audio=False,
        center_crop=False,
        fold=args.fold,
        ucf101_annotation_path=args.ucf101_annotation_path,
        hmdb51_annotation_path=args.hmdb51_annotation_path,
        args=args,
    )
    dataset_test = AVideoDataset(
        ds_name=args.dataset,
        root_dir=args.root_dir,
        mode='test',
        decode_audio=False,
        num_spatial_crops=args.num_spatial_crops,
        num_ensemble_views=args.val_clips_per_video,
        ucf101_annotation_path=args.ucf101_annotation_path,
        hmdb51_annotation_path=args.hmdb51_annotation_path,
        fold=args.fold,
        args=args,
    )

    # Creating dataloaders
    logger.info("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=True,
        shuffle=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=False
    )

    # linearly scale LR and set up optimizer
    logger.info(f"Using SGD with lr: {args.head_lr}, wd: {args.weight_decay}")
    optimizer = torch.optim.SGD(
        params,
        lr=args.head_lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    # Multi-step LR scheduler
    if args.use_scheduler:
        milestones = [int(lr) - args.lr_warmup_epochs for lr in args.lr_milestones.split(',')]
        logger.info(f"Num. of Epochs: {args.epochs}, Milestones: {milestones}")
        if args.lr_warmup_epochs > 0:
            logger.info(f"Using scheduler with {args.lr_warmup_epochs} warmup epochs")
            scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
            lr_scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=8,
                total_epoch=args.lr_warmup_epochs, 
                after_scheduler=scheduler_step
            )
        else: # no warmp, just multi-step
            logger.info("Using scheduler w/out warmup")
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=args.lr_gamma
            )
    else:
        lr_scheduler = None

    # Checkpointing
    if args.resume:
        ckpt_path = os.path.join(args.output_dir, 'checkpoints', 'checkpoint.pth')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        logger.info(f"Resuming from epoch: {args.start_epoch}")

    # Only perform evalaution
    if args.test_only:
        scores_val = evaluate(
            model, 
            data_loader_test,
            epoch=args.start_epoch, 
            writer=writer,
            ds=args.dataset,
        )
        _, vid_acc1, vid_acc5 = scores_val
        return vid_acc1, vid_acc5, args.start_epoch

    start_time = time.time()
    best_vid_acc_1 = -1
    best_vid_acc_5 = -1
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f'Start training epoch: {epoch}')
        scores = train(
            model, 
            optimizer, 
            data_loader,
            epoch, 
            writer=writer,
            ds=args.dataset,
        )
        logger.info(f'Start evaluating epoch: {epoch}')
        lr_scheduler.step()
        if (epoch % 1 == 0) and epoch > 6:
            scores_val = evaluate(
                model, 
                data_loader_test,
                epoch=epoch,
                writer=writer,
                ds=args.dataset,
            )
            _, vid_acc1, vid_acc5 = scores_val
            training_stats.update(scores + scores_val)
            if vid_acc1 > best_vid_acc_1:
                best_vid_acc_1 = vid_acc1
                best_vid_acc_5 = vid_acc5
                best_epoch = epoch
        if args.output_dir:
            logger.info(f'Saving checkpoint to: {args.output_dir}')
            save_checkpoint(args, epoch, model, optimizer, lr_scheduler, ckpt_freq=1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')
    return best_vid_acc_1, best_vid_acc_5, best_epoch


def train(
    model, 
    optimizer, 
    loader, 
    epoch, 
    writer=None,
    ds='hmdb51',
):
    # Put model in train mode
    model.train()

    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    criterion = nn.CrossEntropyLoss().cuda()

    for it, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update iteration
        iteration = epoch * len(loader) + it

        # forward
        video, target, _, _ = batch
        video, target = video.cuda(), target.cuda()
        output = model(video)

        # compute cross entropy loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), video.size(0))
        top1.update(acc1[0], video.size(0))
        top5.update(acc5[0], video.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    it,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

            writer.add_scalar(
                f'{ds}/train/loss/iter', 
                losses.val, 
                iteration
            )
            writer.add_scalar(
                f'{ds}/train/clip_acc1/iter', 
                top1.val, 
                iteration
            )
    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def evaluate(model, val_loader, epoch=0, writer=None, ds='hmdb51'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    # dicts to store labels and softmaxes
    softmaxes = {}
    labels = {}

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for batch_idx, batch in enumerate(val_loader):

            (video, target, _, video_idx) = batch

            # move to gpu
            video = video.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output and loss
            output = model(video)
            loss = criterion(output.view(video.size(0), -1), target)

            # Clip level accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), video.size(0))
            top1.update(acc1[0], video.size(0))
            top5.update(acc5[0], video.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # Video Level accuracy
            for j in range(len(video_idx)):
                video_id = video_idx[j].item()
                sm = output[j]
                label = target[j]

                # append it to video dict
                softmaxes.setdefault(video_id, []).append(sm)
                labels[video_id] = label
    
    # Get video acc@1 and acc@5 and output to tb writer
    video_acc1, video_acc5 = aggregrate_video_accuracy(
        softmaxes, labels, topk=(1, 5)
    )

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "ClipAcc@1 {top1.avg:.3f}\t"
            "VidAcc@1 {video_acc1:.3f}".format(
                batch_time=batch_time, loss=losses, top1=top1, 
                video_acc1=video_acc1.item()))

        writer.add_scalar(
            f'{ds}/val/vid_acc1/epoch', 
            video_acc1.item(), 
            epoch
        )
        writer.add_scalar(
            f'{ds}/val/vid_acc5/epoch', 
            video_acc5.item(), 
            epoch
        )

    # Log final results to terminal
    return losses.avg, video_acc1.item(), video_acc5.item()


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
    parser = argparse.ArgumentParser(description='Video Action Finetune')
    parser.register('type', 'bool', str2bool)

    ### DATA
    parser.add_argument('--dataset', default='hmdb51', type=str, 
                        help='name of dataset')
    parser.add_argument('--fold', default='1', type=str,
                        help='name of dataset')
    parser.add_argument('--root_dir', default=None, 
                        type=str, help='name of dataset')
    parser.add_argument('--ucf101-annotation-path', default='/datasets01/ucf101/112018/ucfTrainTestlist/', 
                        type=str, help='name of dataset')
    parser.add_argument('--hmdb51-annotation-path', default='/datasets01/hmdb51/112018/splits/', 
                        type=str, help='name of dataset')
    parser.add_argument('--target-fps', type=int, default=30,
                        help='video fps')
    parser.add_argument('--train-crop-size', type=int, default=128,
                        help="train crop size")
    parser.add_argument('--test-crop-size', type=int, default=128,
                        help="train crop size")
    parser.add_argument('--multi-crop', type='bool', default='False',
                        help='do multi-crop comparisons')
    parser.add_argument('--num-large-crops', type=int, default=1,
                        help='Number of Large Crops')
    parser.add_argument('--num-small-crops', type=int, default=0,
                        help='Number of small Crops')
    parser.add_argument('--use-grayscale', type='bool', default='False',
                        help='use grayscale augmentation')
    parser.add_argument('--use-gaussian', type='bool', default='False',
                        help='use gaussian augmentation')
    parser.add_argument('--clip-len', default=32, type=int, 
                        help='number of frames per clip')
    parser.add_argument('--colorjitter', default='True', type='bool', 
                        help='scale jittering as augmentations')
    parser.add_argument('--steps-bet-clips', default=1, type=int, 
                        help='number of steps between clips in video')
    parser.add_argument('--num-data-samples', default=None, type=int, 
                        help='number of samples in dataset')
    parser.add_argument('--train-clips-per-video', default=10, type=int, 
                        help='maximum number of clips per video to consider for training')
    parser.add_argument('--val-clips-per-video', default=10, type=int, 
                        help='maximum number of clips per video to consider for testing')
    parser.add_argument('--num-spatial-crops', default=3, type=int, 
                        help='number of spatial clips for testing')
    parser.add_argument('--test-time-cj', default='False', type='bool', 
                        help='test time CJ augmentation')
    parser.add_argument('--workers', default=16, type=int, 
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--use_random_resize_crop', default='True', type='bool', 
                        help='use random resized crop instead of short stide jitter')

    ### MODEL
    parser.add_argument('--weights-path', default='', type=str,
                        help='Path to weights file')
    parser.add_argument('--ckpt-epoch', default='0', type=str,
                        help='Epoch of model checkpoint')
    parser.add_argument('--model', default='av_gdt', help='model',
        choices=['av_gdt', 'vid_text_gdt', 'stica'])
    parser.add_argument('--vid-base-arch', default='r2plus1d_18', type=str,
                        help='Video Base Arch for A-V model',
                        choices=['r2plus1d_18', 'r2plus1d_34'])
    parser.add_argument('--aud-base-arch', default='resnet9', 
                        help='Audio Base Arch for A-V model',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet9'])
    parser.add_argument('--pretrained', default='False', type='bool', 
                        help='Use pre-trained models from the modelzoo')
    parser.add_argument('--supervised', default='False', type='bool', 
                        help='Use supervised model')
    parser.add_argument('--use-mlp', default='False', type='bool', 
                        help='Use MLP projection head')
    parser.add_argument('--mlptype', default=0, type=int,
                        help='MLP type (default: 0)')
    parser.add_argument('--headcount', default=1, type=int, 
                        help='how many heads each modality has')
    parser.add_argument('--use-dropout', default='False', type='bool', 
                        help='Use dropout in classifier')
    parser.add_argument('--use-bn', default='False', type='bool', 
                        help='Use BN in classifier')
    parser.add_argument('--use-l2-norm', default='False', type='bool', 
                        help='Use L2-Norm in classifier')
    parser.add_argument('--agg-model', default='False', type='bool', 
                        help="Aggregate model with transformer")
    parser.add_argument('--num_layer', default=2, type=int,
                        help='num of transformer layers')
    parser.add_argument('--num_sec', default=2, type=int, 
                        help='num of seconds')
    parser.add_argument('--dp', default=0.0, type=float, 
                        help='dropout rate in transformer')
    parser.add_argument('--num_head', default=4, type=int,
                        help='num head in transformer')
    parser.add_argument('--use_larger_last', type='bool', default='False',
                        help='use larger last layer of res5')

    ### TRANSFORMER PARAMS
    parser.add_argument('--positional_emb', default='False', type='bool', 
                        help="use positional emb in transformer")   
    parser.add_argument('--qkv_mha', default='False', type='bool', 
                        help='complete qkv in MHA')
    parser.add_argument('--cross_modal_nce', default='True', type='bool', 
                        help='use cross-modal NCE loss')
    parser.add_argument('--fm_crop', type='bool', default='False',
                        help='use FMCROP model')
    parser.add_argument('--transformer_time_dim', default=8, type=int, 
                        help='temporal input for transformer')
    parser.add_argument('--cross_modal_alpha', type=float, default=0.5,
                        help='weighting of cross-modal loss') 

    ### TRAINING
    parser.add_argument('--feature-extract', default='False', type='bool', 
                        help='Use model as feature extractor;')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='effective batch size')
    parser.add_argument('--epochs', default=12, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--optim-name', default='sgd', type=str, 
                        help='Name of optimizer')
    parser.add_argument('--head-lr', default=0.0025, type=float, 
                        help='initial learning rate')
    parser.add_argument('--base-lr', default=0.00025, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--weight-decay', default=0.005, type=float,
                        help='weight decay for classifier')
    parser.add_argument('--wd-base', default=0.005, type=float,
                        help='weight decay for bas encoder')
    parser.add_argument('--use-scheduler', default='True', type='bool', 
                        help='Use LR scheduler')
    parser.add_argument('--lr-warmup-epochs', default=2, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('--lr-milestones', default='6,10', type=str, 
                        help='decrease lr on milestones (epochs)')
    parser.add_argument('--lr-gamma', default=0.05, type=float, 
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--tsf_lr', default=0.00025, type=float, 
                        help='transformer learning rate')
    parser.add_argument('--wd_tsf', default=0.005, type=float, 
                        help='transformer wd')
    
    ### LOGGING
    parser.add_argument('--print-freq', default=10, type=int, 
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', type=str, 
                        help='path where to save')
    
    ### AUDIO
    parser.add_argument('--num-sec-aud', type=int, default=1,
                        help='number of seconds of audio')
    parser.add_argument('--aud-sample-rate', type=int, default=24000,
                        help='audio sample rate')
    parser.add_argument('--audio-augtype', type=str, default='none',
                        choices=['none', 'mild', 'medium', 'heavy'], 
                        help='audio augmentation strength with Spec Augment')
    parser.add_argument('--aud-spec-type', type=int, default=2,
                        help="audio spec type")
    parser.add_argument('--use-volume-jittering', type='bool', default='True',
                        help='use volume jittering')
    parser.add_argument('--use-audio-temp-jittering', type='bool', default='False',
                        help='use audio temporal jittering')
    parser.add_argument('--z-normalize', type='bool', default='False',
                        help='z-normalize the audio')

    ### CHECKPOINTING
    parser.add_argument('--resume', default='', type=str, 
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, 
                        help='start epoch')
    parser.add_argument('--test-only', default='False', type='bool', 
                        help='Only test the model')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.dump_path = args.output_dir
    args.rank = 0
    args.num_frames = args.clip_len
    args.sample_rate = args.steps_bet_clips
    args.agg_model = args.model == 'stica'
    logger.info(args)

    # Make output dir
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up tensorboard
    writer = writer = SummaryWriter(tbx_path)
    writer.add_text("namespace", repr(args))
    
    # Number of seconds
    if args.clip_len > 32:
        args.num_sec = int(args.clip_len / 30)
        args.transformer_time_dim = 8
    

    # Run over different folds
    best_accs_1 = []
    best_accs_5 = []
    best_epochs = []
    folds = [int(fold) for fold in args.fold.split(',')]
    print(f"Evaluating on folds: {folds}")
    if args.dataset in ['ucf101', 'hmdb51']:
        for fold in folds:
            args.fold = fold
            best_acc1, best_acc5, best_epoch = main(args, writer)
            best_accs_1.append(best_acc1)
            best_accs_5.append(best_acc5)
            best_epochs.append(best_epoch)
        avg_acc1 = np.mean(best_accs_1)
        avg_acc5 = np.mean(best_accs_5)
        logger.info(f'3-Fold ({args.dataset}): Vid Acc@1 {avg_acc1:.3f}, Video Acc@5 {avg_acc5:.3f}')
    else:
        best_acc1, best_acc5, best_epoch = main(args, writer)
        best_accs_1.append(best_acc1)
        best_accs_5.append(best_acc5)
        best_epochs.append(best_epoch)
        avg_acc1 = np.mean(best_accs_1)
        avg_acc5 = np.mean(best_accs_5)
        logger.info(f'Fold-{args.fold} ({args.dataset}): Vid Acc@1 {avg_acc1:.3f}, Video Acc@5 {avg_acc5:.3f}')
