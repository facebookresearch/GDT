import datetime
import os
import sys
import time

import random

import torch
import torch.distributed as dist
import torchvision

from warmup_scheduler_local.scheduler import GradualWarmupScheduler
import utils
from utils import print_or_log, dist_collect_other,dist_collect

try:
    from apex import amp
except ImportError:
    amp = None


def get_loss(q, k, noise_batch, t=0.07, device='cuda'):
    N, C = q.shape
    # l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1)  # positive logit N x 1
    l_pos = torch.einsum("nc,mc -> nm", [q.view(N, C), k.view(N, C)])  # positive Nx N s.t. positives are diagonals
    l_neg = torch.mm(q.view(N, C), noise_batch.transpose(0, 1))  # negative logit N x K
    labels = torch.arange(0, N, dtype=torch.long).to(device)  # positives are the 0-th
    logits = torch.cat([l_pos, l_neg], dim=1) / t
    prob = torch.mean((logits[:, 0] == logits.max(1)[0]).float()) * 100
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss, prob


def compute_feats(model, video1, audio1, video2=None, audio2=None, feats1=None):
    # Perform forwards
    feat_v, feat_a = model(video1, audio1) if feats1 is None else feats1
    if video2 is None:
        return feat_v, feat_a
    feat_vT, feat_aT = model(video2, audio2)
    return (feat_v, feat_a), (feat_vT, feat_aT)


@torch.no_grad()
def get_pos_neg(hyp, feats1, feats1_col=None, feats2=None, feats2_col=None, concats=None):
    # deal only with feats1
    feat_v, feat_a = feats1
    if feats2 is not None:
        feat_vT, feat_aT = feats2
    
    # Get transformation type
    if hyp[0] == 'i':
        transf = "invariant"
    elif hyp[0] == 'v':
        transf = "variant"
    else:
        transf = "basecase"

    # Get POS keys: keys are cross-modal (video, audio) and (audio, video)
    if transf == "invariant":
        feat_v_pos, feat_a_pos = feat_aT.detach(), feat_vT.detach()
    elif transf == "variant" or transf == "basecase":
        feat_v_pos, feat_a_pos = feat_a.detach(), feat_v.detach()

    # if distributed: get all other videos in batch (collated from other GPUs) are the default negatives (cross-modal)
    if feats1_col is not None:
        feat_v_col, feat_a_col = feats1_col
        if concats is None:
            if transf == "invariant" and feats2_col is not None:
                feat_vT_col, feat_aT_col = feats2_col
                feat_a_neg = feat_vT_col
                feat_v_neg = feat_aT_col
                if feats2_col is not None:
                    # add transformed half of the batch to negatives too
                    feat_a_neg.extend(feat_v_col) # now vT,v
                    feat_v_neg.extend(feat_a_col) # now aT,a

            elif transf == "variant" or transf == "basecase":
                feat_a_neg = feat_v_col
                feat_v_neg = feat_a_col
                if feats2_col is not None:
                    feat_vT_col, feat_aT_col = feats2_col
                    feat_a_neg.extend(feat_vT_col) # now v,vT
                    feat_v_neg.extend(feat_aT_col) # now a,aT
            feat_a_neg  = torch.cat(feat_a_neg, dim=0)
            feat_v_neg  = torch.cat(feat_v_neg, dim=0)
            concats = (feat_v_neg, feat_a_neg)
        else:
            (feat_v_neg, feat_a_neg) = concats

    if transf == 'variant':
        feat_a_neg  = torch.cat([feat_a_neg, feat_vT]) # now a,aT, aT_local
        feat_v_neg  = torch.cat([feat_v_neg, feat_aT]) # now v,vT, vT_local

    # Get a subset of negatives to compare to
    if args.num_negatives != -1:
        feat_a_neg, feat_v_neg = utils.reduce_negatives(feat_a_neg, feat_v_neg, args.num_negatives)

    pairs = [feat_v, feat_a, feat_v_pos, feat_a_pos, feat_v_neg, feat_a_neg, concats]
    return pairs


def get_losses(pairs1, pairs2, hyp='basecase'):
    video_loss1, prob_vid1 = get_loss(
        pairs1[0], # v_i
        pairs1[2], # a_i
        pairs1[4], # Ba_j (and maybe hard-neg)
        t=args.nce_t,
    )
    audio_loss1, prob_aud1 = get_loss(
        pairs1[1], # a_i
        pairs1[3], # v_i
        pairs1[5], # Bv_j (and maybe hard-neg)
        t=args.nce_t,
    )
    loss = 0.5 * video_loss1 + 0.5 * audio_loss1
    if pairs2:
        video_loss2, prob_vid2 = get_loss(
            pairs2[0],  # Tv_i
            pairs2[2],  # Ta_i
            pairs2[4],  # TBa_j (and maybe hard-neg)
            t=args.nce_t,
        )
        audio_loss2, prob_aud2 = get_loss(
            pairs2[1],  # Ta_i
            pairs2[3],  # Tv_i
            pairs2[5],  # TBv_j (and maybe hard-neg)
            t=args.nce_t,
        )
        # video_loss = video_loss1 + video_loss2
        # audio_loss = audio_loss1 + audio_loss2
        # prob_vid = 0.5 * (prob_vid1 + prob_vid2)
        # prob_aud = 0.5 * (prob_aud1 + prob_aud2)
        loss = 0.25 * video_loss1 + 0.25 * audio_loss1 + 0.25 * video_loss2 + 0.25 * audio_loss2
    # else:
    #     video_loss = video_loss1
    #     audio_loss = audio_loss1
    #     prob_vid = prob_vid1
    #     prob_aud = prob_aud1
    # loss_dict = {
    #     f'{hyp}_video_loss': video_loss.item(),
    #     f'{hyp}_video_loss_og': video_loss1.item(),
    #     f'{hyp}_prob_vid':prob_vid.item(),
    #     f'{hyp}_prob_vid_og':prob_vid1.item(),
    #     f'{hyp}_audio_loss': audio_loss.item(),
    #     f'{hyp}_audio_loss_og': audio_loss1.item(),
    #     f'{hyp}_prob_aud':prob_aud.item(),
    #     f'{hyp}_prob_aud_og':prob_aud1.item(),
    # }
    # if pairs2:
        # loss_dict[f'{hyp}_video_loss_tsf'] = video_loss2.item()
        # loss_dict[f'{hyp}_prob_vid_tsf'] = prob_vid2.item()
        # loss_dict[f'{hyp}_audio_loss_tsf'] =  audio_loss2.item()
        # loss_dict[f'{hyp}_prob_aud_tsf'] = prob_aud2.item()
    return loss, None


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
    metric_logger = utils.MetricLoggerGDT(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for batch_idx, batch in metric_logger.log_every(data_loader, print_freq, header, logger, writer, 'train',
                                                    epoch=epoch):
        video, spec, label, vid_idx, idx = batch
        video, videoR, video2, video2R  = torch.split(video, [3, 3, 3, 3], dim=1)
        if args.decode_audio:
            audio, audioR, audio2, audio2R = torch.split(spec, [1, 1, 1, 1], dim=1)
        else: # text
            audio, audioR, audio2, audio2R = torch.split(spec, [20, 20, 20, 20], dim=1)
        if batch_idx == 0:
            print_or_log((video.shape, audio.shape), logger=logger)
        start_time = time.time()

        video, audio = video.to(device), audio.to(device)

        # form positive and negative pairs dependent on hypothesis
        if args.hypothesis == 1:
            hyp = 'basecase'

            # compute features
            feats1 = compute_feats(model, video, audio)
            feat_v, feat_a = feats1

            # collation on GPUs
            feat_v_col = dist_collect_other(feat_v.detach(), return_before_cat=True)
            feat_a_col = dist_collect_other(feat_a.detach(), return_before_cat=True)
            feats1_col = (feat_v_col, feat_a_col)

            # basecase cross-modal loss
            pairs1 = get_pos_neg(hyp, feats1, feats1_col)
            loss1, loss_dict1 = get_losses(pairs1, pairs2=None) # (V, A)
            loss = loss1
            loss_dict2 = None
        if args.hypothesis in [2, 3]:
            
            # Add Reversal pairs to GPU
            videoR, audioR = videoR.to(device), audioR.to(device)

            # compute features
            feats1, feats2 = compute_feats(model, video, audio, videoR, audioR)
            feat_v, feat_a = feats1
            feat_vR, feat_aR = feats2

            # collation on GPUs
            feat_v_col = dist_collect_other(feat_v.detach(), return_before_cat=True)
            feat_a_col = dist_collect_other(feat_a.detach(), return_before_cat=True)
            feat_vR_col = dist_collect_other(feat_vR.detach(), return_before_cat=True)
            feat_aR_col = dist_collect_other(feat_aR.detach(), return_before_cat=True)
            feats1_col = (feat_v_col, feat_a_col)
            feats2_col = (feat_vR_col, feat_aR_col)

            if args.hypothesis == 2:
                hyp = 'vtime'
                loss_b = None
            if args.hypothesis == 3:
                hyp = 'itime'
                # base loss
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats2, feats2_col) # Pos: (V, A), Neg: Other (V, A) + (V, AT) except from local 
                pairs2 = get_pos_neg('basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A) except from local
                loss_b, loss_dict_b = get_losses(pairs1, pairs2)

            # time-reversal
            pairs1 = get_pos_neg(hyp, feats1, feats1_col, feats2, feats2_col) # Inv: Pos: (V, AT) Var: Pos: (V, A), Neg: Other (V, A) + (V, AT)
            pairs2 = get_pos_neg(hyp, feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # Inv: Pos: (VT, A) Var: Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A)
            loss1, loss_dict1 = get_losses(pairs1, pairs2)
            loss = 0.5*(loss1 + loss_b) if loss_b is not None else loss1
            loss_dict2 = None
        if args.hypothesis in [4, 5]:
            
            # Add time shift pairs to GPU
            video2, audio2 = video2.to(device), audio2.to(device)

            # compute features
            feats1, feats2 = compute_feats(model, video, audio, video2, audio2)
            feat_v, feat_a = feats1
            feat_v2, feat_a2 = feats2

            # collation on GPUs
            feat_v_col = dist_collect_other(feat_v.detach(), return_before_cat=True)
            feat_a_col = dist_collect_other(feat_a.detach(), return_before_cat=True)
            feat_v2_col = dist_collect_other(feat_v2.detach(), return_before_cat=True)
            feat_a2_col = dist_collect_other(feat_a2.detach(), return_before_cat=True)
            feats1_col = (feat_v_col, feat_a_col)
            feats2_col = (feat_v2_col, feat_a2_col)

            if args.hypothesis == 4:
                hyp = 'vasync'
                loss_b = None
            if args.hypothesis == 5:
                hyp = 'iasync'
                # base loss
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats2, feats2_col) # Pos: (V, A), Neg: Other (V, A) + (V, AT) except from local 
                pairs2 = get_pos_neg('basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A) except from local
                loss_b, loss_dict_b = get_losses(pairs1, pairs2)
            
            # time-shift
            pairs1 = get_pos_neg(hyp, feats1, feats1_col, feats2, feats2_col) # Inv: Pos: (V, AT) Var: Pos: (V, A), Neg: Other (V, A) + (V, AT)
            pairs2 = get_pos_neg(hyp, feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # Inv: Pos: (VT, A) Var: Pos: (VT, AT), Neg: Other (VT, AT) + (VT, A)
            loss1, loss_dict1 = get_losses(pairs1, pairs2)
            loss = 0.5*(loss1 + loss_b) if loss_b is not None else loss1
            loss_dict2 = None

        elif args.hypothesis >= 6:

            # Add time reversal and shift pairs to GPU
            videoR, audioR = videoR.to(device), audioR.to(device)
            video2, audio2 = video2.to(device), audio2.to(device)
            video2R, audio2R = video2R.to(device), audio2R.to(device)

            # compute features
            feats1 = compute_feats(model, video, audio)
            feats2 = compute_feats(model, video2, audio2)
            feats3 = compute_feats(model, videoR, audioR)
            feats4 = compute_feats(model, video2R, audio2R)
            feat_v, feat_a = feats1
            feat_v2, feat_a2 = feats2
            feat_vR, feat_aR = feats3
            feat_v2R, feat_a2R = feats4

            # collation on GPUs
            feat_v_col = dist_collect_other(feat_v.detach(), return_before_cat=True)
            feat_a_col = dist_collect_other(feat_a.detach(), return_before_cat=True)
            feat_v2_col = dist_collect_other(feat_v2.detach(), return_before_cat=True)
            feat_a2_col = dist_collect_other(feat_a2.detach(), return_before_cat=True)
            feat_vR_col = dist_collect_other(feat_vR.detach(), return_before_cat=True)
            feat_aR_col = dist_collect_other(feat_aR.detach(), return_before_cat=True)
            feat_v2R_col = dist_collect_other(feat_v2R.detach(), return_before_cat=True)
            feat_a2R_col = dist_collect_other(feat_a2R.detach(), return_before_cat=True)
            feats1_col = (feat_v_col, feat_a_col)
            feats2_col = (feat_v2_col, feat_a2_col)
            feats3_col = (feat_vR_col, feat_aR_col)
            feats4_col = (feat_v2R_col, feat_a2R_col)

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
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats3, feats3_col) # (V, A)
                pairs2 = get_pos_neg('basecase', feats3, feats3_col, feats1, feats1_col, pairs1[-1]) # (VR, AR)
                loss_b1, loss_dict_b = get_losses(pairs1, pairs2)
            else:
                loss_b1 = None

            if hyp2 == 'iasynced':
                # basecase iasync
                pairs1 = get_pos_neg('basecase', feats1, feats1_col, feats2, feats2_col) # (V, A)
                pairs2 = get_pos_neg('basecase', feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # (VR, AR)
                loss_b2, loss_dict_b = get_losses(pairs1, pairs2)
            else:
                loss_b2 = None

            # time-reversal ##########################################################
            # on normal
            # feats1, feats2 = compute_feats(model, video, audio, videoR, audioR)
            # pairs1 = get_pos_neg(hyp1, feats1, feats2)
            # pairs2 = get_pos_neg(hyp1, feats2, feats1)
            pairs1 = get_pos_neg(hyp1, feats1, feats1_col, feats3, feats3_col) # (V, AR)
            pairs2 = get_pos_neg(hyp1, feats3, feats3_col, feats1, feats1_col, pairs1[-1]) # (VR, A)
            loss1, loss_dict1 = get_losses(pairs1, pairs2)

            ## on time-shifted:
            # feats1, feats2 = compute_feats(model, video2, audio2, video2R, audio2R)
            # pairs1 = get_pos_neg(hyp1, feats1, feats2)
            # pairs2 = get_pos_neg(hyp1, feats2, feats1)
            pairs1 = get_pos_neg(hyp1, feats2, feats2_col, feats4, feats4_col) # (V2, A2R)
            pairs2 = get_pos_neg(hyp1, feats4, feats4_col, feats2, feats2_col, pairs1[-1]) # (V2R, A2)
            loss2, loss_dict2 = get_losses(pairs1, pairs2)
            ##########################################################################

            # time-shift #############################################################
            ## on normal
            # feats1, feats2 = compute_feats(model, video, audio, video2, audio2)
            # pairs1 = get_pos_neg(hyp2, feats1, feats2)
            # pairs2 = get_pos_neg(hyp2, feats2, feats1)
            pairs1 = get_pos_neg(hyp2, feats1, feats1_col, feats2, feats2_col) # (V, A2)
            pairs2 = get_pos_neg(hyp2, feats2, feats2_col, feats1, feats1_col, pairs1[-1]) # (V2, A)
            loss3, loss_dict3 = get_losses(pairs1, pairs2)

            ## on time-reversed
            # feats1, feats2 = compute_feats(model, videoR, audioR, video2R, audio2R)
            # pairs1 = get_pos_neg(hyp2, feats1, feats2)
            # pairs2 = get_pos_neg(hyp2, feats2, feats1)
            pairs1 = get_pos_neg(hyp2, feats3, feats3_col, feats4, feats4_col) # (VR, A2R)
            pairs2 = get_pos_neg(hyp2, feats4, feats4_col, feats3, feats3_col, pairs1[-1]) # (V2R, AR)
            loss4, loss_dict4 = get_losses(pairs1, pairs2, hyp=hyp2)
            ##########################################################################

            # combine losses
            if loss_b1 is not None and loss_b2 is not None:
                loss = (1./6)*(loss1 + loss2 + loss3 + loss4 + loss_b1 + loss_b2)
            elif loss_b1 is not None:
                loss = 0.2*(loss1 + loss2 + loss3 + loss4 + loss_b1)
            elif loss_b2 is not None:
                loss = 0.2*(loss1 + loss2 + loss3 + loss4 + loss_b2)
            else:
                loss = 0.25*(loss1 + loss2 + loss3 + loss4 )

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
                print_or_log("Beginning reqeue", logger=logger)
                utils.trigger_job_requeue(os.path.join(args.output_dir, 'checkpoints', 'checkpoint.pth'))

        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # for key in loss_dict1.keys():
        #     metric_logger.meters[key].update(loss_dict1[key], n=batch_size)
        # if loss_dict2 is not None:
        #     for key in loss_dict2.keys():
        #         metric_logger.meters[key].update(loss_dict2[key], n=batch_size)
        metric_logger.meters['batch_t/s'].update((time.time() - start_time))
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
    if args.distributed:
        dist.barrier()
    torch.cuda.empty_cache()
    return metric_logger.loss.avg


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
        utils.mkdir(args.output_dir)

    # Init distributed mode
    if torch.cuda.is_available():
        utils.init_distributed_mode(args)

    # init signal handler
    utils.init_signal_handler()

    # Set up logger
    logger = None
    if args.distributed:
        filename = str(args.job_id) + '_' + str(args.rank) + '_log.out'
        logger = utils.setup_logger(
            "Video_reader, classification",
            args.output_dir,
            True,
            logname=filename
        )

    # Set up tensorboard
    tbx_path = os.path.join(args.output_dir, 'tensorboard')
    global_rank = args.rank if args.distributed else 0
    is_master = True if global_rank == 0 else False
    writer = None
    if is_master:
        writer = utils.setup_tbx(
            tbx_path,
            is_master
        )
        writer.add_text("namespace", repr(args))

    # Log version information
    print_or_log(args, logger=logger)
    print_or_log(f"torch version: {torch.__version__}", logger=logger)
    print_or_log(f"torchvision version: {torchvision.__version__}", logger=logger)

    # Set distributed mode
    device = torch.device(args.device)

    # Set CudNN benchmark
    torch.backends.cudnn.benchmark = True

    # Create model
    print_or_log("Creating model", logger=logger)
    model = utils.load_model(
        model_name=args.model,
        vid_base_arch=args.vid_base_arch,
        aud_base_arch=args.aud_base_arch,
        pretrained=args.pretrained,
        norm_feat=args.norm_feat,
        use_mlp=args.use_mlp,
        mlptype=args.mlptype,
        headcount=1,
        use_max_pool=args.use_max_pool,
    )
    assert args.headcount == 1 # "old option, keep number heads to 1"
    model.to(device)
    if args.distributed and args.sync_bn:
        print_or_log("Sync BN on model", logger=logger)
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

    # Warm up batch-norm
    if args.warmup_bn and not args.resume:
        print_or_log(f'Warming up BN', logger=logger)
        dataset, _dl = utils.get_dataloader(args, 0)
        utils._warmup_batchnorm(args, model, dataset, device, batches=100)
        del dataset, _dl

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
                print_or_log(f'Using Multi-Step LR scheduler', logger=logger)
                scheduler_step = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                print_or_log(f'Using Cosine Annealing LR scheduler', logger=logger)
                scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
            lr_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=args.world_size,
                total_epoch=args.lr_warmup_epochs,
                after_scheduler=scheduler_step
            )
        else:
            if args.scheduler_type == 'multi_step':
                print_or_log(f'Using Multi-Step LR scheduler', logger=logger)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=milestones,
                    gamma=args.lr_gamma
                )
            else:
                print_or_log(f'Using Cosine Annealing LR scheduler', logger=logger)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Checkpointing restart
    ckp_path = os.path.join(args.output_dir, 'checkpoints', 'checkpoint.pth')
    if os.path.isfile(ckp_path):
        print_or_log(f'Loading checkpoint', logger=logger)
        checkpoint = torch.load(ckp_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch']
        print_or_log(f'Restrating at epoch {args.start_epoch}', logger=logger)

    # Create dataloader
    ds = utils.get_ds(args, 0)

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
        print_or_log(f'Start training epoch: {epoch}', logger=logger)
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
            logger=logger,
            writer=writer,
        )
        if lr_scheduler:
            lr_scheduler.step()
        if args.output_dir:
            utils.save_checkpoint(args, epoch, model, optimizer, lr_scheduler)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_or_log(f'Training time {total_time_str}', logger=logger)


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
    parser.add_argument('--dataset', default='kinetics', type=str,
        help='name of dataset')
    parser.add_argument('--dualdata', default='True', type='bool',
        help='use dataloader that returns two samples per video')
    parser.add_argument('--num-data-samples', default=None, type=int,
        help='number of samples in dataset')
    parser.add_argument('--fold', default=1, type=str,
        help='fold of dataset (ucf101/ hmdb51)')
    parser.add_argument('--workers', default=0, type=int,
        help='number of data loading workers (default: 16)')

    # GDT NCE loss
    parser.add_argument('--hypothesis', default=1, type=int,
        help="use it for encoding what learning hypothesis we're using")
    parser.add_argument('--nce-t', default=0.07, type=float, 
        help='softmax weighting')
    parser.add_argument('--num-negatives', default=-1, type=int,
        help='number of negatives in contrastive loss')

    # Video Augmentations
    parser.add_argument('--clip-len', default=30, type=int,
        help='number of frames per clip')
    parser.add_argument('--target-fps', default=30, type=int,
        help='target fps')
    parser.add_argument('--sample-rate', default=1, type=int,
        help='Subsampling rate: num frames between clips')
    parser.add_argument('--clips-per-video', default=1, type=int,
        help='number of clips to sample from video')
    parser.add_argument('--train-crop-size', default=112, type=int,
        help='Size of spatial crops')
    parser.add_argument('--colorjitter', default='False', type='bool',
        help='Apply random color jitter')
    parser.add_argument('--use-scale-jittering', default='False', type='bool',
        help='scale jittering as augmentations')
    parser.add_argument('--augtype', default=1, type=int,
        help='augmentation type (default: 1)')
    parser.add_argument('--use-temp-jitter', default='True', type='bool',
        help='Get clips from random timestamps each epoch')
    parser.add_argument('--center-crop', default='False', type='bool',
        help='Use center cropping instead of random cropping')
    
    # Audio Augmentation
    parser.add_argument('--aud-sample-rate', default=24000, type=int,
        help='audio sample rate')
    parser.add_argument('--aud-spec-type', default=1, type=int,
        help='audio spec type') # 1 : (40, 99), (257, 199)
    parser.add_argument('--use-volume-jittering', default='False', type='bool',
        help='use volume jittering')
    parser.add_argument('--use-temporal-jittering', default='False', type='bool',
        help='use temporal jittering')
    parser.add_argument('--num-sec', default=1, type=int,
        help='Number of seconds')
    parser.add_argument('--z-normalize', default='False', type='bool',
        help='normalize audio')
    parser.add_argument('--aug-audio', default='False', type='bool',
        help='whether to augment audio')
    parser.add_argument('--audio-augtype', default='mild', type=str,
        choices=['na', 'mild', 'medium', 'heavy'],
        help='type of audio-augment default: mild')
    parser.add_argument('--decode-audio', default='True', type='bool',
        help='whether to deocde audio')

    # Model
    parser.add_argument('--model', default='av_gdt', help='model',
        choices=['av_gdt', 'avc', 'vid_text'])
    parser.add_argument('--vid-base-arch', default='r2plus1d_18',
        help='Video Base Arch for A-V model',
        choices=['r2plus1d_18', 'mc3_18', 's3d', 'r2plus1d_34', 'r2plus1d_50'])
    parser.add_argument('--aud-base-arch', default='vgg_audio',
        help='Audio Base Arch for A-V model',
        choices=['resnet9', 'resnet18', 'vgg_audio', 'resnet34', 'resnet50'])
    parser.add_argument('--pretrained', default='False', type='bool',
        help='Use pre-trained models from the modelzoo')
    parser.add_argument('--headcount', default=1, type=int,
        help='how many heads each modality has')
    parser.add_argument('--use-mlp', default='True', type='bool',
        help='Use MLP projection head')
    parser.add_argument('--use-max-pool', default='False', type='bool',
        help='Use max pool instead of GAP')
    parser.add_argument('--mlptype', default=0, type=int,
        help='MLP type (default: 0)')

    # Training
    parser.add_argument('--batch-size', default=16, type=int,
        help='batch-size / GPU')
    parser.add_argument('--epochs', default=200, type=int,
        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
        help='weight decay (default: 1e-4)')
    parser.add_argument('--use-scheduler', default='True', type='bool',
        help='Use LR scheduler')
    parser.add_argument('--scheduler-type', default='multi_step', type=str,
        choices=['multi_step', 'cosine'],
        help='Type of LR scheduler')
    parser.add_argument('--lr-milestones', default='150,175', type=str,
        help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int,
        help='number of warmup epochs')
    parser.add_argument('--sync-bn', default='False', type='bool',
        help='Use sync batch norm')
    parser.add_argument('--warmup-bn', default='False', type='bool',
        help='Warmup batchnorm')
    parser.add_argument('--norm-feat', default='True', type='bool',
        help='Normalize embeddings')

    # Logging
    parser.add_argument('--print-freq', default=10, type=int,
        help='print frequency')
    parser.add_argument('--output-dir', default='.',
        help='path where to save')

    # Checkpointing
    parser.add_argument('--resume', default='False', type='bool',
        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
        help='start epoch')

    # Mixed precision training parameters
    parser.add_argument('--apex', default='False', type='bool', 
        help='Use apex for mixed precision training'
    )
    parser.add_argument('--apex-opt-level', default='O1', type=str,
        help='For apex mixed precision training'
             'O0 for FP32 training, O1 for mixed precision training.'
             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )

    # distributed training parameters
    parser.add_argument('--device', default='cuda', 
        help='device')
    parser.add_argument('--distributed', default='False', type='bool',
        help='ddp mode')
    parser.add_argument('--dist-backend', default='nccl', type=str,
        help='distributed backend')
    parser.add_argument('--dist-url', default='env://',
        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
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
