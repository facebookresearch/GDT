#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from utils import dist_collect_other,reduce_negatives


def collate_feats(list_of_feats):
    """"convenience function to tidy code"""
    return_list = []
    for f in list_of_feats:
        return_list.append(dist_collect_other(f.detach(), return_before_cat=True))
    return return_list


def compute_feats(model, video1, audio1, video2=None, audio2=None, feats1=None):
    # Perform forward pass to get features
    feat_v, feat_a = model(video1, audio1) if feats1 is None else feats1
    if video2 is None:
        return feat_v, feat_a
    feat_vT, feat_aT = model(video2, audio2)
    return (feat_v, feat_a), (feat_vT, feat_aT)


@torch.no_grad()
def get_pos_neg(hyp, feats1, feats1_col=None, feats2=None, feats2_col=None, concats=None, num_negatives=-1):
    # deal only with feats1
    feat_v, feat_a = feats1
    if feats2 is not None:
        feat_vT, feat_aT = feats2
    
    # Get transformation type
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

    # if distributed: get all other videos in batch
    # (collated from other GPUs) are the default negatives (cross-modal)
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
    if num_negatives != -1:
        feat_a_neg, feat_v_neg = reduce_negatives(
            feat_a_neg, feat_v_neg, num_negatives)

    pairs = [feat_v, feat_a, feat_v_pos, feat_a_pos, feat_v_neg, feat_a_neg, concats]
    return pairs



def get_loss(q, k, noise_batch, t=0.07, device='cuda'):
    N, C = q.shape
    # positive N x N s.t. positives are diagonals
    l_pos = torch.einsum("nc,mc -> nm", [q.view(N, C), k.view(N, C)])
    # negative logit N x K
    l_neg = torch.mm(q.view(N, C), noise_batch.transpose(0, 1))
    # positives are the 0-th
    labels = torch.arange(0, N, dtype=torch.long).to(device)
    logits = torch.cat([l_pos, l_neg], dim=1) / t
    prob = torch.mean((logits[:, 0] == logits.max(1)[0]).float()) * 100
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss, prob


def get_losses(pairs1, pairs2, nce_t=0.07):
    video_loss1, prob_vid1 = get_loss(
        pairs1[0], # v_i
        pairs1[2], # a_i
        pairs1[4], # Ba_j (and maybe hard-neg)
        t=nce_t,
    )
    audio_loss1, prob_aud1 = get_loss(
        pairs1[1], # a_i
        pairs1[3], # v_i
        pairs1[5], # Bv_j (and maybe hard-neg)
        t=nce_t,
    )
    loss = 0.5 * video_loss1 + 0.5 * audio_loss1
    if pairs2:
        video_loss2, prob_vid2 = get_loss(
            pairs2[0],  # Tv_i
            pairs2[2],  # Ta_i
            pairs2[4],  # TBa_j (and maybe hard-neg)
            t=nce_t,
        )
        audio_loss2, prob_aud2 = get_loss(
            pairs2[1],  # Ta_i
            pairs2[3],  # Tv_i
            pairs2[5],  # TBv_j (and maybe hard-neg)
            t=nce_t,
        )
        loss = (
            0.25 * video_loss1 + 
            0.25 * audio_loss1 + 
            0.25 * video_loss2 + 
            0.25 * audio_loss2
        )
    return loss, None