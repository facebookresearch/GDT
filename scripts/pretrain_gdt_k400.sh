#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --gres=gpu:8
#SBATCH --job-name=GDT-AV
#SBATCH --mem=450GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/%u/jobs/%j.out
#SBATCH --partition=learnfair
#SBATCH --signal=USR1@120
#SBATCH --time=72:00:00

module load anaconda3
source activate GDT

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

if [ -z "$1" ]
then
	HYP_DESC='av_basecase'
else
	HYP_DESC=$1
fi
if [ -z "$2" ]
then
	HYP=1
else
	HYP=$2
fi

SAV_FOLDER="/checkpoint/mandelapatrick/gdt_av_oss/${HYP_DESC}"
mkdir -p ${SAV_FOLDER}
DS='kinetics'
MUM_EPOCHS=201
MOMENTUM=0.9
WEIGHT_DECAY=1e-5
USE_SCHEDULER='True'
SCHEDULER_TYPE='multi_step'
LR_GAMMA=0.1
LR_WARM_EPOCHS=10
LR_MIL='251'
MODEL='av_gdt'
USE_MLP_HEAD='True'
VID_BASE_ARCH='r2plus1d_18'
PRETRAINED='False'
MLPTYPE=0
NUM_CLIPS=1
AUGTYPE=1
NUM_WORKERS=10
NCE_T=0.07
WARM_BN='False'
SYNC_BN='True'
DUALDATA='True'
TARGET_FPS=30
CLIP_LEN=30
CROP_SIZE=112
SAMPLE_RATE=1
BATCH_SIZE=8
LR=1e-2
HEADCOUNT=1
COLORJITTER='False'
NUM_NEGATIVES=-1

# AUDIO AUGS
AUD_BASE_ARCH='resnet9'
AUG_AUDIO='True'
AUD_AUG_TYPE='medium'
AUD_SAMPLE_RATE=24000
AUD_SPEC_TYPE=1
AUD_VOLUME_JITTERING='True'
AUD_TEMPORAL_JITTERING='False'
AUD_NUM_SEC=1
AUD_Z_NORMALIZE='True'


# command
srun --label python3 main_gdt.py \
--hypothesis ${HYP} \
--output_dir ${SAV_FOLDER} \
--dataset ${DS} \
--batch_size ${BATCH_SIZE} \
--epochs ${MUM_EPOCHS} \
--lr ${LR} \
--momentum ${MOMENTUM} \
--weight_decay ${WEIGHT_DECAY} \
--use_scheduler ${USE_SCHEDULER} \
--scheduler_type ${SCHEDULER_TYPE} \
--lr_gamma ${LR_GAMMA} \
--lr_warmup_epochs ${LR_WARM_EPOCHS} \
--lr_milestones ${LR_MIL} \
--warmup_bn ${WARM_BN} \
--sync_bn ${SYNC_BN} \
--model ${MODEL} \
--vid_base_arch ${VID_BASE_ARCH} \
--aud_base_arch ${AUD_BASE_ARCH} \
--pretrained ${PRETRAINED} \
--mlptype ${MLPTYPE} \
--augtype ${AUGTYPE} \
--target_fps ${TARGET_FPS} \
--clip_len ${CLIP_LEN} \
--train_crop_size ${CROP_SIZE} \
--sample_rate ${SAMPLE_RATE} \
--clips_per_video ${NUM_CLIPS} \
--workers ${NUM_WORKERS} \
--nce_t ${NCE_T} \
--use_mlp ${USE_MLP_HEAD} \
--aug_audio ${AUG_AUDIO} \
--headcount ${HEADCOUNT} \
--audio_augtype ${AUD_AUG_TYPE} \
--colorjitter ${COLORJITTER} \
--num_negatives ${NUM_NEGATIVES} \
--dualdata ${DUALDATA} \
--aud_sample_rate ${AUD_SAMPLE_RATE} \
--aud_spec_type ${AUD_SPEC_TYPE} \
--use_volume_jittering ${AUD_VOLUME_JITTERING} \
--use_temporal_jittering ${AUD_TEMPORAL_JITTERING} \
--num_sec ${AUD_NUM_SEC} \
--z_normalize ${AUD_Z_NORMALIZE} \
