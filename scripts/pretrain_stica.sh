#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --gres=gpu:8
#SBATCH --job-name=STICA
#SBATCH --mem=450GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/%u/jobs/%j.out
#SBATCH --partition=learnfair
#SBATCH --signal=USR1@120
#SBATCH --time=72:00:00

module load anaconda3
conda activate GDT
source activate GDT

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES


master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


if [ -z "$1" ]
then
	NUM_FRAMES=30
else
	NUM_FRAMES=$1
fi
if [ -z "$2" ]
then
    AUD_NUM_SEC=1
else
    AUD_NUM_SEC=$2
fi
########## spatial crops #########
if [ -z "$3" ]
then
	NUM_LARGE_CROPS=1
else
	NUM_LARGE_CROPS=$3
fi
if [ -z "$4" ]
then
	NUM_SMALL_CROPS=0
else
	NUM_SMALL_CROPS=$4
fi
########## temporal crops #########
if [ -z "$5" ]
then
	NUM_SMALL_TCROPS=0
else
	NUM_SMALL_TCROPS=$5
fi
if [ -z "$6" ]
then
	NUM_LARGE_TCROPS=0
else
	NUM_LARGE_TCROPS=$6
fi
###### transformer pooling?#######
if [ -z "$7" ]
then
	NUM_LAYER=2
else
	NUM_LAYER=$7
fi

BATCH_SIZE=8
VID_BASE_ARCH='r2plus1d_18'
USE_FP_16='False'

### PARAMETERS #################################
CROSS_MODAL_ALPHA=0.5
### other
RESIZE_CROP='True' # ie it uses RRC
MULTI_CROP='True'
TEMP=0.5
##########################################################################################

### EXP DUMP PATH
EXP_NAME="fm_multi_crop_temp_${TEMP}_${MULTI_CROP}_${NUM_LARGE_CROPS}L${NUM_SMALL_CROPS}S${NUM_LARGE_TCROPS}TL${NUM_SMALL_TCROPS}TS_${NUM_LAYER}_cross_Transf_alpha_${CROSS_MODAL_ALPHA}_bandit_algo_${BANDIT_ALGO}"
SAV_FOLDER="/checkpoint/mandelapatrick/CVPR21_MC_GDT_oss/${EXP_NAME}"
mkdir -p ${SAV_FOLDER}

### DATA PARAMS
TARGET_FPS=30
SAMPLE_RATE=1
NUM_CLIPS=1
CROP_SIZE=112
AUD_SAMPLE_RATE=24000
AUD_SPEC_TYPE=2
AUD_VOLUME_JITTERING='True'
AUD_TEMPORAL_JITTERING='False'
AUD_Z_NORMALIZE='False'
AUD_AUG_TYPE='none'

### OPTIM PARAMS
EPOCHS=151
BASE_LR=64e-2
WEIGHT_DECAY=1e-5
LR_WARM_EPOCHS=10
START_WARMUP_LR=1e-2
USE_WAMRUP='True'
USE_LR_SCHEDULER='False'
FINAL_LR=0
USE_LARS='False'
NUM_SEC=${AUD_NUM_SEC}

### MODEL PARAMS
AUD_BASE_ARCH='resnet9'
USE_MLP='True'
MLP_DIM=256

### DATA AUGS
COLORJITTER='True'
USE_GRAYSCALE='True'
USE_GAUSSIAN='True'

# command
srun --label python -u main_stica.py \
--dump_path ${SAV_FOLDER} \
--num_frames ${NUM_FRAMES} \
--target_fps ${TARGET_FPS} \
--sample_rate ${SAMPLE_RATE} \
--num_train_clips ${NUM_CLIPS} \
--train_crop_size ${CROP_SIZE} \
--num_sec_aud ${AUD_NUM_SEC} \
--num_sec ${NUM_SEC} \
--aud_sample_rate ${AUD_SAMPLE_RATE} \
--aud_spec_type ${AUD_SPEC_TYPE} \
--use_volume_jittering ${AUD_VOLUME_JITTERING} \
--use_audio_temp_jittering ${AUD_TEMPORAL_JITTERING} \
--z_normalize ${AUD_Z_NORMALIZE} \
--audio_augtype ${AUD_AUG_TYPE} \
--epochs ${EPOCHS} \
--batch_size ${BATCH_SIZE} \
--base_lr ${BASE_LR} \
--final_lr ${FINAL_LR} \
--wd ${WEIGHT_DECAY} \
--warmup_epochs ${LR_WARM_EPOCHS} \
--start_warmup ${START_WARMUP_LR} \
--use_lars ${USE_LARS} \
--use_warmup_scheduler ${USE_WAMRUP} \
--use_lr_scheduler ${USE_LR_SCHEDULER} \
--vid_base_arch ${VID_BASE_ARCH} \
--aud_base_arch ${AUD_BASE_ARCH} \
--use_mlp ${USE_MLP} \
--mlp_dim ${MLP_DIM} \
--use_fp16 ${USE_FP_16} \
--dist_url $dist_url \
--num_layer ${NUM_LAYER} \
--colorjitter ${COLORJITTER} \
--use_grayscale ${USE_GRAYSCALE} \
--use_gaussian ${USE_GAUSSIAN} \
--multi_crop ${MULTI_CROP} \
--num_large_crops ${NUM_LARGE_CROPS} \
--num_small_crops ${NUM_SMALL_CROPS} \
--num_large_tcrops ${NUM_LARGE_TCROPS} \
--num_small_tcrops ${NUM_SMALL_TCROPS} \
--use_random_resize_crop ${RESIZE_CROP} \
--cross_modal_alpha ${CROSS_MODAL_ALPHA} \
--temp ${TEMP} \