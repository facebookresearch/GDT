#!/bin/bash

# Parameters
PROJECT_ROOT="./"
HEADER="$PROJECT_ROOT/temp"
cat > ${HEADER} <<- EOM
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --partition=learnfair
#SBATCH --output=/checkpoint/%u/jobs/%j.out 
#SBATCH --error=/checkpoint/%u/jobs/%j.err
#SBATCH --mem=450GB
#SBATCH --time=36:00:00
module load anaconda3
module load NCCL/2.4.7-1-cuda.10.0
source activate GDT
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo
EOM

MAINSCRIPT="./eval_video.py"

# create exp variant
FEAT_EXTRACT='False'
BATCH_SIZE=32
NUM_EPOCHS=12
OPTIM_NAME='sgd'
HEAD_LR=0.0025
BASE_LR=0.00025
WEIGHT_DECAY=0.005
MOMENTUM=0.9
LR_GAMMA=0.05
USE_SCHEDULER='True'
LR_WARM_UP_EPOCHS=2
LR_MIL='6,10'
COLORJITTER='True'
MODEL='av_gdt'
USE_DROPOUT='False'
USE_BN='False'
USE_L2_NORM='False'
VID_BASE_ARCH='r2plus1d_18'
if [ -z "$4" ]
then
  CLIP_LEN=32 
else
  CLIP_LEN=$4
fi
AUD_BASE_ARCH='resnet9' # resnet9, resnet34, resnet18
NUM_SPATIAL_CROPS=3
TEST_TIME_CJ='False'
MAX_TRAIN_CLIPS_PER_VID=10
MAX_VAL_CLIPS_PER_VID=10
NUM_WORKERS=16
if [ -z "$6" ]
then
  FOLDS='1'
else
  FOLDS=$6
fi
USE_MLP_HEAD='False'
PRETRAINED='False'
if [ -z "$5" ]
then
  MODEL='stica'
else
  MODEL=$5
fi
HEADCOUNT=1
echo $1 $2 $3 $4 $5 $6 $7 $8 $9

# command
for dataset in 'ucf101' 'hmdb51'
do
    for lr in $HEAD_LR
    do    
        EXP_NAME="$dataset-$3-clip-len-$CLIP_LEN-model-${MODEL}"
        EXP="$2/$EXP_NAME/$FOLDS"
        mkdir -p $EXP
        SCRIPT="${EXP}/launcher.sh"
        cp $HEADER $SCRIPT
        echo "
python3 $MAINSCRIPT \
--fold ${FOLDS} \
--output-dir $EXP \
--dataset $dataset \
--ckpt-epoch $3 \
--weights-path $1 \
--feature-extract ${FEAT_EXTRACT} \
--batch-size ${BATCH_SIZE} \
--epochs ${NUM_EPOCHS} \
--optim-name ${OPTIM_NAME} \
--head-lr $lr \
--base-lr ${BASE_LR} \
--momentum ${MOMENTUM} \
--weight-decay ${WEIGHT_DECAY} \
--use-scheduler ${USE_SCHEDULER} \
--lr-gamma ${LR_GAMMA} \
--lr-warmup-epochs ${LR_WARM_UP_EPOCHS} \
--lr-milestones ${LR_MIL} \
--model ${MODEL} \
--vid-base-arch ${VID_BASE_ARCH} \
--aud-base-arch ${AUD_BASE_ARCH} \
--pretrained ${PRETRAINED} \
--use-mlp ${USE_MLP_HEAD} \
--use-dropout ${USE_DROPOUT} \
--use-bn ${USE_BN} \
--use-l2-norm ${USE_L2_NORM} \
--headcount ${HEADCOUNT} \
--clip-len ${CLIP_LEN} \
--colorjitter ${COLORJITTER} \
--train-clips-per-video ${MAX_TRAIN_CLIPS_PER_VID} \
--val-clips-per-video ${MAX_VAL_CLIPS_PER_VID} \
--num-spatial-crops ${NUM_SPATIAL_CROPS} \
--test-time-cj ${TEST_TIME_CJ} \
--use_random_resize_crop True \
--num_layer 2 \
--workers ${NUM_WORKERS}" >> $SCRIPT
        chmod +x $SCRIPT
        
        # launch experiment
        # $SCRIPT
        sbatch --job-name=$EXP_NAME "$SCRIPT"
    done
done
