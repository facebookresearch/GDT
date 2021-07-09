if [ -z "$1" ]
then
  SAV_FOLDER='./outfiles'
else
  SAV_FOLDER=$1
fi
if [ -z "$2" ]
then
  N_WEIGHTS_PATH='./outfiles/ckpt.pth'
else
  N_WEIGHTS_PATH=$2
fi
if [ -z "$3" ]
then
  CLIP_LEN=32
else
  CLIP_LEN=$3
fi
if [ -z "$4" ]
then
  MODEL='stica'
else
  MODEL=$4
fi
if [ -z "$5" ]
then
  num='100'
else
  num=$5
fi
OUTPUT_DIR=${SAV_FOLDER}/eval
mkdir -p $OUTPUT_DIR

echo $CLIP_LEN
echo $N_WEIGHTS_PATH

for fold in 1 2 3
do 
    bash scripts/run_finetune.sh ${N_WEIGHTS_PATH} ${OUTPUT_DIR} $num ${CLIP_LEN} ${MODEL} ${fold}
done
