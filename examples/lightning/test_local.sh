#!/usr/bin/env bash

HOST="KENTU"
BS=512
MAX_GPUS=`nvidia-smi -L | wc -l`
EPOCHS=2


echo $HOST

### Files
TRAIN_FOLDER=/scratch/imagenet-files/train
VAL_FOLDER=/scratch/imagenet-files/val


python3 train_model_no_IO.py --train-folder ${TRAIN_FOLDER} --val-folder ${VAL_FOLDER} -g 1 -b ${BS} --log-csv ${HOST}_1_GPU_TFR_BS_${BS}

python3 train_model_no_IO.py --train-folder ${TRAIN_FOLDER} --val-folder ${VAL_FOLDER} -g ${MAX_GPUS} -b ${BS} --log-csv ${HOST}_${MAX_GPUS}_GPU_TFR_BS_${BS}


### Tfrecords
TRAIN_TFR=/scratch/imagenet-tfrecords/train
TRAIN_INDEX=/scratch/imagenet-tfrecords/train_idx/
VAL_TFR=/scratch/imagenet-tfrecords/val
VAL_INDEX=/scratch/imagenet-tfrecords/val_idx/

python3 train_model_no_IO.py --train-tfr-folder ${TRAIN_TFR} --train-index-folder ${TRAIN_INDEX} --val-tfr-folder ${VAL_TFR} --val-index-folder ${VAL_INDEX} -g 1 -b ${BS} --log-csv ${HOST}_1_GPU_TFR_BS_${BS}

python3 train_model_no_IO.py --train-tfr-folder ${TRAIN_TFR} --train-index-folder ${TRAIN_INDEX} --val-tfr-folder ${VAL_TFR} --val-index-folder ${VAL_INDEX} -g ${MAX_GPUS} -b ${BS} --log-csv ${HOST}_${MAX_GPUS}_GPU_TFR_BS_${BS}
