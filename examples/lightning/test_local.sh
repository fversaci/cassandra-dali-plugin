#!/bin/bash

HOST="KENTU"
BS=512
MAX_GPUS=`nvidia-smi -L | wc -l`
EPOCHS=2


echo $HOST

### NO Dataloader 
python3 train_model_no_IO.py --epoch ${EPOCHS} -b ${BS} --no-io -g 1 --log-csv ${HOST}_1_GPU_NO_IO_BS_${BS}
python3 train_model_no_IO.py --epoch ${EPOCHS} -b ${BS} --no-io -g ${MAX_GPUS} --log-csv ${HOST}_${MAX_GPUS}_GPU_NO_IO_BS_${BS}

### Files
TRAIN_FOLDER=
VAL_FOLDER=


### Tfrecords
### Cassandra
TRAIN_DATA=/scratch/imagenet-tfrecords/train
TRAIN_INDEX=/scratch/imagenet-tfrecords/train_idx/
VAL_DATA=/scratch/imagenet-tfrecords/val
VAL_INDEX=/scratch/imagenet-tfrecords/val_idx/

python3 train_model_no_IO.py --train-tfr-folder ${TRAIN_DATA} --train-index-folder ${TRAIN_INDEX} --val-tfr-folder ${VAL_DATA} --val-index-folder ${VAL_INDEX} -g 1 -b ${BS} --log-csv ${HOST}_1_GPU_TFR_BS_${BS}

python3 train_model_no_IO.py --train-tfr-folder ${TRAIN_DATA} --train-index-folder ${TRAIN_INDEX} --val-tfr-folder ${VAL_DATA} --val-index-folder ${VAL_INDEX} -g ${MAX_GPUS} -b ${BS} --log-csv ${HOST}_${MAX_GPUS}_GPU_TFR_BS_${BS}
