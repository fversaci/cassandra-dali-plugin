#!/usr/bin/env bash

HOST="KENTU"
BS=512
MAX_GPUS=`nvidia-smi -L | wc -l`
EPOCHS=2


echo $HOST

### NO Dataloader 
python3 train_model_no_IO.py --epoch ${EPOCHS} -b ${BS} --no-io -g 1 --log-csv ${HOST}_1_GPU_NO_IO_BS_${BS}
python3 train_model_no_IO.py --epoch ${EPOCHS} -b ${BS} --no-io -g ${MAX_GPUS} --log-csv ${HOST}_${MAX_GPUS}_GPU_NO_IO_BS_${BS}
