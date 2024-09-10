#!/usr/bin/env bash

HOST="KENTU"
BS=512
MAX_GPUS=`nvidia-smi -L | wc -l`
EPOCHS=2


echo $HOST

### NO Dataloader 
test_no_io.sh ${HOST} ${BS} ${MAX_GPU} ${EPOCHS}

### Local test
test_local.sh ${HOST} ${BS} ${MAX_GPU} ${EPOCHS}

## Hi-lat
test_hi_lat.sh ${HOST} ${BS} ${MAX_GPU} ${EPOCHS}
