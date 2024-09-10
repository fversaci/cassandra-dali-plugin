#!/usr/bin/env bash

HOST="KENTU"
BS=512
MAX_GPUS=`nvidia-smi -L | wc -l`
EPOCHS=2


echo $HOST

### Hi-latency Tests

### Prefetch data
rm -f ids_cache/* 
python3 cache_uuids.py --metadata-table=imagenette.metadata_train 
python3 cache_uuids.py --metadata-table=imagenette.metadata_val


### Cassandra
TRAIN_DATA=imagenet.data_train
TRAIN_METADATA=imagenet.metadata_train
VAL_DATA=imagenet.data_val
VAL_METADATA=imagenet.metadata_val
IO_TREHADS=4
PREFETCH_BUFF=2
CASSANDRA_IP=172.19.179.85

python3 train_model_no_IO.py --train-data-table ${TRAIN_DATA} --train-metadata-table ${TRAIN_METADATA} --val-data-table ${VAL_DATA} --val-metadata-table ${VAL_METADATA} --n-io-threads ${IO_THREADS} --n-prefetch-buffers ${PREFETCH_BUFF} -g 1 -b ${BS} --ip-addr ${CASSANDRA_IP} --log-csv ${HOST}_1_GPU_CASSANDRA_BS_${BS} -ips ${CASSANDRA_IP}

python3 train_model_no_IO.py --train-data-table ${TRAIN_DATA} --train-metadata-table ${TRAIN_METADATA} --val-data-table ${VAL_DATA} --val-metadata-table ${VAL_METADATA} --n-io-threads ${IO_THREADS} --n-prefetch-buffers ${PREFETCH_BUFF} -g ${MAX_GPUS} -b ${BS} --ip-addr ${CASSANDRA_IP} --log-csv ${HOST}_${MAX_GPUS}_GPU_CASSANDRA_BS_${BS} -ips ${CASSANDRA_IP}


### Scylla
TRAIN_DATA=imagenet.data_train
TRAIN_METADATA=imagenet.metadata_train
VAL_DATA=imagenet.data_val
VAL_METADATA=imagenet.metadata_val
IO_TREHADS=4
PREFETCH_BUFF=2
SCYLLA_IP=172.19.179.85

python3 train_model_no_IO.py --train-data-table ${TRAIN_DATA} --train-metadata-table ${TRAIN_METADATA} --val-data-table ${VAL_DATA} --val-metadata-table ${VAL_METADATA} --n-io-threads ${IO_THREADS} --n-prefetch-buffers ${PREFETCH_BUFF} -g 1 -b ${BS} --ip-addr ${SCYLLA_IP} --log-csv ${HOST}_1_GPU_SCYLLA_BS_${BS} -ips ${SCYLLA_IP}

python3 train_model_no_IO.py --train-data-table ${TRAIN_DATA} --train-metadata-table ${TRAIN_METADATA} --val-data-table ${VAL_DATA} --val-metadata-table ${VAL_METADATA} --n-io-threads ${IO_THREADS} --n-prefetch-buffers ${PREFETCH_BUFF} -g ${MAX_GPUS} -b ${BS} --ip-addr ${SCYLLA_IP} --log-csv ${HOST}_${MAX_GPUS}_GPU_SCYLLA_BS_${BS} -ips ${SCYLLA_IP}
