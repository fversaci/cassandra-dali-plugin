#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ipc <ip_cassandra> --ips<ip_scylla>"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'ipc' 'ips' -- $argv

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_ipc
    set _flag_ipc 172.19.179.85
end

if not set -q _flag_ips
    set _flag_ips 172.19.179.86
end


# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set MAX_GPUS (nvidia-smi -L | wc -l)

echo $HOST

### Hi-latency Tests

### CASSANDRA

set TRAIN_DATA imagenet.data_train
set TRAIN_METADATA imagenet.metadata_train
set VAL_DATA imagenet.data_val
set VAL_METADATA imagenet.metadata_val
set IO_THREADS 4
set PREFETCH_BUFF 2
set CASSANDRA_IP $_flag_ipc

sed -i --follow-symlinks '/cassandra_ips/s/\(\[.*\]\)/\[\"$CASSANDRA_IP\"\]/' private_data.py

rm -f ids_cache/*
python3 cache_uuids.py --metadata-table=imagenette.metadata_train
python3 cache_uuids.py --metadata-table=imagenette.metadata_val

python3 train_model_no_IO.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-metadata-table $TRAIN_METADATA --val-data-table $VAL_DATA --val-metadata-table $VAL_METADATA --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g 1 -b $BS --log-csv "$HOST"_1_GPU_CASSANDRA_BS_"$BS"

python3 train_model_no_IO.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-metadata-table $TRAIN_METADATA --val-data-table $VAL_DATA --val-metadata-table $VAL_METADATA --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g $MAX_GPUS -b $BS --log-csv "$HOST"_"$MAX_GPUS"_GPU_CASSANDRA_BS_"$BS"


### SCYLLA

set TRAIN_DATA imagenet.data_train
set TRAIN_METADATA imagenet.metadata_train
set VAL_DATA imagenet.data_val
set VAL_METADATA imagenet.metadata_val
set IO_THREADS 4
set PREFETCH_BUFF 2
set SCYLLA_IP $_flag_ips

sed -i --follow-symlinks '/cassandra_ips/s/\(\[.*\]\)/\[\"$SCYLLA_IP\"\]/' private_data.py

rm -f ids_cache/*
python3 cache_uuids.py --metadata-table=imagenette.metadata_train
python3 cache_uuids.py --metadata-table=imagenette.metadata_val

python3 train_model_no_IO.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-metadata-table $TRAIN_METADATA --val-data-table $VAL_DATA --val-metadata-table $VAL_METADATA --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g 1 -b $BS --log-csv "$HOST"_1_GPU_SCYLLA_BS_"$BS"

python3 train_model_no_IO.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-metadata-table $TRAIN_METADATA --val-data-table $VAL_DATA --val-metadata-table $VAL_METADATA --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g $MAX_GPUS -b $BS --log-csv "$HOST"_"$MAX_GPUS"_GPU_SCYLLA_BS_"$BS"
