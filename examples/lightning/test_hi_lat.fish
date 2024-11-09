#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ip <ip_server> --pc <port_cassandra> --ps <port_scylla> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'ip=' 'pc=' 'ps=' 'logdir=' 'debug' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_logdir
    set _flag_logdir "log"
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_ip
    set _flag_ip 172.19.179.86
end

if not set -q _flag_pc
    set _flag_pc 9042
end

if not set -q _flag_ps
    set _flag_ps 9043
end

# Access the values passed to the named parameters
set HOST $_flag_host
set MAX_GPUS (nvidia-smi -L | wc -l)
set BS $_flag_bs
set EPOCHS $_flag_epochs
set LOG $_flag_logdir

echo $HOST

## Create log dir
mkdir -p $LOG

### Hi-latency Tests

set TRAIN_DATA imagenet.data_train
set TRAIN_METADATA imagenet.metadata_train
set TRAIN_ROWS train.rows
set VAL_DATA imagenet.data_val
set VAL_METADATA imagenet.metadata_val
set VAL_ROWS val.rows

### CASSANDRA

set IO_THREADS 4
set PREFETCH_BUFF 2
set CASSANDRA_IP $_flag_ip
set CASSANDRA_PORT $_flag_pc

echo "Editing private_data.py"
sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$CASSANDRA_IP\"\]/" private_data.py
sed -i --follow-symlinks "/cassandra_port/s/= \(.*\)/= $CASSANDRA_PORT/" private_data.py

rm $TRAIN_ROWS
rm $VAL_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$VAL_METADATA --rows-fn=$VAL_ROWS

echo "CASSANDRA TEST"

#python3 train_model.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-rows-fn $TRAIN_ROWS --val-data-table $VAL_DATA --val-rows-fn $VAL_ROWS --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g 1 -b $BS --out-of-order --slow-start 4 --log-csv "$LOG/$HOST"_1_GPU_CASSANDRA_BS_"$BS"

python3 train_model.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-rows-fn $TRAIN_ROWS --val-data-table $VAL_DATA --val-rows-fn $VAL_ROWS --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g $MAX_GPUS -b $BS --out-of-order --slow-start 4 --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_CASSANDRA_BS_"$BS"


### SCYLLA

set IO_THREADS 4
set PREFETCH_BUFF 2
set SCYLLA_IP $_flag_ip
set SCYLLA_PORT $_flag_ps

sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$SCYLLA_IP\"\]/" private_data.py
sed -i --follow-symlinks "/cassandra_port/s/= \(.*\)/= $SCYLLA_PORT/" private_data.py

rm $TRAIN_ROWS
rm $VAL_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$VAL_METADATA --rows-fn=$VAL_ROWS

echo "SCYLLA TEST"

#python3 train_model.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-rows-fn $TRAIN_ROWS --val-data-table $VAL_DATA --val-rows-fn $VAL_ROWS --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g 1 -b $BS --out-of-order --slow-start 4 --log-csv "$LOG/$HOST"_1_GPU_SCYLLA_BS_"$BS"

python3 train_model.py --epoch $EPOCHS --train-data-table $TRAIN_DATA --train-rows-fn $TRAIN_ROWS --val-data-table $VAL_DATA --val-rows-fn $VAL_ROWS --n-io-threads $IO_THREADS --n-prefetch-buffers $PREFETCH_BUFF -g $MAX_GPUS -b $BS --out-of-order --slow-start 4 --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_SCYLLA_BS_"$BS"
