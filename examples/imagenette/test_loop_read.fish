#!/usr/bin/env fish


function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ip <ip_server> --pc <port_cassandra> --ps <port_scylla> --rootdir <root data dir> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'ip=' 'pc=' 'ps=' 'd/rootdir=' 'logdir=' 'debug' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_rootdir
    set _flag_rootdir "/data"
end

if not set -q _flag_logdir
    set _flag_logdir "log"
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 10
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
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir
# create log dir
mkdir -p $LOG

## Local filesystem
### files with DALI
echo "-- DALI FILES TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader file --file-root $ROOT/imagenet-files/train/ --log-fn "$LOG/$HOST"_loop_read_DALI_file_BS_"$BS".pickle
	
### TFRecords with DALI
echo "-- DALI TFRECORDS TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader tfrecord --file-root $ROOT/imagenet-tfrecords/train/ --index-root $ROOT/imagenet-tfrecords/train_idx/ --log-fn "$LOG/$HOST"_loop_read_DALI_tfrecord_BS_"$BS".pickle

### files with Pytorch


## Hi latency
set TRAIN_DATA imagenet.data_train
set TRAIN_METADATA imagenet.metadata_train
set TRAIN_ROWS train.rows

### SCYLLA
echo "-- SCYLLA TEST --"
set IO_THREADS 4
set PREFETCH_BUFF 2
set SCYLLA_IP $_flag_ip
set SCYLLA_PORT $_flag_ps

sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$SCYLLA_IP\"\]/" private_data.py
sed -i --follow-symlinks "/cassandra_port/s/= \(.*\)/= $SCYLLA_PORT/" private_data.py

rm -f $TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 loop_read.py --bs=$BS --epochs=$EPOCHS --data-table=$TRAIN_DATA --rows-fn=$TRAIN_ROWS --log-fn "$LOG/$HOST"_loop_read_scylla_BS_"$BS".pickle

### CASSANDRA
echo "-- CASSANDRA TEST --"
set IO_THREADS 4
set PREFETCH_BUFF 2
set CASSANDRA_IP $_flag_ip
set CASSANDRA_PORT $_flag_pc

sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$CASSANDRA_IP\"\]/" private_data.py
sed -i --follow-symlinks "/cassandra_port/s/= \(.*\)/= $CASSANDRA_PORT/" private_data.py

rm -f $TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 loop_read.py --bs=$BS --epochs=$EPOCHS --data-table=$TRAIN_DATA --rows-fn=$TRAIN_ROWS --log-fn "$LOG/$HOST"_loop_read_cassandra_BS_"$BS".pickle

# Set environment variables to test S3
set S3_IP $_flag_ip
set S3_PORT 9000
set -e http_proxy
set -x AWS_ENDPOINT_URL "http://$S3_IP:$S3_PORT"
set -x AWS_ACCESS_KEY_ID root
set -x AWS_SECRET_ACCESS_KEY passpass
set -x S3_ENDPOINT_URL "http://$S3_IP:$S3_PORT"

### S3, files with DALI
echo "-- S3 DALI FILES TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader file --file-root s3://imagenet/files/train/ --log-fn "$LOG/$HOST"_loop_read_S3_DALI_file_BS_"$BS".pickle
	
### S3, TFRecords with DALI
echo "-- S3 DALI TFRECORDS TEST --"
python3 loop_read.py --epochs $EPOCHS --bs $BS --reader tfrecord --file-root s3://imagenet/tfrecords/train/ --index-root s3://imagenet/tfrecords/train_idx/ --log-fn "$LOG/$HOST"_loop_read_S3_DALI_tfrecord_BS_"$BS".pickle

# disable debug print
set -e fish_trace
