#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ipc <ip_cassandra> --ips <ip_scylla> --ipm <ip_s3>"
end

# Parse arguments
argparse 'h/host=' 'b/bs=' 'e/epochs=' 'c/ipc=' 's/ips=' 'm/ipm' -- $argv

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_bs
    set _flag_bs 512
end

if not set -q _flag_epochs
    set _flag_epochs 10
end

if not set -q _flag_ipc
    set _flag_ipc 172.19.179.85
end

if not set -q _flag_ips
    set _flag_ips 172.19.179.86
end

if not set -q _flag_ipm
    set _flag_ipm 172.19.179.86
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs

echo $HOST
echo $_flag_ipc
echo $_flag_ips

### Set environment variables to test S3
set S3_IP $_flag_ipm
set -e http_proxy
set -x AWS_ENDPOINT_URL "http://$S3_IP:9000"
set -x AWS_ACCESS_KEY_ID root
set -x AWS_SECRET_ACCESS_KEY passpass
set -x S3_ENDPOINT_URL "http://$S3_IP:9000"

### Local filesystem
 ## file with Pytorch

  ## file with DALI
python loop_read.py --epochs 10 --bs 1024 --reader file --file-root /scratch/imagenet-files/train/ --log-fn "$HOST"_loop_read_DALI_file_BS_"$BS".pickle
	
  ## TfRecord with DALI
python loop_read.py --epochs 10 --bs 1024 --reader tfrecord --file-root /scratch/imagenet-tfrecords/train/ --index-root /scratch/imagenet-tfrecords/train_idx/ --log-fn "$HOST"_loop_read_DALI_tfrecord_BS_"$BS".pickle

  ## TfRecord with DALI 


### Remote testing 
set TRAIN_DATA imagenet.data_train
set TRAIN_METADATA imagenet.metadata_train
set TRAIN_ROWS train.rows
set VAL_DATA imagenet.data_val
set VAL_METADATA imagenet.metadata_val
set VAL_ROWS val.rows

### Medium latency


### Hi latency
set IO_THREADS 4
set PREFETCH_BUFF 2
set CASSANDRA_IP $_flag_ipc

sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$CASSANDRA_IP\"\]/" private_data.py

rm $TRAIN_ROWS
rm $VAL_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$VAL_METADATA --rows-fn=$VAL_ROWS

echo "CASSANDRA TEST"

### SCYLLA

set IO_THREADS 4
set PREFETCH_BUFF 2
set SCYLLA_IP $_flag_ips

sed -i --follow-symlinks "/cassandra_ips/s/\(\[.*\]\)/\[\"$SCYLLA_IP\"\]/" private_data.py

rm $TRAIN_ROWS
rm $VAL_ROWS
python3 cache_uuids.py --metadata-table=$TRAIN_METADATA --rows-fn=$TRAIN_ROWS
python3 cache_uuids.py --metadata-table=$VAL_METADATA --rows-fn=$VAL_ROWS

echo "SCYLLA TEST"
