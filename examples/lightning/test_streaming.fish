#!/usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --ip <ip_server> --logdir <log dir> --debug"
end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'ip=' 'logdir=' 'debug' -- $argv

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_logdir
    set _flag_logdir "log"
end

if not set -q _flag_bs
    set _flag_bs 1024
end

if not set -q _flag_epochs
    set _flag_epochs 4
end

if not set -q _flag_ip
    set _flag_ips 172.19.179.86
end

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set MAX_GPUS (nvidia-smi -L | wc -l)
set LOG $_flag_logdir
echo $HOST

# Set environment variables to test S3
set S3_IP $_flag_ip
set S3_PORT 9000
set -e http_proxy
set -x AWS_ENDPOINT_URL "http://$S3_IP:$S3_PORT"
set -x AWS_ACCESS_KEY_ID root
set -x AWS_SECRET_ACCESS_KEY passpass
set -x S3_ENDPOINT_URL "http://$S3_IP:$S3_PORT"

## create logdir
mkdir -p $LOG

### S3, files with DALI
echo "-- S3 STREAMING TRAINING --"
~/bin/mc alias set myminio http://$S3_IP:9000 root passpass
~/bin/mc cp myminio/imagenet/streaming/train/index_jpeg.json myminio/imagenet/streaming/train/index.json

#python3 train_model_streaming.py --epoch $EPOCHS --streaming-remote s3://imagenet/streaming/ -g 1 -b $BS --log-csv "$LOG/$HOST"_1_GPU_STREAMING_BS_"$BS"

python3 train_model_streaming.py --epoch $EPOCHS --streaming-remote s3://imagenet/streaming/ -g $MAX_GPUS -b $BS --log-csv "$LOG/$HOST"_"$MAX_GPUS"_GPU_CASSANDRA_BS_"$BS"
