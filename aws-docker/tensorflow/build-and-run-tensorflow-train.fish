#! /usr/bin/env fish

function usage
    echo "Usage: script_name.fish --host <host> --bs <batch size> --epochs <num_epochs> --rootdir <root data dir> --logdir <log dir> --ip <ip_server> --debug"
    end

# Parse arguments
argparse 'h/host=' 'bs=' 'e/epochs=' 'd/rootdir=' 'logdir=' 'ip=' -- $argv

if set -q _flag_debug
    set fish_trace 1
end

if not set -q _flag_host
    set _flag_host NONE
end

if not set -q _flag_rootdir
    echo "rootdir is mandatory"
    exit
end

if not set -q _flag_logdir
    echo "logdir is mandatory"
    exit
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

# Access the values passed to the named parameters
set HOST $_flag_host
set BS $_flag_bs
set EPOCHS $_flag_epochs
set ROOT $_flag_rootdir
set LOG $_flag_logdir
set IP $_flag_ip

set TIMESTAMP (date +%s)
echo $TIMESTAMP

docker build --build-arg="TIMESTAMP=$TIMESTAMP" --progress=plain -t tensorflow:pap -f Dockerfile . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 200GB \
    --gpus=all \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $_flag_rootdir:/data:rw \
    -v $_flag_logdir:/logs:rw \
    #-p 5050:5050 -p 5051:5051 \
    --name tensorflow tensorflow:pap \
    /home/user/cassandra-dali-plugin/aws-docker/tensorflow/test_all_train.fish --host $HOST --bs $BS --epochs $EPOCHS --rootdir /data --ip $IP --logdir /logs --debug
