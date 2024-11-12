#! /usr/bin/env fish

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
    set _flag_logdir "/log"
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
set IP $_flag_ip
set PORT_SCYLLA $_flag_ps
set PORT_CASS $_flag_pc

#set TEST_CMD /home/user/cassandra-dali-plugin/aws-docker/torch-client/test_all.fish --host "$HOST" --bs="$BS" --epochs="$EPOCHS" --ip "$IP" --rootdir /data  --logdir /logs 2>&1 | tee /logs/test_all_torch_output.txt | tee /dev/stderr
set TEST_CMD ./aws-docker/torch-client/test_all.fish --host "$HOST" --bs="$BS" --epochs="$EPOCHS" --ip "$IP" --rootdir /data  --logdir /logs
#set TEST_CMD pwd 

set TIMESTAMP (date +%s)
echo $TIMESTAMP

docker build --build-arg="TIMESTAMP=$TIMESTAMP" --progress=plain -t dali:aws -f Dockerfile.dali . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 200GB \
    --rm -it \
    --gpus=all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $ROOT:/data:rw \
    -v $LOG:/logs:rw \
    # -v /mnt/bla:/ebs:rw \
    --name client dali:aws \
    aws-docker/torch-client/test_all.fish --host $HOST --bs=$BS --epochs=$EPOCHS --ip $IP --rootdir /data  --logdir /logs
