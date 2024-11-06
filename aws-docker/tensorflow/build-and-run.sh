#! /usr/bin/env fish

function usage
    echo "Usage: script_name.fish --rootdir <root data dir> --logdir <log dir>"
    end

# Parse arguments
argparse 'd/rootdir=' 'logdir=' -- $argv

if not set -q _flag_rootdir
    echo "rootdir is mandatory"
    exit
end

if not set -q _flag_logdir
    echo "logdir is mandatory"
    exit
end

set TIMESTAMP (date +%s)
echo $TIMESTAMP

docker build --build-arg="TIMESTAMP=$TIMESTAMP" --progress=plain -t tensorflow:pap -f Dockerfile . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 200GB \
    -e NVIDIA_VISIBLE_DEVICES="none" \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $_flag_rootdir:/data:rw \
    -v $_flag_logdir:/logs:rw \
    -p 5050:5050 -p 5051:5051 \
    --entrypoint fish \
    --name tensorflow tensorflow:pap
