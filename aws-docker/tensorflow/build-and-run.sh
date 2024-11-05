#! /usr/bin/env fish
docker build --progress=plain -t tensorflow:pap -f Dockerfile . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 200GB \
    -e NVIDIA_VISIBLE_DEVICES="none" \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/cesco/code/aws-docker/tensorflow/code/:/code:rw \
    -v /mnt/scratch/ldic-varia/:/data:rw \
    -p 5050:5050 -p 5051:5051 \
    --entrypoint fish \
    --name tensorflow tensorflow:pap
