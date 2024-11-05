#! /usr/bin/env fish

docker build --progress=plain -t tensorflow:aws -f Dockerfile.tf . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 70GB \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e NVIDIA_VISIBLE_DEVICES="none" \
    -v /mnt/scratch/ldic-varia/:/data:rw \
    -v /mnt/tdm-dic/users/cesco/code:/code:rw \
    # -v /mnt/bla:/ebs:rw \
    --name client-tf tensorflow:aws
