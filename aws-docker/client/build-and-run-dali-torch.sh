#! /usr/bin/env fish

docker build --progress=plain -t dali:aws -f Dockerfile.dali . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 200GB \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/scratch/ldic-varia/:/data:rw \
    # -v /mnt/bla:/ebs:rw \
    --entrypoint fish \
    --name client dali:aws
