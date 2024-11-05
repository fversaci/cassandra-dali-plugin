#! /usr/bin/env fish

docker build --progress=plain -t cassandra-dali-plugin:aws -f Dockerfile.cassandra . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 30GB \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/scratch/ldic-varia/paper-data/:/data:rw \
    --entrypoint fish \
    -p 9042:9042 \
    --name cassandra cassandra-dali-plugin:aws
