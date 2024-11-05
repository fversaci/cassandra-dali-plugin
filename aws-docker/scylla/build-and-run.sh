#! /usr/bin/env fish

docker build --progress=plain -t scylla:aws -f Dockerfile.scylla . ; \
and docker run \
    --cap-add=sys_admin --cap-add=net_admin --shm-size 30GB \
    --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt/scratch/ldic-varia/paper-data/scylla:/var/lib/scylla:rw \
    -p 9043:9042 \
    --name scylla scylla:aws \
    --reactor-backend=epoll
