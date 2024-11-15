#! /usr/bin/env fish

set TODAY (date -u +"%Y-%m-%d")

docker run \
   --rm -it \
   -p 9000:9000 \
   -p 9001:9001 \
   --name minio \
   -v /mnt/nvme/paper/:/data:rw \
   -v /mnt/log/server/$TODAY/:/logs:rw \
   -e "MINIO_ROOT_USER=root" \
   -e "MINIO_ROOT_PASSWORD=passpass" \
   quay.io/minio/minio server /data/minio/data/ --console-address ":9001"
