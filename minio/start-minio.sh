docker run \
   --rm -it \
   -p 9000:9000 \
   -p 9001:9001 \
   --name minio \
   -v /mnt/scratch/ldic-varia/paper-data/:/data:rw \
   -e "MINIO_ROOT_USER=root" \
   -e "MINIO_ROOT_PASSWORD=passpass" \
   quay.io/minio/minio server /data/minio/data/ --console-address ":9001"
