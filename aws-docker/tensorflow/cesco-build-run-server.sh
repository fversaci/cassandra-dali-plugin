#! /usr/bin/env fish

set TODAY (date -u +"%Y-%m-%d")

./build-and-run-server.sh --rootdir /mnt/nvme/paper/ --logdir /mnt/log/server/$TODAY/
