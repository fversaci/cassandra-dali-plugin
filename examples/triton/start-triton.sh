#!/bin/sh
tritonserver --model-repository ./models --backend-config dali,plugin_libs=/usr/local/lib/python3.12/dist-packages/libcrs4cassandra.so
# --cuda-memory-pool-byte-size 0:134217728 --pinned-memory-pool-byte-size 536870912
