#!/bin/sh
# Dynamically locate libcrs4cassandra.so using Python's sys.path
LIB_PATH=$(python3 -c "
import os, sys
for p in sys.path:
    if os.path.isdir(p):
        candidate = os.path.join(p, 'libcrs4cassandra.so')
        if os.path.exists(candidate):
            print(candidate)
            exit(0)
print('Error: libcrs4cassandra.so not found in Python path', file=sys.stderr)
exit(1)
")

tritonserver --model-repository ./models --backend-config dali,plugin_libs="${LIB_PATH}"
# --cuda-memory-pool-byte-size 0:134217728 --pinned-memory-pool-byte-size 536870912
