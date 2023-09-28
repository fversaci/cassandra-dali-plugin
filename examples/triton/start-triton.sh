#!/bin/sh
tritonserver --model-repository ./models --backend-config dali,plugin_libs=/opt/conda/lib/python3.8/site-packages/libcrs4cassandra.so
