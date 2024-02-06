# Integration with NVIDIA Triton

This plugin also supports inference via the powerful and flexible
[NVIDIA Triton
server](https://github.com/triton-inference-server/server).

This allows a client to request that images stored in a remote
Cassandra server be inferenced on a different remote, GPU-powered
server.

## Cassandra DALI operators

The plugin provides two operators to be used with Triton:

### `fn.crs4.cassandra_interactive`

This operator expects a batch of UUIDs as input, represented as pairs
of uint64, and produces as output a batch containing the raw images
which are stored as BLOBs in the database, possibly paired with the
corresponding labels.

### `fn.crs4.cassandra_uncoupled`

The uncoupled version of the operator splits the input UUIDs (which,
in this case, can form a very long list) into mini-batches and
proceeds to request the images from the database using prefetching to
increase the throughput and hide the network latencies.

## Testing the examples

The directory [models](models) contains two subdirectories with
examples of pipelines using both `cassandra_interactive` and
`cassandra_uncoupled`.

### Building and running the docker container

The most convenient method to test the cassandra-dali-plugin with
Triton is by utilizing the provided
[Dockerfile.triton](../../Dockerfile.triton) (derived from [NVIDIA
PyTorch
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)),
which contains our plugin, NVIDIA Triton, NVIDIA DALI, Cassandra C++
and Python drivers, as well as a Cassandra server. To build and run
the container, use the following commands:

```bash
# Build and run cassandra-dali-triton docker container
$ docker build -t cassandra-dali-triton -f Dockerfile.triton .
$ docker run --cap-add=sys_admin --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --name cass-dali cassandra-dali-triton
```

### Starting and filling the DB

Once the Docker container is set up, it is possible to start the
database and populate it with images from the imagenette dataset using
the provided script:

```bash
./start-and-fill-db.sh  # might take a few minutes
```

### Starting Triton server

After the database is populated, we can start the Triton server with

```bash
./start-triton.sh
# i.e., tritonserver --model-repository ./models --backend-config dali,plugin_libs=/opt/conda/lib/python3.8/site-packages/libcrs4cassandra.so
```

Now you can leave this shell open, and it will display the logs of the
Triton server.

### Testing the inference

To run the clients, start a new shell in the container with following
command:

```bash
docker exec -ti cassandra-dali-triton fish
```

Now, within the container, run the following commands to test the
inference:

```bash
python3 client-http-triton.py
python3 client-grpc-triton.py
python3 client-grpc-stream-triton.py
```

You can also benchmark the inference performance using NVIDIA's
`perf_analyzer`:

```bash
perf_analyzer -m dali_cassandra_interactive --input-data uuids.json -b 256 --concurrency-range 16 -p 10000
perf_analyzer -m dali_cassandra_interactive --input-data uuids.json -b 256 --concurrency-range 16 -p 10000 -i grpc
perf_analyzer -m dali_cassandra_uncoupled --input-data uuids_2048.json --shape Reader:2048,2 --concurrency-range 4 -i grpc --streaming -p 10000
```
