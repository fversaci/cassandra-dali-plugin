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

### `fn.crs4.cassandra_decoupled`

The decoupled version of the operator splits the input UUIDs (which,
in this case, can form a very long list) into mini-batches and
proceeds to request the images from the database using prefetching to
increase the throughput and hide the network latencies.

## Testing the examples

The directory [models](models) contains the following subdirectories,
with examples of pipelines using both `cassandra_interactive` and
`cassandra_decoupled`:

### `dali_cassandra_interactive`

This model retrieves the raw data from the database, decodes it into
images, performs normalization and cropping, and returns the images as
a tensor. It utilizes the `fn.crs4.cassandra_interactive` class.

### `dali_cassandra_interactive_stress`

This model retrieves the raw data from the database and returns the
first byte of every BLOB. It utilizes the
`fn.crs4.cassandra_interactive` class.

### `dali_cassandra_decoupled`

This model retrieves the raw data from the database, decodes it into
images, performs normalization and cropping, and returns the images as
a tensor. It utilizes the `fn.crs4.cassandra_decoupled` class.

### `dali_cassandra_decoupled_stress`

This model retrieves the raw data from the database and returns the
first byte of every BLOB. It utilizes the
`fn.crs4.cassandra_decoupled` class.

### `classification_resnet`

This model utilizes a pre-trained ResNet50 for ImageNet classification
to perform inference, predownloaded using the
[runme.py](models/classification_resnet/1/runme.py) script.

### `cass_to_inference`

This ensemble model connects `dali_cassandra_interactive` and
`classification_resnet` to load and preprocess images from the
database and perform inference on them.

### `cass_to_inference_decoupled`

This ensemble model connects `dali_cassandra_decoupled` and
`classification_resnet` to load and preprocess images from the
database and perform inference on them.


### Building and running the docker container

The most convenient method to test the cassandra-dali-plugin with
Triton is by utilizing the provided
[docker-compose.triton.yml](../../docker-compose.triton.yml), which
runs a Cassandra container and another container, (derived from
[NVIDIA Triton Inference Server
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)),
which contains our plugin, NVIDIA Triton, NVIDIA DALI, Cassandra C++
and Python drivers. To build and run the containers, use the following
commands:

```bash
docker compose -f docker-compose.triton.yml up --build -d
docker compose exec dali-cassandra fish
```

### Filling the DB

Once the Docker containers are set up, it is possible to populate the
database with images from the imagenette dataset using the provided
script:

```bash
./fill-db.sh  # might take a few minutes
```

### Starting Triton server

After the database is populated, we can start the Triton server with

```bash
./start-triton.sh
```

Now you can leave this shell open, and it will display the logs of the
Triton server.

### Testing the inference

To run the clients, start a new shell in the container with following
command:

```bash
docker compose exec dali-cassandra fish
```

Now, within the container, run the following commands to test the
inference:

```bash
python3 client-http-stress.py
python3 client-grpc-stress.py
python3 client-grpc-stream-stress.py
python3 client-http-ensemble.py
python3 client-grpc-ensemble.py
python3 client-grpc-stream-ensemble.py
```

You can also benchmark the inference performance using NVIDIA's
[perf_analyzer](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B/perf_analyzer#readme). For
example:

```bash
perf_analyzer -m dali_cassandra_interactive_stress --input-data uuids.json -b 256 --concurrency-range 16 -p 10000
perf_analyzer -m dali_cassandra_interactive_stress --input-data uuids.json -b 256 --concurrency-range 16 -p 10000 -i grpc
perf_analyzer -m dali_cassandra_decoupled_stress --input-data uuids_2048.json --shape UUID:2048,2 --concurrency-range 4 -i grpc --streaming -p 10000
perf_analyzer -m cass_to_inference --input-data uuids.json -b 256 --concurrency-range 16 -i grpc
perf_analyzer -m cass_to_inference_decoupled --input-data uuids_2048.json --shape UUID:2048,2 --concurrency-range 4 -i grpc --streaming -p 10000
```
