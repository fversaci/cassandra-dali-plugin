# Cassandra plugin for NVIDIA DALI

## Overview

This plugin enables data loading from an [Apache Cassandra NoSQL
database](https://cassandra.apache.org) to [NVIDIA Data Loading
Library (DALI)](https://github.com/NVIDIA/DALI) (which can be used to
load and preprocess images for Pytorch or TensorFlow).

## Requirements

cassandra-dali-plugin requires:
- NVIDIA DALI
- Cassandra C/C++ driver
- Cassandra Python driver

All the required packages are already installed in the provided
Dockerfile.

## Installation

The easiest way to test the cassandra-dali-plugin is by using the
provided Dockerfile (derived from [NVIDIA PyTorch
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)),
which also contains NVIDIA DALI, Cassandra C++ and Python drivers,
a Cassandra server, PyTorch and Apache Spark.

In a system which already provides the dependencies above, the plugin
can easily be installed with pip:
```bash
$ pip3 install .
```

The details of how to install missing dependencies in a system which
provides only some of the packages above, can be deduced from the
[Dockerfile](Dockerfile).

## Running the docker container

For better performance and for data persistence, it is advised to
mount a host directory for Cassandra on a fast disk (e.g.,
`/mnt/fast_disk/cassandra`), as shown in the commands below.

```bash
## Build and run cassandradl docker container
$ docker build -t cassandra-dali-plugin .
$ docker run --rm -it -v /mnt/fast_disk/cassandra:/cassandra/data:rw \
  --cap-add=sys_nice cassandra-dali-plugin

## Inside the Docker container:

## - Start Cassandra server
$ /cassandra/bin/cassandra   # Note that the shell prompt is immediately returned
                             # Wait until "state jump to NORMAL" is shown (about 1 minute)
```

## Dataset example

See the following annotated example for details on how to use and optimize this
plugin:
- [Imagenette](examples/imagenette/)

## Authors

Cassandra Data Loader is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonera, CRS4 <giovanni.busonera@crs4.it>

## License

cassandra-dali-plugin is licensed under the MIT License.  See LICENSE
for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
