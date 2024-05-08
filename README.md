# Cassandra plugin for NVIDIA DALI

## Overview

This plugin enables data loading from an [Apache Cassandra NoSQL
database](https://cassandra.apache.org) to [NVIDIA Data Loading
Library (DALI)](https://github.com/NVIDIA/DALI) (which can be used to
load and preprocess images for PyTorch or TensorFlow).

### DALI compatibility
The plugin has been tested and is compatible with DALI v1.37.

## Running the docker container

The easiest way to test the cassandra-dali-plugin is by using the
provided [Dockerfile](Dockerfile) (derived from [NVIDIA PyTorch
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)),
which also contains NVIDIA DALI, Cassandra C++ and Python drivers,
a Cassandra server, PyTorch and Apache Spark, as shown in the commands below.

```bash
# Build and run cassandra-dali-plugin docker container
$ docker build -t cassandra-dali-plugin .
$ docker run --rm -it --cap-add=sys_admin cassandra-dali-plugin
```

Alternatively, for better performance and for data persistence, it is
advised to mount a host directory for Cassandra on a fast disk (e.g.,
`/mnt/fast_disk/cassandra`):

```bash
# Run cassandra-dali-plugin docker container with external data dir
$ docker run --rm -it -v /mnt/fast_disk/cassandra:/cassandra/data:rw \
  --cap-add=sys_nice cassandra-dali-plugin
```

## How to call the plugin

Once installed the plugin can be loaded with

```python
import crs4.cassandra_utils
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.fn as fn
import pathlib

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)
```

At this point the plugin can be integrated in a [DALI
pipeline](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html),
for example replacing a call to `fn.readers.file` with
```python
images, labels = fn.crs4.cassandra(
    name="Reder", cassandra_ips=["127.0.0.1"],
    table="imagenet.train_data", label_col="label", label_type="int",
    data_col="data", id_col="img_id",
    source_uuids=train_uuids, prefetch_buffers=2,
)
```

Below, we'll provide a full summary of the parameters' meanings. If
you prefer to skip this section, here you can find some [working
examples](README.md#examples).

### Basic parameters

- `name`: name of the module to be passed to DALI (e.g. "Reader")
- `cassandra_ips`: list of IP pointing to the DB (e.g., `["127.0.0.1"]`)
- `cassandra_port`: Cassandra TCP port (default: `9042`)
- `table`: data table (e.g., `imagenet.train_data`)
- `label_col`: name of the label column (e.g., `label`)
- `label_type`: type of label: "int", "blob" or "none" ("int" is
  typically used for classification, "blob" for segmentation)
- `data_col`: name of the data column (e.g., `data`)
- `id_col`: name of the UUID column (e.g., `img_id`)
- `source_uuids`: full list of UUIDs, as strings, to be retrieved

### Authentication and authorization

Cassandra server provides a wide range of (non-mandatory) options for
configuring authentication and authorization. Our plugin supports
them by using the following parameters:

- `username`: username for Cassandra
- `password`: password for Cassandra
- `use_ssl`: use SSL to encrypt the transfers: True or False
- `ssl_certificate`: public key of the Cassandra server (e.g., "server.crt")
- `ssl_own_certificate`: public key of the client (e.g., "client.crt")
- `ssl_own_key`: private key of the client (e.g., "client.key")
- `ssl_own_key_pass`: password protecting the private key of the client (e.g., "blablabla")
- `cloud_config`: Astra-like configuration (e.g., `{'secure_connect_bundle': 'secure-connect-blabla.zip'}`)

### Performance tuning

This plugin offers extensive internal parallelism that can be adjusted
to enhance pipeline performance. Refer for example to this
[discussion](docs/LFN.md) on how to improve the throughput over a
long fat network.

## Data model

The main idea behind this plugin is that relatively small files can be
efficiently stored and retrieved as BLOBs in a NoSQL DB. This enables
scalability in data loading through prefetching and
pipelining. Furthermore, it enables data to be stored in a separate
location, potentially even at a significant distance from where it is
processed, as discussed [here](docs/LFN.md). This capability also
facilitates storing data along with a comprehensive set of associated
metadata, which can be more conveniently utilized during machine
learning.

For the sake of convenience and improved performance, we choose to
store data and metadata in separate tables within the database. The
*metadata* table will be utilized for selecting the images that need
to be processed. These images are identified by UUIDs and stored as
BLOBs in the *data* table. During the machine learning process, we
will exclusively access the *data* table. Below, you will find
examples of functional code for creating and populating these tables
in the database.

## Examples

### Classification

See the following annotated example for details on how to use this plugin:
- [Imagenette](examples/imagenette/)

#### Lightning

A variant of the same example implemented with PyTorch Lightning is available in:
- [Lightning](examples/lightning)

### Segmentation

A (less) annotated example for segmentation can be found in:
- [ADE20k](examples/ade20k/)

### Multilabel

An example showing how to save and decode multilabels as serialized
numpy tensors can be found in:
- [Corel-5k](examples/corel5k/)

### Split-file

An example of how to automatically create a single file with data
split to feed the training application:
- [Split-file](examples/splitfile)

### Inference with NVIDIA Triton

This plugin also supports efficient inference via [NVIDIA Triton
server](https://github.com/triton-inference-server/server):
- [Triton pipelines](examples/triton)

## Installation on a bare machine

cassandra-dali-plugin requires:
- NVIDIA DALI
- Cassandra C/C++ driver
- Cassandra Python driver

The details of how to install missing dependencies, in a system which
provides only some of the dependencies, can be deduced from the
[Dockerfile](Dockerfile), which contains all the installation
commands for the packages above.

**Once the dependencies have been installed**, the plugin
can easily be installed with pip:
```bash
$ pip3 install .
```

## Authors

Cassandra Data Loader is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>
  * Giovanni Busonera, CRS4 <giovanni.busonera@crs4.it>

## License

cassandra-dali-plugin is licensed under the under the Apache License,
Version 2.0. See LICENSE for further details.

## Acknowledgment

- Jakob Progsch for his [ThreadPool code](https://github.com/progschj/ThreadPool)
