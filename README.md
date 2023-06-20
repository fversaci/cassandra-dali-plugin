# Cassandra plugin for NVIDIA DALI

## Overview

This plugin enables data loading from an [Apache Cassandra NoSQL
database](https://cassandra.apache.org) to [NVIDIA Data Loading
Library (DALI)](https://github.com/NVIDIA/DALI) (which can be used to
load and preprocess images for PyTorch or TensorFlow).

## Running the docker container

The easiest way to test the cassandra-dali-plugin is by using the
provided [Dockerfile](Dockerfile) (derived from [NVIDIA PyTorch
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)),
which also contains NVIDIA DALI, Cassandra C++ and Python drivers,
a Cassandra server, PyTorch and Apache Spark, as shown in the commands below.

```bash
# Build and run cassandradl docker container
$ docker build -t cassandra-dali-plugin .
$ docker run --rm -it --cap-add=sys_admin cassandra-dali-plugin
```

Alternatively, for better performance and for data persistence, it is
advised to mount a host directory for Cassandra on a fast disk (e.g.,
`/mnt/fast_disk/cassandra`):

```bash
# Run cassandradl docker container with external data dir
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
    name,
    cassandra_ips, cassandra_port, username, password,
    use_ssl, ssl_certificate, cloud_config,
    table, label_col, data_col, id_col, label_type,
    prefetch_buffers,
    io_threads, comm_threads, copy_threads, wait_threads,
    ooo, slow_start,
)
```

Below, we'll provide a summary of the parameters' meanings. If you
prefer to skip this section, here you can find some [working
examples](README.md#examples).

### Basic parameters

- `name`: name of the module to be passed to DALI (e.g. "Reader")
- `cassandra_ips`: list of IP pointing to the DB (e.g., `["127.0.0.1"]`)
- `cassandra_port`: Cassandra TCP port (default: `9042`)
- `username`: username for Cassandra
- `password`: password for Cassandra
- `use_ssl`: use SSL to encrypt the trasnfers: True or False
- `ssl_certificate`: public-key of the Cassandra server (e.g., "node0.cer.pem")
- `cloud_config`: Astra-like configuration (e.g., `{'secure_connect_bundle': 'secure-connect-blabla.zip'}`)
- `table`: data table (e.g., `imagenet.data_orig`)
- `label_col`: name of the label column (e.g., `label`)
- `label_type`: type of label: "int", "image" or "none" ("int" is
  typically used for classification, "image" for segmentation)
- `data_col`: name of the data column (e.g., `data`)
- `id_col`: name of the UUID column (e.g., `img_id`)

### Long fat networks

Let's explore how to optimize our data loader for use across long fat
networks, i.e., networks that have a high [bandwidth-delay
product](https://en.wikipedia.org/wiki/Bandwidth-delay_product), e.g.,
100 ms latency and 10 Gb/s bandwidth.

For instance, imagine a setup where you have your Cassandra DB,
containing the required training images in datacenter A, while the
computing nodes with the GPUs are located in datacenter B, which may
even be far away in a different country.

To take advantage of such networks, it is crucial to have a deep
prefetch queue that can be processed in parallel. To this purporse,
our plugin provides the following configurable parameters:

- `prefetch_buffers`: the plugin employs multi-buffering, to hide the
  network latencies. Default: 2.
- `io_threads`: number of IO threads used by the Cassandra driver
  (which also limits the number of TCP connections). Default: 2.
- `comm_threads`: number of threads handling the
  communications. Default: 2.
- `copy_threads`: number of threads copying the data. Default: 2.

As an example, we loaded the original ImageNet dataset over a 25 GbE
network with an artificial latency of 100ms (set with `tc-netem`, with
no packet loss), using a `batch_size` of 512 and without any decoding
or preprocessing. Our test nodes (equipped with an Intel Xeon CPU
E5-2650 v4 @ 2.20GHz), achieved about 40 batches per second, which
translates to more than 20,000 images per second and a throughput of
roughly 20 Gb/s. Note that this throughput refers to a single python process,
and that in [a distributed training](examples/imagenette/README.md#multi-gpu-training)
there is such a process *for each GPU*. We used the following
parameters for the test:

- `prefetch_buffers`: 16
- `io_threads`: 8
- `comm_threads`: 1
- `copy_threads`: 4



#### Handling variance and packet loss

When sending packets at large distance across the internet it is
common to experience packet loss due to congested routes. This can
significantly impact throughput, especially when requesting a sequence
of transfers, as a delay in one transfer can stall the entire
pipeline. Prefetching can exacerbate this issue by producing an
initial burst of requests, leading to even higher packet loss.

To address these problems and enable high-bandwidth transfers over
long distances (i.e., high latencies), we have extended our code in
two ways:

1. We have developed an out-of-order version of the data loader that
   can be activated by setting `ooo=True`. This version of the loader
   returns the images as soon as they are received, *potentially
   altering their sequence and mixing different batches*.
2. We have implemented a parametrized diluted prefetching method that
   requests an additional image every `n` normal requests, thus
   limiting the initial burst. To activate it, set `slow_start=4`, for
   example.

## Examples

### Classification

See the following annotated example for details on how to use this plugin:
- [Imagenette](examples/imagenette/)

### Segmentation

A (less) annotated example for segmentation can be found in:
- [ADE20k](examples/ade20k/)

### Split-file

An example of how to automatically create a single file with data
split to feed the training application:
- [Split-file](examples/splitfile)

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
