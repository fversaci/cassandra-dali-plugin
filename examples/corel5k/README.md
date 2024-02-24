# Corel-5k multilabel image dataset

In this multilabel example we will import the [Corel-5k
dataset](https://www.kaggle.com/datasets/parhamsalar/corel5k) as a
Cassandra dataset and then read the data into NVIDIA DALI.  We will
save the original images as JPEG blobs and the labels as NPY blobs
(i.e., serialized numpy tensors).

As a first step, the raw files are to be downloaded from:
- https://www.kaggle.com/datasets/parhamsalar/corel5k

or, if you have installed [Kaggle API](https://www.kaggle.com/docs/api), you
can just run this command:

```bash
$ kaggle datasets download -d parhamsalar/corel5k
```

In the following we will assume the original images are stored in the
`/data/Corel-5k/`directory.

## Starting Cassandra server
We begin by starting the Cassandra server shipped with the provided
Docker container:

```bash
# Start Cassandra server
$ /cassandra/bin/cassandra

```

Note that the shell prompt is immediately returned.  Wait until `state
jump to NORMAL` is shown (about 1 minute).

## Storing the (unchanged) images in the DB
The following commands will insert the original dataset in Cassandra
and use the plugin to read the images in NVIDIA DALI.

```bash
# - Create the tables in the Cassandra DB
$ cd examples/corel5k/
$ /cassandra/bin/cqlsh -f create_tables.cql

# - Fill the tables with data and metadata
$ python3 extract_serial.py /data/Corel-5k/images/ /data/Corel-5k/npy_labs /data/Corel-5k/train.json --table-suffix=orig

# - Read the list of UUIDs and cache it to disk
$ python3 cache_uuids.py --table-suffix=orig

# - Tight loop, data loading and decoding in GPU memory (GPU:0)
$ python3 loop_read.py --table-suffix=orig --use-gpu

# - Sharded, tight loop test, using 2 GPUs via torchrun
$ torchrun --nproc_per_node=2 loop_read.py --table-suffix=orig --use-gpu
```
