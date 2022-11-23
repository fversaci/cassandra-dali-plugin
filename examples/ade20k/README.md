# Pascal VOC2012 dataset

In this segmentation example we will import the [ADE20K Outdoors
dataset](https://www.kaggle.com/datasets/residentmario/ade20k-outdoors)
as a Cassandra dataset and then read the data into NVIDIA DALI.

As a first step, the raw files are to be downloaded from:
- https://www.kaggle.com/datasets/residentmario/ade20k-outdoors

or, if you have installed [Kaggle API](https://www.kaggle.com/docs/api), you
can just run this command:

```bash
$ kaggle datasets download -d residentmario/ade20k-outdoors
```

In the following we will assume the original images are stored in the
`/data/ade20k/`directory.

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
$ cd examples/ade20k/
$ /cassandra/bin/cqlsh -f create_tables.cql

# - Fill the tables with data and metadata
$ python3 extract_serial.py /data/ade20k/images/training/ /data/ade20k/annotations/training/ --table-suffix=orig

# - Tight loop data loading test in host memory
$ python3 loop_read.py --table-suffix=orig

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --table-suffix=orig --device-id=0
```

## Compare with DALI fn.readers.file
The same scripts can be used to read the dataset from the filesystem,
using the standard DALI file reader.

```bash
# - Tight loop data loading test in host memory
$ python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --reader=file --image-root=/data/ade20k/images/ --mask-root=/data/ade20k/annotations/ --device-id=0
```
