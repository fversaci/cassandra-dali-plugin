# Pascal VOC2012 dataset

In this segmentation example we will import the [Pascal VOC2012
dataset] (host.robots.ox.ac.uk/pascal/VOC/voc2012/) as a Cassandra
dataset and then read the data into NVIDIA DALI.

As a first step, the raw files are to be downloaded from one of the
following URL:
- https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset
- https://paperswithcode.com/dataset/voc-2012
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

In the following we will assume the original images are stored in the
`/data/VOC2012/`directory.

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
The following commands will inseter the original dataset in Cassandra
and use the plugin to read the images in NVIDIA DALI.

```bash
# - Create the tables in the Cassandra DB
$ cd examples/pascal2/
$ /cassandra/bin/cqlsh -f create_tables.cql

# - Fill the tables with data and metadata
$ python3 extract_serial.py /data/VOC2012/JPEGImages/ /data/VOC2012/SegmentationObject/ --table-suffix=orig

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
$ python3 loop_read.py --reader=file --image-root=/data/VOC2012/dali/images --mask-root=/data/VOC2012/dali/masks

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --reader=file --image-root=/data/VOC2012/dali/images --mask-root=/data/VOC2012/dali/masks --device-id=0
```
