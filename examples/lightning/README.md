# Imagenette Dataset

In this example we will import the [Imagenette2-320
dataset](https://github.com/fastai/imagenette) (a subset of ImageNet)
as a Cassandra dataset and then read the data into NVIDIA DALI.

The raw files are already present in the `/tmp` directory of the
provided [Docker container](../../README.md#running-the-docker-container),
from which the following commands can be run.

## Starting Cassandra server
We begin by starting the Cassandra server shipped with the provided
Docker container:

```bash
# Start Cassandra server
$ /cassandra/bin/cassandra

```

Note that the shell prompt is immediately returned.  Wait until `state
jump to NORMAL` is shown (about 1 minute).

## Resized dataset
The following commands will create a resized dataset in Cassandra
(with minimum dimension equal to 256 pixels) and use the plugin to
read the images in NVIDIA DALI.

```bash
# - Create the tables in the Cassandra DB
$ cd examples/imagenette/
$ /cassandra/bin/cqlsh -f create_tables.cql

# - Fill the tables with data and metadata
$ python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --table-suffix=train_256_jpg
$ python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --table-suffix=val_256_jpg

# Read the list of UUIDs and cache it to disk
$ python3 cache_uuids.py --table-suffix=train_256_jpg

# - Tight loop data loading test in host memory
$ python3 loop_read.py --table-suffix=train_256_jpg

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --table-suffix=train_256_jpg --use-gpu

# - Sharded, tight loop data loading test, using 2 processes via torchrun
$ torchrun --nproc_per_node=2 loop_read.py --table-suffix=train_256_jpg
```

## Compare with DALI fn.readers.file
The same scripts can be used to save the pre-processed images in the
filesystem and to read them using the standard DALI file reader.

```bash
# - Save the resized files in the filesystem
$ python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --target-dir=/data/imagenette/train_256_jpg
$ python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --target-dir=/data/imagenette/val_256_jpg

# - Tight loop data loading test in host memory
$ python3 loop_read.py --reader=file --file-root=/data/imagenette/train_256_jpg

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --reader=file --file-root=/data/imagenette/train_256_jpg --use-gpu

# - Sharded, tight loop data loading test, using 2 processes via torchrun
$ torchrun --nproc_per_node=2 loop_read.py --reader=file --file-root=/data/imagenette/train_256_jpg
```

## Storing the unchanged images in the DB (no resize)
We can also store the original, unchanged files in the DB:

```bash
# - Fill the tables with data and metadata
$ python3 extract_serial.py /tmp/imagenette2-320 --img-format=UNCHANGED --split-subdir=train --table-suffix=train_orig
$ python3 extract_serial.py /tmp/imagenette2-320 --img-format=UNCHANGED --split-subdir=val --table-suffix=val_orig

# Read the list of UUIDs and cache it to disk
$ python3 cache_uuids.py --table-suffix=train_orig

# - Tight loop data loading test in host memory
$ python3 loop_read.py --table-suffix=train_orig

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --table-suffix=train_orig --use-gpu
```

## Insert Imagenet dataset in parallel (with Apache Spark)
The same scripts can also be used to process the full ImageNet dataset
(166 GB), which can be downloaded from
[Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
We assume that the dataset has been stored in
`/data/imagenet/{train,val}`, with `val` having the same directory
structure as `train` (i.e., non-flat, with labels as subdirs).

Since this dataset is much larger, it is convenient to process it in
parallel, using Apache Spark (pre-installed in the Docker container).

```bash
# - Start Spark master+worker
$ /spark/sbin/start-master.sh
$ /spark/sbin/start-worker.sh spark://$HOSTNAME:7077

# - Create the tables in the Cassandra DB
$ /cassandra/bin/cqlsh -f create_tables.imagenet.cql

# - Fill the tables in parallel (10 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
  --py-files extract_common.py extract_spark.py /data/imagenet/ \
  --keyspace=imagenet --split-subdir=train --table-suffix=train_256_jpg
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
  --py-files extract_common.py extract_spark.py /data/imagenet/ \
  --keyspace=imagenet --split-subdir=val --table-suffix=val_256_jpg

# Read the list of UUIDs and cache it to disk
$ python3 cache_uuids.py --keyspace=imagenet --table-suffix=train_256_jpg

# - Tight loop data loading test in host memory
$ python3 loop_read.py --keyspace=imagenet --table-suffix=train_256_jpg

```

## Multi-GPU training

We have adapted NVIDIA DALI [ImageNet Training in PyTorch
example](https://github.com/NVIDIA/DALI/tree/main/docs/examples/use_cases/pytorch/resnet50)
to read data from Cassandra using our plugin.

The [original script](distrib_train_from_file.py) can be run with:
```bash
# Original script, reading from filesystem:
$ torchrun --nproc_per_node=NUM_GPUS distrib_train_from_file.py \
  -a resnet50 --dali_cpu --b 64 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2 \
  /tmp/imagenette2-320/train /tmp/imagenette2-320/val
```

while [our modified version](distrib_train_from_cassandra.py) with:
```bash
# Read the lists of UUIDs and cache them to disk
$ python3 cache_uuids.py --table-suffix=train_orig
$ python3 cache_uuids.py --table-suffix=val_orig

# Modified script, reading from Cassandra:
$ python3 distrib_train_from_cassandra.py --num-gpu NUM_GPUS \
  -a resnet50 --dali_cpu --b 64 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2 \
  --keyspace=imagenette --train-table-suffix=train_orig --val-table-suffix=val_orig 
