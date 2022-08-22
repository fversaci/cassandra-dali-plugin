# Imagenette Dataset

In this example we will import the [Imagenette2-320
dataset](https://github.com/fastai/imagenette) as a Cassandra dataset
and then read the data into NVIDIA DALI. 

The raw files are already present in the `/tmp` directory of the
provided [Docker container](../../), from which the following commands
can be run.

## Resize and center-crop
In this example the images will be resized and center-cropped to
224x224 pixels.

```bash
# - Create the tables in the Cassandra DB
$ cd examples/imagenette/
$ /cassandra/bin/cqlsh -f create_tables.cql

# - Fill the tables with data and metadata
$ python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --keyspace=imagenette

# - Tight loop data loading test in host memory
$ python3 loop_read.py

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --device-id=0
```

## Compare with DALI fn.readers.file
The same scripts can be used to save the pre-processed images in the
filesystem and to read them using the standard DALI file reader.

```bash
# - Save the center-cropped files in the filesystem
$ python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --target-dir=/data/imagenette/224_jpg

# - Tight loop data loading test in host memory
$ python3 loop_read.py --reader=file --file-root=/data/imagenette/224_jpg

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --device-id=0 --reader=file --file-root=/data/imagenette/224_jpg
```

## Storing the unchanged images in the DB (no resize)
We can also store the original, unchanged images in the DB:

```bash
# - Fill the tables with data and metadata
$ python3 extract_serial.py /tmp/imagenette2-320 --img-format=UNCHANGED --keyspace=imagenette

# - Tight loop data loading test in host memory
$ python3 loop_read.py --table-suffix=orig

# - Tight loop data loading test in GPU memory (GPU:0)
$ python3 loop_read.py --table-suffix=orig --device-id=0
```

## Insert Imagenet dataset in parallel (with Apache Spark)
The same scripts can also be used to process the full ImageNet dataset
(166 GB), which can be downloaded from
[Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).

Since this dataset is much larger, we will process it in parallel
using Apache Spark (pre-installed in the Docker container).  We assume
that the dataset has been unzipped in the `/data/imagenet/` directory.

```bash
# - Start Spark master+worker
$ sudo /spark/sbin/start-master.sh
$ sudo /spark/sbin/start-worker.sh spark://$HOSTNAME:7077

# - Create the tables in the Cassandra DB
$ /cassandra/bin/cqlsh -f create_tables.imagenet.cql

# - Fill the tables in parallel (10 jobs) with Spark
$ /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files extract_common.py extract_spark.py /data/imagenet/ILSVRC/Data/CLS-LOC/ --img-format=JPEG --keyspace=imagenet

# - Tight loop data loading test in host memory
$ python3 loop_read.py --keyspace=imagenet

```

## Tuning cassandra-dali-plugin internal parallelism

The plugin has a (rather rich) internal parallelism which, when
necessary, can be finely tuned to improve the maximum throughput.

These are the main configurable parameters of the `fn.crs4.cassandra`
class, which might help to remove data loading bottlenecks:
- `prefetch_buffers`: the plugin employs multi-buffering, to help
  cover the network latencies. Default: 2.
- `io_threads`: number of IO threads used by the Cassandra driver
  (which also limits the number of TCP connections). Default: 2.
- `comm_threads`: number of threads handling the communication. Default: 2.
- `copy_threads`: number of threads copying the data. Default: 2.
