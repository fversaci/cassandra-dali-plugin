# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cassandra reader
from cassandra_reader import get_cassandra_reader, read_uuids

# dali
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

# some preconfigured operators
from fn_shortcuts import (
    fn_decode,
    fn_normalize,
    fn_image_random_crop,
    fn_resize,
    fn_crop_normalize,
)

# varia
from clize import run
from tqdm import trange, tqdm
import math
import time
import boto3
import statistics
import os
from IPython import embed
import numpy as np
import pickle

global_rank = int(os.getenv("RANK", default=0))
local_rank = int(os.getenv("LOCAL_RANK", default=0))
world_size = int(os.getenv("WORLD_SIZE", default=1))


def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")

    # Remove the "s3://" prefix
    s3_uri = s3_uri[5:]

    # Split the remaining part into bucket and prefix
    parts = s3_uri.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def list_s3_files(s3_uri):
    bucket_name, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    paths = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                paths.append(f"s3://{bucket_name}/{obj['Key']}")

    return sorted(paths)


def just_sleep(im1, im2):
    time.sleep(2e-5 * world_size)
    return im1, im2


def read_data(
    *,
    data_table="imagenette.data_train",
    rows_fn="train.rows",
    reader="cassandra",
    use_gpu=False,
    bs=128,
    epochs=10,
    file_root=None,
    index_root=None,
    out_of_order=True,
    slow_start=4,
    log_fn=None,
):
    """Read images from DB or filesystem, in a tight loop

    :param data_table: Name of the data table (in the form: keyspace.tablename)
    :param rows_fn: Filename of local copy of UUIDs (default: train.rows)
    :param reader: "cassandra", "file" or "tfrecord" (default: cassandra)
    :param use_gpu: enable output to GPU (default: False)
    :param bs: batch size (default: 128)
    :param epochs: Number of epochs (default: 10)
    :param file_root: File root to be used (only when reading files or tfrecords)
    :param index_root: Root path to index files (only when reading tfrecords)
    """
    if use_gpu:
        device_id = local_rank
    else:
        device_id = types.CPU_ONLY_DEVICE_ID

    if reader == "cassandra":
        source_uuids = read_uuids(
            rows_fn,
        )
        chosen_reader = get_cassandra_reader(
            data_table=data_table,
            prefetch_buffers=4,
            io_threads=12,
            name="Reader",
            comm_threads=1,
            copy_threads=4,
            ooo=out_of_order,
            slow_start=slow_start,
            source_uuids=source_uuids,
            shard_id=global_rank,
            num_shards=world_size,
        )
    elif reader == "file":
        # read images from filesystem or s3
        file_reader = fn.readers.file(
            file_root=file_root,
            name="Reader",
            shard_id=global_rank,
            num_shards=world_size,
            pad_last_batch=True,
            # speed up reading
            prefetch_queue_depth=4,
            # dont_use_mmap=True,
            read_ahead=True,
        )
        chosen_reader = file_reader
    elif reader == "tfrecord":
        # read tfrecords from filesystem or s3
        if file_root.startswith("s3://"):
            path = list_s3_files(file_root)
            index_path = list_s3_files(index_root)
        else:
            path = sorted([f.path for f in os.scandir(file_root) if f.is_file()])
            index_path = sorted([f.path for f in os.scandir(index_root) if f.is_file()])

        tf_reader = fn.readers.tfrecord(
            path=path,
            index_path=index_path,
            features={
                "image/encoded": tfrec.FixedLenFeature([], tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([], tfrec.int64, -1),
            },
            name="Reader",
            shard_id=global_rank,
            num_shards=world_size,
            pad_last_batch=True,
            # speed up reading
            prefetch_queue_depth=4,
            # dont_use_mmap=True,
            read_ahead=True,
        )
        chosen_reader = tf_reader["image/encoded"], tf_reader["image/class/label"]
    else:
        raise ('--reader: expecting either "cassandra", "file" or "tfrecord"')

    # create dali pipeline
    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=device_id,
        prefetch_queue_depth=2,
        #########################
        # - uncomment to enable delay via just_sleep
        # exec_async=False,
        # exec_pipelined=False,
        #########################
        # py_start_method="spawn",
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        images, labels = chosen_reader
        ####################################################################
        # - add a delay proportional to the number of ranks
        # images, labels = fn.python_function(
        #     images, labels, function=just_sleep, num_outputs=2
        # )
        ####################################################################
        # - decode, resize and crop, must use GPU (e.g., --use-gpu)
        # images = fn_image_random_crop(images)
        # images = fn_resize(images)
        # images = fn_crop_normalize(images)
        ####################################################################
        if device_id != types.CPU_ONLY_DEVICE_ID:
            images = images.gpu()
            labels = labels.gpu()
        return images, labels

    pl = get_dali_pipeline()
    pl.build()

    ########################################################################
    # DALI iterator
    ########################################################################
    shard_size = math.ceil(pl.epoch_size()["Reader"] / world_size)
    steps = math.ceil(shard_size / bs)
    timestamps_np = np.zeros((epochs, steps))
    batch_bytes_np = np.zeros((epochs, steps))
    first_epoch = True
    for epoch in range(epochs):
        # read data for current epoch
        with trange(steps) as t:
            start_ts = time.time()
            for step in t:
                images, labels = pl.run()
                images_batch_bytes = np.sum(np.array(images.shape()).reshape(bs))

                # Using tfrecord labels.shape is empty
                if labels.shape()[0]:
                    labels_batch_bytes = np.sum(np.array(labels.shape()).reshape(bs))
                else:
                    labels_batch_bytes = bs
                batch_bytes = images_batch_bytes + labels_batch_bytes

                batch_bytes_np[epoch][step] = batch_bytes
                timestamps_np[epoch, step] = time.time() - start_ts
                start_ts = time.time()

        pl.reset()
    # Calculate the average and standard deviation
    if epochs > 3:
        # First epoch is skipped
        ## Speed im/s
        average_io_GBs_per_epoch = (
            np.sum(batch_bytes_np[1:], axis=1) / np.sum(timestamps_np[1:], axis=1)
        ) / 1e9
        std_dev_io_GBs = np.std(average_io_GBs_per_epoch)
        average_time_per_epoch = np.mean(timestamps_np[1:], axis=(1))
        std_dev_time = np.std(average_time_per_epoch)
        average_speed_per_epoch = bs / average_time_per_epoch
        std_dev_speed = np.std(average_speed_per_epoch)

        print(f"Stats for epochs > 1, batch_size = {bs}")
        print(
            f"  Average IO: {np.mean(average_io_GBs_per_epoch):.2e} ± {std_dev_io_GBs:.2e} GB/s"
        )
        print(
            f"  Average batch time: {np.mean(average_time_per_epoch):.2e} ± {std_dev_time:.2e} s"
        )
        print(
            f"  Average speed: {np.mean(average_speed_per_epoch):.2e} ± {std_dev_speed:.2e} im/s"
        )

    if log_fn:
        data = (bs, timestamps_np, batch_bytes_np)
        pickle.dump(data, open(log_fn, "wb"))

    ########################################################################
    # alternatively: use pytorch iterator
    # (note: decode of images must be enabled)
    ########################################################################
    # ddl = DALIGenericIterator(
    #     [pl],
    #     ["data", "label"],
    #     reader_name="Reader",
    #     last_batch_policy=LastBatchPolicy.PARTIAL #FILL, PARTIAL, DROP
    # )
    # for _ in range(epochs):
    #     # consume data
    #     for data in tqdm(ddl):
    #         x, y = data[0]["data"], data[0]["label"]
    #     ddl.reset()  # rewind data loader


# parse arguments
if __name__ == "__main__":
    run(read_data)
