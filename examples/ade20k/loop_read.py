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

# supporting torchrun
import os

global_rank = int(os.getenv("RANK", default=0))
local_rank = int(os.getenv("LOCAL_RANK", default=0))
world_size = int(os.getenv("WORLD_SIZE", default=1))


def read_data(
    *,
    data_table="ade20k.data_orig",
    metadata_table="ade20k.metadata_orig",
    ids_cache_dir="ids_cache",
    reader="cassandra",
    use_gpu=False,
    image_root=None,
    mask_root=None,
    epochs=10,
):
    """Read images from DB or filesystem, in a tight loop

    :param data_table: Name of the data table (in the form: keyspace.tablename)
    :param metadata_table: Name of the data metadata table (in the form: keyspace.tablename)
    :param reader: "cassandra" or "file" (default: cassandra)
    :param use_gpu: enable output to GPU (default: False)
    :param image_root: File root for images (only when reading from the filesystem)
    :param mask_root: File root for masks (only when reading from the filesystem)
    :param ids_cache_dir: Directory containing the cached list of UUIDs (default: ./ids_cache)
    """
    if use_gpu:
        device_id = local_rank
    else:
        device_id = types.CPU_ONLY_DEVICE_ID

    bs = 128
    source_uuids = read_uuids(
        metadata_table,
        ids_cache_dir=ids_cache_dir,
    )
    if reader == "cassandra":
        db_reader = get_cassandra_reader(
            data_table=data_table,
            prefetch_buffers=16,
            io_threads=8,
            label_type="blob",
            name="Reader",
            # comm_threads=4,
            # copy_threads=4,
            # ooo=True,
            slow_start=4,
            source_uuids=source_uuids,
            shard_id=global_rank,
            num_shards=world_size,
        )
    elif reader == "file":
        # alternatively: use fn.readers.file
        print(image_root, " ", mask_root)
        file_reader = fn.readers.file(
            file_root=image_root,
            name="Reader",
            shard_id=global_rank,
            num_shards=world_size,
            pad_last_batch=True,
            # speed up reading
            prefetch_queue_depth=2,
            dont_use_mmap=True,
            read_ahead=True,
        )
        mask_reader = fn.readers.file(
            file_root=mask_root,
            shard_id=global_rank,
            num_shards=world_size,
            pad_last_batch=True,
            # speed up reading
            prefetch_queue_depth=2,
            dont_use_mmap=True,
            read_ahead=True,
        )
    else:
        raise ('--reader: expecting either "cassandra" or "file"')

    # create dali pipeline
    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=device_id,
        prefetch_queue_depth=2,
        # py_start_method="spawn",
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        if reader == "cassandra":
            images, labels = db_reader
        else:
            images, _ = file_reader
            labels, _ = mask_reader
        ####################################################################
        # images = fn_decode(images)
        # images = fn_resize(images)
        # labels = fn_decode(labels)
        # labels = fn_resize(labels)
        ####################################################################
        if device_id != types.CPU_ONLY_DEVICE_ID:
            images = images.gpu()
            labels = labels.gpu()
        return images, labels

    pl = get_dali_pipeline()
    pl.build()

    shard_size = math.ceil(len(source_uuids) / world_size)
    steps = math.ceil(shard_size / bs)
    ########################################################################
    # DALI iterator
    ########################################################################
    # produce images
    if reader == "cassandra":
        # consume uuids to get images from DB
        for _ in range(epochs):
            # read data for current epoch
            for _ in trange(steps):
                pl.run()
            pl.reset()
    else:
        for _ in range(epochs):
            for _ in trange(steps):
                x, y = pl.run()

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
