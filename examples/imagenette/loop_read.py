# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra reader
from cassandra_reader import get_cassandra_reader, get_uuids
from crs4.cassandra_utils import get_shard

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
    keyspace="imagenette",
    table_suffix="train_256_jpg",
    reader="cassandra",
    use_gpu=False,
    file_root=None,
    epochs=10,
):
    """Read images from DB or filesystem, in a tight loop

    :param keyspace: Cassandra keyspace (i.e., name of the dataset)
    :param table_suffix: Suffix for table names
    :param reader: "cassandra" or "file" (default: cassandra)
    :param use_gpu: enable output to GPU (default: False)
    :param file_root: File root to be used (only when reading from the filesystem)
    """
    if use_gpu:
        device_id = local_rank
    else:
        device_id = types.CPU_ONLY_DEVICE_ID

    bs = 128
    if reader == "cassandra":
        uuids = get_uuids(
            keyspace,
            table_suffix,
            shard_id=local_rank,
        )
        uuids, real_sz = get_shard(
            uuids,
            batch_size=bs,
            shard_id=global_rank,
            num_shards=world_size,
        )
        chosen_reader = get_cassandra_reader(
            keyspace,
            table_suffix,
            batch_size=bs,
            prefetch_buffers=16,
            io_threads=8,
            name="Reader",
            # comm_threads=4,
            # copy_threads=4,
        )
    elif reader == "file":
        # alternatively: use fn.readers.file
        file_reader = fn.readers.file(
            file_root=file_root,
            name="Reader",
            shard_id=global_rank,
            num_shards=world_size,
            pad_last_batch=True,
            # speed up reading
            prefetch_queue_depth=2,
            dont_use_mmap=True,
            read_ahead=True,
        )
        chosen_reader = file_reader
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
        images, labels = chosen_reader
        ####################################################################
        # - decode, resize and crop, must use GPU (e.g., --device-id=0)
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

    if reader == "cassandra":
        # feed epoch 0 uuid to the pipeline
        for u in uuids:
            pl.feed_input("Reader[0]", u)

    ########################################################################
    # DALI iterator
    ########################################################################
    # produce images
    if reader == "cassandra":
        # consume uuids to get images from DB
        for _ in range(epochs):
            # feed next epoch to the pipeline
            for u in uuids:
                pl.feed_input("Reader[0]", u)
            # read data for current epoch
            for _ in trange(len(uuids)):
                pl.run()
            pl.reset()
    else:
        steps = pl.epoch_size()["Reader"] / (bs*world_size)
        steps = math.ceil(steps)
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
    #     # reader_name="Reader", # thid option works only with file reader
    #     size=real_sz,
    #     last_batch_padded=True,
    #     last_batch_policy=LastBatchPolicy.PARTIAL #FILL, PARTIAL, DROP
    # )
    # for _ in range(epochs):
    #     # feed next epoch to the pipeline
    #     if reader == "cassandra":
    #         for u in uuids:
    #             pl.feed_input("Reader[0]", u)
    #     # consume data
    #     for data in tqdm(ddl):
    #         x, y = data[0]["data"], data[0]["label"]
    #     ddl.reset()  # rewind data loader


# parse arguments
if __name__ == "__main__":
    run(read_data)
