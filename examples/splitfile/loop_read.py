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
import pickle
import math

# supporting torchrun
import os

global_rank = int(os.getenv("RANK", default=0))
local_rank = int(os.getenv("LOCAL_RANK", default=0))
world_size = int(os.getenv("WORLD_SIZE", default=1))


def read_data(
    split_fn,
    *,
    use_index=0,
    use_gpu=False,
    epochs=10,
):
    """Read images from DB or filesystem, in a tight loop

    :param use_gpu: enable output to GPU (default: False)
    """
    if use_gpu:
        device_id = local_rank
    else:
        device_id = types.CPU_ONLY_DEVICE_ID

    data = pickle.load(open(split_fn, "rb"))
    keyspace = data["keyspace"]
    table_suffix = data["table_suffix"]
    row_keys = data["row_keys"]
    split = data["split"]
    source_uuids = row_keys[split[use_index]]
    source_uuids = list(source_uuids)
    
    bs = 128
    chosen_reader = get_cassandra_reader(
        keyspace,
        table_suffix,
        batch_size=bs,
        prefetch_buffers=4,
        io_threads=8,
        name="Reader",
        comm_threads=1,
        copy_threads=4,
        ooo=True,
        slow_start=4,
        source_uuids=source_uuids,
        shard_id=global_rank,
        num_shards=world_size,
    )

    # create dali pipeline
    @pipeline_def(
        batch_size=bs,
        num_threads=4,
        device_id=device_id,
        prefetch_queue_depth=2,
        #########################
        # py_start_method="spawn",
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        images, labels = chosen_reader

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

    shard_size = math.ceil(len(source_uuids)/world_size)
    steps = math.ceil(shard_size/bs)
    ########################################################################
    # DALI iterator
    ########################################################################
    # produce images
    # consume uuids to get images from DB
    for _ in range(epochs):
        # read data for current epoch
        for _ in trange(steps):
            pl.run()
        pl.reset()

    ########################################################################
    # alternatively: use pytorch iterator
    # (note: decode of images must be enabled)
    ########################################################################
    # ddl = DALIGenericIterator(
    #     [pl],
    #     ["data", "label"],
    #     # reader_name="Reader", # works only with file reader
    #     size=shard_size,
    #     last_batch_padded=True,
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
