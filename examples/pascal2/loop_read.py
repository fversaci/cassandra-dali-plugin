# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra reader
from cassandra_reader import get_cassandra_reader

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


def read_data(
    *,
    keyspace="pascal2",
    table_suffix="orig",
    device_id=types.CPU_ONLY_DEVICE_ID,
):
    """Read images from DB or filesystem, in a tight loop

    :param keyspace: Cassandra keyspace (i.e., name of the dataset)
    :param table_suffix: Suffix for table names
    :param device_id: DALI device id (>=0 for GPUs)
    """
    chosen_reader = get_cassandra_reader(
        keyspace,
        table_suffix,
        prefetch_buffers=16,
        io_threads=8,
        label_type="image",
        # comm_threads=4,
        # copy_threads=4,
        name="Reader",
    )

    # create dali pipeline
    @pipeline_def(
        batch_size=128,
        num_threads=4,
        device_id=device_id,
        # prefetch_queue_depth=2,
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        images, labels = chosen_reader
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

    ########################################################################
    # DALI iterator
    ########################################################################
    bs = pl.max_batch_size
    steps = (pl.epoch_size()["Reader"] + bs - 1) // bs
    for _ in range(10):
        for _ in trange(steps):
            x, y = pl.run()
    return

    ########################################################################
    # alternatively: use pytorch iterator
    ########################################################################
    # ddl = DALIGenericIterator(
    #     [pl],
    #     ["data", "label"],
    #     reader_name="Reader",
    #     last_batch_policy=LastBatchPolicy.PARTIAL #FILL, PARTIAL, DROP
    # )
    # for _ in range(10):
    #     for data in tqdm(ddl):
    #         x, y = data[0]["data"], data[0]["label"]
    #     ddl.reset()  # rewind data loader


# parse arguments
if __name__ == "__main__":
    run(read_data)
