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
    keyspace="imagenette",
    table_suffix="train_224_jpg",
    reader="cassandra",
    device_id=types.CPU_ONLY_DEVICE_ID,
    file_root=None,
):
    """Read images from DB or filesystem, in a tight loop

    :param keyspace: Cassandra keyspace (i.e., name of the dataset)
    :param table_suffix: Suffix for table names
    :param reader: "cassandra" or "file"
    :param device_id: DALI device id (>=0 for GPUs)
    :param file_root: File root to be used (only when reading from the filesystem)
    """
    if reader == "cassandra":
        chosen_reader = get_cassandra_reader(
            keyspace,
            table_suffix,
            prefetch_buffers=16,
            io_threads=8,
            # comm_threads=4,
            # copy_threads=4,
        )
    elif reader == "file":
        # alternatively: use fn.readers.file
        file_reader = fn.readers.file(
            file_root=file_root,
            name="Reader",
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
        batch_size=128,
        num_threads=4,
        device_id=device_id,
        prefetch_queue_depth=2,
        py_start_method="spawn",
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        images, labels = chosen_reader
        ####################################################################
        # - do not resize (images have already been already resized)
        # images = fn_decode(images)
        # images = fn_normalize(images)
        ####################################################################
        # - alternatively: resize images (if using originals)
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
    if reader == "cassandra":
        for _ in range(10):
            with tqdm(total=74) as pbar:
                while True:
                    try:
                        pl.run()
                        pbar.update(1)
                    except StopIteration:
                        pl.reset()
                        break
    else:
        bs = pl.max_batch_size
        steps = (pl.epoch_size()["Reader"] + bs - 1) // bs
        for _ in range(10):
            for _ in trange(steps):
                x, y = pl.run()

    ########################################################################
    # alternatively: use pytorch iterator
    # (note: decode of images must be enabled)
    ########################################################################
    # ddl = DALIGenericIterator(
    #     [pl],
    #     ["data", "label"],
    #     # reader_name="Reader", # works only with file reader
    #     size=9469,
    #     last_batch_padded=True,
    #     last_batch_policy=LastBatchPolicy.PARTIAL #FILL, PARTIAL, DROP
    # )
    # for _ in range(10):
    #     for data in tqdm(ddl):
    #         x, y = data[0]["data"], data[0]["label"]
    #     ddl.reset()  # rewind data loader


# parse arguments
if __name__ == "__main__":
    run(read_data)
