# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import MiniListManager

# dali
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.types as types

# load cassandra-dali-plugin
import crs4.cassandra_utils
import pathlib

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)

# varia
from clize import run
from tqdm import trange, tqdm
import getpass
import os
import pickle


def get_cassandra_reader(keyspace, table_suffix):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ips, username, password
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")
    # cache directory
    ids_cache = "ids_cache"
    if not os.path.exists(ids_cache):
        os.makedirs(ids_cache)
    rows_fn = os.path.join(ids_cache, f"{keyspace}_{table_suffix}.rows")
    # Load list of uuids from Cassandra DB
    ap = PlainTextAuthProvider(username=username, password=password)
    id_col = "patch_id"
    if not os.path.exists(rows_fn):
        lm = MiniListManager(ap, cassandra_ips)
        conf = {
            "table": f"{keyspace}.ids_{table_suffix}",
            "id_col": id_col,
        }
        lm.set_config(conf)
        print("Loading list of uuids from DB...")
        lm.read_rows_from_db()
        lm.save_rows(rows_fn)
        stuff = lm.get_rows()
    else:  # use cached file
        print("Loading list of uuids from cached file...")
        with open(rows_fn, "rb") as f:
            stuff = pickle.load(f)
    uuids = stuff["row_keys"]
    uuids = list(map(str, uuids))  # convert uuids to strings
    table = f"{keyspace}.data_{table_suffix}"
    cassandra_reader = fn.crs4.cassandra(
        name="CassReader",
        uuids=uuids,
        shuffle_after_epoch=True,
        cassandra_ips=cassandra_ips,
        table=table,
        label_col="label",
        data_col="data",
        id_col=id_col,
        username=username,
        password=password,
        tcp_connections=20,
        prefetch_buffers=64,
        copy_threads=4,
        wait_par=4,
        comm_par=2,
        # use_ssl=True,
        # ssl_certificate="node0.cer.pem",
    )
    return cassandra_reader


def fn_decode(images):
    return fn.decoders.image(
        images,
        device="cpu",
        output_type=types.RGB,
        # hybrid_huffman_threshold=100000,
        # memory_stats=True,
    )  # note: output is HWC (channels-last)


def fn_normalize(images):
    return fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )


def fn_image_random_crop(images):
    return fn.decoders.image_random_crop(
        images,
        device="mixed",
        output_type=types.RGB,
        hybrid_huffman_threshold=100000,
        random_aspect_ratio=[0.8, 1.25],
        random_area=[0.1, 1.0],
        # memory_stats=True,
    )  # note: output is HWC (channels-last)


def fn_resize(images):
    return fn.resize(
        images,
        resize_x=256,
        resize_y=256,
    )


def fn_crop_normalize(images):
    return fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )


def read_data(
    *,
    keyspace="imagenette",
    table_suffix="224_jpg",
    reader="cassandra",
    device_id=types.CPU_ONLY_DEVICE_ID,
    file_root=None,
):
    """Read images from DB or filesystem, in tight loop

    :param keyspace: Name of the dataset (i.e., used keyspace in Cassandra)
    :param table_suffix: Suffix for table names
    :param reader: "cassandra" o "file"
    :param device_id: Device id (>=0 for GPUs)
    :param file_root: File root to be used when reading from filesystem
    """
    if reader == "cassandra":
        chosen_reader = get_cassandra_reader(keyspace, table_suffix)
    elif reader == "file":
        # alternatively: use fn.readers.file
        file_reader = fn.readers.file(
            file_root=file_root,
            name="CassReader",
        )
        chosen_reader = file_reader
    else:
        raise ('--reader: expecting either "cassandra" or "file"')

    # create dali pipeline
    @pipeline_def(
        batch_size=128,
        num_threads=20,
        device_id=device_id,
        # prefetch_queue_depth=2,
        # enable_memory_stats=True,
    )
    def get_dali_pipeline():
        images, labels = chosen_reader
        ####################################################################
        # - do not resize (images have already been already resized)
        # images = fn_decode(images)
        # images = fn_normalize(images)
        ####################################################################
        # - alternatively: resize images
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
    bs = pl.max_batch_size
    steps = (pl.epoch_size()["CassReader"] + bs - 1) // bs
    for _ in range(10):
        for _ in trange(steps):
            x, y = pl.run()
    return

    ########################################################################
    # alternatively: use pytorch iterator
    ########################################################################

    ddl = DALIGenericIterator(
        [pl],
        ["data", "label"],
        reader_name="CassReader",
        # last_batch_policy=LastBatchPolicy.FULL, # or PARTIAL, DROP
    )
    for _ in range(10):
        for data in tqdm(ddl):
            x, y = data[0]["data"], data[0]["label"]
        ddl.reset()  # rewind data loader


# parse arguments
if __name__ == "__main__":
    run(read_data)
