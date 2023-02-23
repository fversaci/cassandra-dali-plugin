# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import MiniListManager

# load cassandra-dali-plugin
import crs4.cassandra_utils
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pathlib

# varia
import os
import pickle
import numpy as np
import random
import math

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)


def get_uuids(
    keyspace,
    table_suffix,
    id_col="patch_id",
    label_type="int",
    label_col="label",
    data_col="data",
    shard_id=0,
):
    # Read Cassandra parameters
    from private_data import CassConf as CC

    # set uuids cache directory
    ids_cache = "ids_cache"
    rows_fn = os.path.join(ids_cache, f"{keyspace}_{table_suffix}.rows")

    # Load list of uuids from Cassandra DB...
    ap = PlainTextAuthProvider(username=CC.username, password=CC.password)
    if not os.path.exists(rows_fn):
        lm = MiniListManager(
            auth_prov=ap,
            cassandra_ips=CC.cassandra_ips,
            cloud_config=CC.cloud_config,
            port=CC.cassandra_port,
        )
        conf = {
            "table": f"{keyspace}.metadata_{table_suffix}",
            "id_col": id_col,
        }
        lm.set_config(conf)
        print("Loading list of uuids from DB... ", end="", flush=True)
        lm.read_rows_from_db()
        if shard_id == 0:
            if not os.path.exists(ids_cache):
                os.makedirs(ids_cache)
            lm.save_rows(rows_fn)
        stuff = lm.get_rows()
    else:  # ...or from the cached file
        print("Loading list of uuids from cached file... ", end="", flush=True)
        with open(rows_fn, "rb") as f:
            stuff = pickle.load(f)
    # init and return Cassandra reader
    uuids = stuff["row_keys"]
    real_sz = len(uuids)
    print(f" {real_sz} images")
    return uuids


def get_shard(
    uuids,
    batch_size,
    epoch=0,
    shard_id=0,
    num_shards=1,
    seed=0,
):
    random.seed(seed)
    for _ in range(epoch):
        random.seed(random.getrandbits(32))
    random.shuffle(uuids)
    real_sz = len(uuids)
    uuids = uuids_as_tensors(uuids, batch_size)
    pad_sz = uuids.size / 2  # padded size
    del_sz = pad_sz - real_sz
    num_batches = uuids.shape[0]
    shard_size = math.ceil(num_batches / num_shards)
    shard_begin = math.floor(shard_id * num_batches / num_shards)
    shard_end = shard_begin + shard_size
    shard_uuids = uuids[shard_begin:shard_end]
    shard_sz = shard_uuids.size / 2
    if shard_id == num_shards - 1:
        shard_sz -= del_sz

    return shard_uuids, shard_sz


def uuid2ints(uuid):
    # convert to CassUuid format
    i1 = int.from_bytes(uuid.bytes_le[:8], byteorder="little")
    i2 = int.from_bytes(uuid.bytes[8:], byteorder="big")
    return (i1, i2)


def uuids_as_tensors(uuids, bs):
    uuids = list(map(uuid2ints, uuids))  # convert uuids to ints
    uuids = np.array(uuids, dtype=np.uint64)
    uuids = np.pad(uuids, ((0, bs - len(uuids) % bs), (0, 0)), "edge")
    uuids = uuids.reshape([-1, bs, 2])
    return uuids


def get_cassandra_reader(
    keyspace,
    table_suffix,
    batch_size,
    id_col="patch_id",
    label_type="int",
    label_col="label",
    data_col="data",
    io_threads=2,
    prefetch_buffers=2,
    name="Reader",
    shuffle_after_epoch=True,
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
    use_ssl=False,  # True,
    ssl_certificate="",  # "node0.cer.pem",
):
    # Read Cassandra parameters
    from private_data import CassConf as CC

    table = f"{keyspace}.data_{table_suffix}"
    if CC.cloud_config:
        connect_bundle = CC.cloud_config["secure_connect_bundle"]
    else:
        connect_bundle = None

    cassandra_reader = fn.crs4.cassandra(
        name=name,
        cloud_config=connect_bundle,
        cassandra_ips=CC.cassandra_ips,
        cassandra_port=CC.cassandra_port,
        username=CC.username,
        password=CC.password,
        table=table,
        label_col=label_col,
        data_col=data_col,
        id_col=id_col,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
        use_ssl=use_ssl,
        ssl_certificate=ssl_certificate,
        label_type=label_type,
    )
    return cassandra_reader
