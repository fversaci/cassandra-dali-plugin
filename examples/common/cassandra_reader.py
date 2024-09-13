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

# load cassandra-dali-plugin
import crs4.cassandra_utils
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.fn as fn
import pathlib

# varia
import os
import pickle

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)


def read_uuids(rows_fn):
    print("Loading list of uuids from cached file... ", end="", flush=True)
    with open(rows_fn, "rb") as f:
        stuff = pickle.load(f)
    # init and return Cassandra reader
    uuids = stuff["row_keys"]
    real_sz = len(uuids)
    print(f" {real_sz} images")
    return uuids


def get_cassandra_reader(
    data_table,
    id_col="id",
    label_type="int",
    label_col="label",
    data_col="data",
    shard_id=0,
    num_shards=1,
    io_threads=2,
    prefetch_buffers=2,
    name="Reader",
    shuffle_every_epoch=True,
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
    ooo=False,
    slow_start=0,
    source_uuids=None,
    loop_forever=True,
):
    # Read Cassandra parameters
    from private_data import cass_conf as CC

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
        use_ssl=CC.use_ssl,
        ssl_certificate=CC.ssl_certificate,
        ssl_own_certificate=CC.ssl_own_certificate,
        ssl_own_key=CC.ssl_own_key,
        ssl_own_key_pass=CC.ssl_own_key_pass,
        table=data_table,
        label_col=label_col,
        data_col=data_col,
        id_col=id_col,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        num_shards=num_shards,
        shard_id=shard_id,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
        label_type=label_type,
        ooo=ooo,
        slow_start=slow_start,
        source_uuids=source_uuids,
        loop_forever=loop_forever,
        shuffle_every_epoch=shuffle_every_epoch,
    )
    return cassandra_reader
