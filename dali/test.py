# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra
from cassandradl import CassandraDataset, CassandraListManager
from cassandra.auth import PlainTextAuthProvider

# dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.plugin_manager as plugin_manager

plugin_manager.load_library("./cpp/build/libcrs4cassandra.so")
# varia
import pickle
from tqdm import trange, tqdm
import getpass
import numpy as np
from time import sleep
import os
import torch
from torchvision import transforms

# Read Cassandra parameters
try:
    from private_data import cass_ips, cass_user, cass_pass
except ImportError:
    cass_ip = getpass("Insert Cassandra's IP address: ")
    cass_ips = [cassandra_ip]
    cass_user = getpass("Insert Cassandra user: ")
    cass_pass = getpass("Insert Cassandra password: ")

# read list of uuids
dataset_nm = "imagenet"
suff = "jpg"
split_fn = f"{dataset_nm}_{suff}.split"
with open(split_fn, "rb") as f:
    x = pickle.load(f)
uuids = x["row_keys"]
uuids = list(map(str, uuids))  # convert uuids to strings

cass_conf = [
    f"{dataset_nm}.data_224_{suff}",
    "label",
    "data",
    "patch_id",
    cass_user,
    cass_pass,
]

# test file reader
src_dir = os.path.join(f"/data/{dataset_nm}-cropped/", suff)

# create dali pipeline
@pipeline_def(
    batch_size=1,
    num_threads=10,
    device_id=1, #types.CPU_ONLY_DEVICE_ID,
    prefetch_queue_depth=4,
)
def get_dali_pipeline():
    # images, labels = fn.readers.file(file_root=src_dir, name="CassReader")
    images, labels = fn.crs4.cassandra(        
        name="CassReader",
        uuids=uuids,
        cass_conf=cass_conf,
        cass_ips=cass_ips,
        prefetch_queue_depth=4,
        tcp_connections=4,
        copy_threads=2,
    )
    # images = fn.decoders.image(
    #     images,
    #     device="mixed",
    #     output_type=types.RGB,
    # )
    labels = labels.gpu()
    return images, labels


pl = get_dali_pipeline()
#pl.build()

ddl = DALIGenericIterator([pl], ["data", "label"], reader_name="CassReader")

for _ in range(100):
    for data in tqdm(ddl):
        x = data[0]["data"]
        y = data[0]["label"]
    ddl.reset()  # rewind data loader
