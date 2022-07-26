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
plugin_manager.load_library('./cpp/build/libcrs4cassandra.so')
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
dataset_nm="imagenette"
suff="_jpg"
split_fn = f"{dataset_nm}{suff}.split"
with open(split_fn, "rb") as f:
  x = pickle.load(f)
uuids = x['row_keys']
uuids = list(map(str, uuids)) # convert uuids to strings

cass_conf = [
    f"{dataset_nm}.ids_224{suff}",
    "label",
    "data",
    "patch_id",
    cass_user,
    cass_pass,
]

# create dali pipeline
@pipeline_def(batch_size=3, num_threads=1, device_id=1)
def get_dali_pipeline():
    images = fn.crs4.cassandra(name="CassReader", uuids=uuids,
                               cass_conf=cass_conf,
                               cass_ips=cass_ips)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    return images

pl = get_dali_pipeline()
pl.build()

ddl = DALIGenericIterator(
   [pl], ['data'],
   reader_name='CassReader'
)

for data in tqdm(ddl):
    x = data[0]['data']
ddl.reset() # rewind data loader
