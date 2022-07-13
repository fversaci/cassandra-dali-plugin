# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandradl import CassandraDataset, CassandraListManager

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
from time import sleep
import torch
from torchvision import transforms
import os

# Read Cassandra parameters
try:
    from private_data import cassandra_ips, cass_user, cass_pass
except ImportError:
    cassandra_ip = getpass("Insert Cassandra's IP address: ")
    cassandra_ips = [cassandra_ip]
    cass_user = getpass("Insert Cassandra user: ")
    cass_pass = getpass("Insert Cassandra password: ")

# Init Cassandra dataset
ap = PlainTextAuthProvider(username=cass_user, password=cass_pass)

cd = CassandraDataset(ap, cassandra_ips, gpu_id=1,
                      tcp_connections=4, threads=32, prefetch_buffers=32)

dataset_nm="imagenette"
suff="_jpg"

if dataset_nm=="imagenet":
    num_classes = 1000
elif dataset_nm=="imagenette":
    num_classes = 10
else:
    raise RuntimeError(f"Dataset {dataset_nm} not supported.")

split_fn = f"{dataset_nm}{suff}.split"
if not os.path.exists(split_fn):
    # Create one split
    id_col = "patch_id"
    label_col = "label"
    num_classes = 1000
    clm = CassandraListManager(ap, cassandra_ips)
    clm.set_config(
        table=f"{dataset_nm}.ids_224{suff}",
        id_col=id_col,
        label_col=label_col,
        num_classes=num_classes,
    )
    clm.read_rows_from_db()
    clm.split_setup(split_ratios=[1])
    cd.use_splits(clm)
    cd.set_config(
        table=f"{dataset_nm}.data_224{suff}",
        bs=128,
        id_col=id_col,
        label_col=label_col,
        num_classes=num_classes,
        #rgb=True,
    )
    cd.save_splits(split_fn)
else:
    cd.load_splits(split_fn)
    cd.set_config(bs=128)


cd.rewind_splits(shuffle=True)

for _ in trange(10):
    for i in range(cd.num_batches[0]):
        x, y = cd.load_batch()
    cd.rewind_splits(shuffle=True)

#exit()

# RGB with augmentations
aug_fn = "/tmp/aug.pt"
# rescale and normalize
n_scale = 255.  # divide by 255
n_mean = (n_scale*np.array((0.485, 0.456, 0.406))).tolist()
n_std = (n_scale*np.array((0.229, 0.224, 0.225))).tolist()
aug = torch.nn.Sequential(
    transforms.Normalize(n_mean, n_std, inplace=True),
    transforms.GaussianBlur(5),
)
s_aug = torch.jit.script(aug)
s_aug.save(aug_fn)

cd.set_config(rgb=True, augs=[aug_fn])

cd.rewind_splits(shuffle=True)

for _ in trange(10):
    for i in range(cd.num_batches[0]):
        x, y = cd.load_batch()
    cd.rewind_splits(shuffle=True)
