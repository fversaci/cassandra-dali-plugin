# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import MiniListManager

# dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.plugin_manager as plugin_manager

# load cassandra-dali-plugin
import crs4
import pathlib
plugin_path = pathlib.Path(crs4.__path__[-1])
plugin_path = plugin_path.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)

# varia
import pickle
from tqdm import trange, tqdm
import getpass
import os

# Read Cassandra parameters
try:
    from private_data import cass_ips, username, password
except ImportError:
    cass_ip = getpass("Insert Cassandra's IP address: ")
    cass_ips = [cassandra_ip]
    username = getpass("Insert Cassandra user: ")
    password = getpass("Insert Cassandra password: ")

# read list of uuids
dataset_nm = "imagenet"
suff = "224_jpg"

ids_cache = "ids_cache"
if not os.path.exists(ids_cache):
    os.makedirs(ids_cache)
rows_fn = os.path.join(ids_cache, f"{dataset_nm}_{suff}.rows")

# Load list of uuids from Cassandra DB
ap = PlainTextAuthProvider(username=username, password=password)

if not os.path.exists(rows_fn):
    id_col = "patch_id"
    label_col = "label"
    lm = MiniListManager(ap, cass_ips)
    conf = {
        "table": f"{dataset_nm}.ids_{suff}",
        "id_col": id_col,
    }
    lm.set_config(conf)
    print("Loading list of uuids from DB...")
    lm.read_rows_from_db()
    lm.save_rows(rows_fn)
    stuff = lm.get_rows()
else:
    print("Loading list of uuids from cached file...")
    with open(rows_fn, "rb") as f:
        stuff = pickle.load(f)

uuids = stuff["row_keys"]
uuids = list(map(str, uuids))  # convert uuids to strings
table = f"{dataset_nm}.data_{suff}"
label_col = "label"
data_col = "data"
id_col = "patch_id"

# alternatively: directory for fn.readers.file
src_dir = os.path.join(f"/data/{dataset_nm}-cropped/", suff)

# create dali pipeline
@pipeline_def(
    batch_size=128,
    num_threads=2,
    device_id=1,  # types.CPU_ONLY_DEVICE_ID,
    # prefetch_queue_depth=2,
    # enable_memory_stats=True,
)
def get_dali_pipeline():
    # images, labels = fn.readers.file(
    #     file_root=src_dir,
    #     name="CassReader",
    # )
    images, labels = fn.crs4.cassandra(
        name="CassReader",
        uuids=uuids,
        shuffle_after_epoch=True,
        cass_ips=cass_ips,
        table=table,
        label_col=label_col,
        data_col=data_col,
        id_col=id_col,
        username=username,
        password=password,
        tcp_connections=20,
        prefetch_buffers=64,
        copy_threads=4,
        wait_par=3,
        comm_par=2,
        # use_ssl=True,
    )
    # images = fn.decoders.image(
    #     images,
    #     device="cpu",
    #     output_type=types.RGB,
    #     # hybrid_huffman_threshold=100000,
    #     # memory_stats=True,
    # )  # note: output is HWC (channels-last)
    # images = fn.crop_mirror_normalize(images,
    #                    dtype=types.FLOAT,
    #                    output_layout="CHW",
    #                    mean=[0.485 * 255,0.456 * 255,0.406 * 255],
    #                    std=[0.229 * 255,0.224 * 255,0.225 * 255])
    ########################################################################
    # images = fn.decoders.image_random_crop(
    #     images,
    #     device="mixed",
    #     output_type=types.RGB,
    #     hybrid_huffman_threshold=100000,
    #     random_aspect_ratio=[0.8, 1.25],
    #     random_area=[0.1, 1.0],
    #     # memory_stats=True,
    # ) # note: output is HWC (channels-last)
    # images = fn.resize(images, resize_x=256, resize_y=256,
    #                    interp_type=types.INTERP_TRIANGULAR)
    # images = fn.crop_mirror_normalize(images.gpu(),
    #                    dtype=types.FLOAT,
    #                    output_layout="CHW",
    #                    crop=(224, 224),
    #                    mean=[0.485 * 255,0.456 * 255,0.406 * 255],
    #                    std=[0.229 * 255,0.224 * 255,0.225 * 255])
    ########################################################################
    images = images.gpu()  # redundant if already in gpu
    labels = labels.gpu()
    return images, labels


pl = get_dali_pipeline()
pl.build()

bs = pl.max_batch_size
steps = (pl.epoch_size()["CassReader"] + bs - 1) // bs

for _ in range(10):
    for _ in trange(steps):
        x, y = pl.run()
exit()


########################################################################
# pytorch iterator
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

# print(len(x))
