# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import CassandraClassificationWriter
from tqdm import tqdm
import io
import numpy as np
import os
import os
import uuid

def_size = 256


def get_data(img_format="JPEG", img_size=def_size, crop=False):
    # img_format:
    # - UNCHANGED: unchanged input files (no resizing and cropping)
    # - JPEG: compressed JPEG
    # - PNG: compressed PNG
    # - TIFF: non-compressed TIFF
    def r(path):
        if img_format == "UNCHANGED":
            # just return the unchanged raw file
            with open(path, "rb") as fh:
                out_stream = io.BytesIO(fh.read())
        else:  # resize and, optionally, center crop
            img = Image.open(path).convert("RGB")
            sz = np.array(img.size)
            min_d = sz.min()
            sc = float(img_size) / min_d
            new_sz = (sc * sz).astype(int)
            img = img.resize(new_sz)
            if crop:
                off = (new_sz.max() - img_size) // 2
                if new_sz[0] > new_sz[1]:
                    box = [off, 0, off + img_size, img_size]
                else:
                    box = [0, off, img_size, off + img_size]
                img = img.crop(box)
            # save to stream
            out_stream = io.BytesIO()
            img.save(out_stream, format=img_format)
        # return raw file
        out_stream.flush()
        data = out_stream.getvalue()
        return data

    return r


def get_jobs(src_dir, splits=["train", "val"]):
    jobs = []
    labels = dict()

    ln = 0  # next-label number
    for or_split in splits:
        sp_dir = os.path.join(src_dir, or_split)
        subdirs = [d.name for d in os.scandir(sp_dir) if d.is_dir()]
        for or_label in subdirs:
            # if label is new, assign a new number
            if or_label not in labels:
                labels[or_label] = ln
                ln += 1
            label = labels[or_label]
            partition_items = (or_split, or_label)
            cur_dir = os.path.join(sp_dir, or_label)
            fns = os.listdir(cur_dir)
            for fn in fns:
                path = os.path.join(cur_dir, fn)
                jobs.append((path, label, partition_items))
    return jobs


def send_images_to_db(
    username,
    password,
    img_format,
    keyspace,
    table_suffix,
    cloud_config=None,
    cassandra_ips=None,
    cassandra_port=None,
    img_size=def_size,
):
    auth_prov = PlainTextAuthProvider(username, password)

    def ret(jobs):
        cw = CassandraClassificationWriter(
            cloud_config=cloud_config,
            auth_prov=auth_prov,
            cassandra_ips=cassandra_ips,
            cassandra_port=cassandra_port,
            use_ssl=True,
            table_data=f"{keyspace}.data_{table_suffix}",
            table_metadata=f"{keyspace}.metadata_{table_suffix}",
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["or_split", "or_label"],
            get_data=get_data(img_format, img_size=img_size),
        )
        for path, label, partition_items in tqdm(jobs):
            cw.save_image(path, label, partition_items)

    return ret


def save_image_to_dir(target_dir, path, label, raw_data, table_suffix):
    out_dir = os.path.join(target_dir, str(label))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, str(uuid.uuid4()) + table_suffix)
    with open(out_name, "wb") as fd:
        fd.write(raw_data)


def save_images_to_dir(target_dir, img_format, img_size=def_size):
    if img_format == "JPEG":
        table_suffix = ".jpg"
    elif img_format == "PNG":
        table_suffix = ".png"
    elif img_format == "TIFF":
        table_suffix = ".tiff"
    elif img_format == "UNCHANGED":
        table_suffix = ".jpg"
    else:
        raise ("Supporting only JPEG, PNG, TIFF, and UNCHANGED")

    def ret(jobs):
        for path, label, _ in tqdm(jobs):
            raw_data = get_data(img_format, img_size=img_size)(path)
            save_image_to_dir(target_dir, path, label, raw_data, table_suffix)

    return ret
