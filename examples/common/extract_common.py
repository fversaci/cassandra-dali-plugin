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

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import CassandraClassificationWriter
from tqdm import tqdm
import io
import numpy as np
import os
import uuid

def_size = 256


def get_data(img_format="UNCHANGED", img_size=def_size, crop=False):
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
    cass_conf,
    img_format,
    data_table,
    metadata_table,
    img_size=def_size,
):
    def ret(jobs):
        cw = CassandraClassificationWriter(
            cass_conf=cass_conf,
            data_table=data_table,
            metadata_table=metadata_table,
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["or_split", "or_label"],
            get_data=get_data(img_format, img_size=img_size),
        )
        for path, label, partition_items in tqdm(jobs):
            cw.enqueue_image(path, label, partition_items)
        cw.send_enqueued()

    return ret


def save_image_to_dir(target_dir, path, label, raw_data, file_ext):
    out_dir = os.path.join(target_dir, str(label))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, str(uuid.uuid4()) + file_ext)
    with open(out_name, "wb") as fd:
        fd.write(raw_data)


def save_images_to_dir(target_dir, img_format, img_size=def_size):
    if img_format == "JPEG":
        file_ext = ".jpg"
    elif img_format == "PNG":
        file_ext = ".png"
    elif img_format == "TIFF":
        file_ext = ".tiff"
    elif img_format == "UNCHANGED":
        file_ext = ".jpg"
    else:
        raise ("Supporting only JPEG, PNG, TIFF, and UNCHANGED")

    def ret(jobs):
        for path, label, _ in tqdm(jobs):
            raw_data = get_data(img_format, img_size=img_size)(path)
            save_image_to_dir(target_dir, path, label, raw_data, file_ext)

    return ret
