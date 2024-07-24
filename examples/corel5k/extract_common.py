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
from crs4.cassandra_utils import CassandraSegmentationWriter
from tqdm import tqdm
import io
import numpy as np
import os
import json
from pathlib import Path


def get_data(img_format="JPEG"):
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
        else:  # resize and crop to 224x224
            img = Image.open(path).convert("RGB")
            tg = 224
            sz = np.array(img.size)
            min_d = sz.min()
            sc = float(tg) / min_d
            new_sz = (sc * sz).astype(int)
            img = img.resize(new_sz)
            off = (new_sz.max() - tg) // 2
            if new_sz[0] > new_sz[1]:
                box = [off, 0, off + tg, tg]
            else:
                box = [0, off, tg, off + tg]
            img = img.crop(box)
            # save to stream
            out_stream = io.BytesIO()
            img.save(out_stream, format=img_format)
        # return raw file
        out_stream.flush()
        data = out_stream.getvalue()
        return data

    return r


def get_jobs(src_dir, npy_dir, label_file):
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    jobs = []
    with open(label_file) as f:
        train = json.load(f)
    # map labels to numbers
    labels = dict()
    for n, l in enumerate(train["labels"]):
        labels[l] = n
    sz = len(labels)  # 260 possible labels
    # save npy labels for each image
    samples = train["samples"]
    for sample in samples:
        fn = sample["image_name"]
        path_img = os.path.join(src_dir, fn)
        labs = sample["image_labels"]
        labs = [labels[l] for l in labs]  # convert to numbers
        tens_lab = np.zeros(sz)
        tens_lab[labs] = 1
        path_npy = os.path.join(npy_dir, fn)
        path_npy = Path(path_npy).with_suffix(".npy")
        np.save(path_npy, tens_lab)
        path_img = str(path_img)
        path_npy = str(path_npy)
        jobs.append((path_img, path_npy))

    return jobs


def send_images_to_db(
    cass_conf,
    img_format,
    data_table,
    metadata_table,
):
    def ret(jobs):
        cw = CassandraSegmentationWriter(
            cass_conf=cass_conf,
            data_table=data_table,
            metadata_table=metadata_table,
            data_id_col="patch_id",
            data_label_col="label",
            data_col="data",
            cols=["filename"],
            get_data=get_data(img_format),
        )
        for path_img, path_mask in tqdm(jobs):
            cw.enqueue_image(path_img, path_mask, (path_img,))
        cw.send_enqueued()

    return ret
