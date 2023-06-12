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
import os
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


def get_jobs(src_dir, mask_dir, new_suffix=".jpg"):
    jobs = []
    fns = os.listdir(mask_dir)
    for fn in fns:
        path_img = os.path.join(src_dir, fn)
        path_mask = os.path.join(mask_dir, fn)
        if new_suffix:
            path_img = Path(path_img).with_suffix(new_suffix)
        path_img = str(path_img)
        path_mask = str(path_mask)
        jobs.append((path_img, path_mask))

    return jobs


def send_images_to_db(
    cass_conf,
    img_format,
    keyspace,
    table_suffix,
):
    def ret(jobs):
        cw = CassandraSegmentationWriter(
            cass_conf=cass_conf,
            table_data=f"{keyspace}.data_{table_suffix}",
            table_metadata=f"{keyspace}.metadata_{table_suffix}",
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["filename"],
            get_data=get_data(img_format),
        )
        for path_img, path_mask in tqdm(jobs):
            cw.enqueue_image(path_img, path_mask, (path_img,))
        cw.send_enqueued()

    return ret
