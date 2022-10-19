# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import CassandraWriter
from tqdm import tqdm
import io
import numpy as np
import os
import os
from pathlib import Path
import uuid


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


def send_images_to_db(username, password, img_format, keyspace,
                      table_suffix, cloud_config=None,
                      cassandra_ips=None, cassandra_port=None, ):
    auth_prov = PlainTextAuthProvider(username, password)

    def ret(jobs):
        cw = CassandraWriter(
            cloud_config=cloud_config,
            auth_prov=auth_prov,
            cassandra_ips=cassandra_ips,
            cassandra_port=cassandra_port,
            table_data=f"{keyspace}.data_{table_suffix}",
            table_metadata=f"{keyspace}.metadata_{table_suffix}",
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["filename"],
            get_data=get_data(img_format),
            masks=True,
        )
        for path_img, path_mask in tqdm(jobs):
            cw.save_image(path_img, path_mask, (path_img,))

    return ret

