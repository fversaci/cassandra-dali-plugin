# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import tqdm
import io
import numpy as np
import os
import uuid
import tifffile
import os
from cassandradl import CassandraWriter


def get_data(img_format="JPEG"):
    # img_format:
    # - JPEG: compressed JPEG, pixels as bytes
    # - TIFF: non-compressed TIFF, with pixels as bytes
    # - TIFF_FLOAT: non-compressed TIFF, with pixels as float32
    # - UNCHANGED: unchanged input files (no resizing and cropping)
    def r(path):
        if (img_format=="UNCHANGED"):
            # just return the raw file
            with open(path, "rb") as fh:
                out_stream = io.BytesIO(fh.read())
        else: # resize and crop to 224x224
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
            if img_format == "TIFF_FLOAT":
                np_img = np.array(img).astype(np.float32) / 255.0
                tifffile.imwrite(out_stream, np_img)
            else:
                img.save(out_stream, format=img_format)
        # write to db
        out_stream.flush()
        data = out_stream.getvalue()
        return data
    return r

def get_jobs(src_dir):
    jobs = []
    labels = dict()

    ln = 0  # next-label number
    for or_split in ["train", "val"]:
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


def send_images_to_db(cassandra_ips, cass_user, cass_pass, img_format,
                      dataset):
    if img_format == "JPEG":
        suff="224_jpg"
    elif img_format == "TIFF":
        suff="224_tiff"
    elif img_format == "TIFF_FLOAT":
        suff="224_float"
    elif img_format == "UNCHANGED":
        suff="orig"
    else:
        raise("Supporting only JPEG, TIFF, TIFF_FLOAT and UNCHANGED")
    auth_prov = PlainTextAuthProvider(cass_user, cass_pass)

    def ret(jobs):
        cw = CassandraWriter(
            auth_prov,
            cassandra_ips,
            table_ids=f"{dataset}.ids_{suff}",
            table_data=f"{dataset}.data_{suff}",
            table_metadata=f"{dataset}.metadata_{suff}",
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["or_split", "or_label"],
            get_data=get_data(img_format),
        )
        for path, label, partition_items in tqdm(jobs):
            cw.save_image(path, label, partition_items)

    return ret

def save_image_to_dir(target_dir, path, label, raw_data, suff):
    out_dir = os.path.join(target_dir, str(label))
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)
    out_name= os.path.join(out_dir, str(uuid.uuid4())+suff)
    with open(out_name, "wb") as fd:
        fd.write(raw_data)

def save_images_to_dir(target_dir, img_format):
    if img_format == "JPEG":
        suff=".jpg"
    elif img_format == "TIFF":
        suff=".tiff"
    elif img_format == "TIFF_FLOAT":
        suff=".tiff"
    elif img_format == "UNCHANGED":
        suff=".jpg"
    else:
        raise("Supporting only JPEG, TIFF, TIFF_FLOAT and UNCHANGED")
    def ret(jobs):
        for path, label, _ in tqdm(jobs):
            raw_data = get_data(img_format)(path)
            save_image_to_dir(target_dir, path, label, raw_data, suff)

    return ret

