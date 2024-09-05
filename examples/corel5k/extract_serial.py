# Copyright 2024 CRS4 (http://www.crs4.it/)
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

import extract_common
from clize import run


def save_images(
    src_dir,
    npy_dir,
    label_file,
    *,
    img_format="UNCHANGED",
    data_table="corel5k.data",
    metadata_table="corel5k.metadata",
):
    """Save center-cropped images to Cassandra DB or directory

    :param src_dir: Input directory of images
    :param npy_dir: Directory which will contain the labels as npy files
    :param label_file: JSON with the labels
    :param img_format: Format of output images
    :param data_table: Name of datatable (i.e.: keyspace.tablename)
    :param metadata_table: Name of metadatatable (i.e.: keyspace.tablename)
    """
    jobs = extract_common.get_jobs(src_dir, npy_dir, label_file)
    # Read Cassandra parameters
    from private_data import cass_conf

    extract_common.send_images_to_db(
        cass_conf=cass_conf,
        img_format=img_format,
        data_table=data_table,
        metadata_table=metadata_table,
    )(jobs)


# parse arguments
if __name__ == "__main__":
    run(save_images)
