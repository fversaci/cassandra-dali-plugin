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

import extract_common
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from clize import run


def save_images(
    src_dir,
    *,
    img_format="UNCHANGED",
    data_table="imagenette.data_orig",
    metadata_table="imagenette.metadata_orig",
    split_subdir="train",
    target_dir=None,
    img_size=256,
):
    """Save resized images to Cassandra DB or directory

    :param src_dir: Input directory for Imagenette
    :param img_format: Format of output images
    :param data_table: Name of data table data in the format keyspace.tablename
    :param metadata_table: Name of data table data in the format keyspace.tablename
    :param target_dir: Output directory (when saving to filesystem)
    :param split_subdir: Subdir to be processed
    :param img_size: Target image size
    """
    splits = [split_subdir]
    jobs = extract_common.get_jobs(src_dir, splits)
    # run spark
    conf = SparkConf().setAppName("data-extract")
    # .setMaster("spark://spark-master:7077")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    par_jobs = sc.parallelize(jobs)
    if not target_dir:
        # Read Cassandra parameters
        from private_data import cass_conf

        par_jobs.foreachPartition(
            extract_common.send_images_to_db(
                cass_conf=cass_conf,
                img_format=img_format,
                data_table=data_table,
                metadata_table=metadata_table,
                img_size=img_size,
            )
        )
    else:
        par_jobs.foreachPartition(
            extract_common.save_images_to_dir(
                target_dir,
                img_format,
                img_size=img_size,
            )
        )


# parse arguments
if __name__ == "__main__":
    run(save_images)
