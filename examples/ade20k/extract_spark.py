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

from getpass import getpass
import extract_common
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from clize import run


def save_images(
    src_dir,
    mask_dir,
    *,
    img_format="UNCHANGED",
    keyspace="ade20k",
    table_suffix="orig",
):
    """Save center-cropped images to Cassandra DB or directory

    :param src_dir: Input directory of images
    :param mask_dir: Input directory of masks
    :param keyspace: Name of dataset (for the Cassandra table)
    :param table_suffix: Suffix for table names
    :param img_format: Format of output images
    """
    jobs = extract_common.get_jobs(src_dir, mask_dir)
    # run spark
    conf = SparkConf().setAppName("data-extract")
    # .setMaster("spark://spark-master:7077")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    par_jobs = sc.parallelize(jobs)

    try:
        # Read Cassandra parameters
        from private_data import CassConf as CC
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    par_jobs.foreachPartition(
        extract_common.send_images_to_db(
            cloud_config=CC.cloud_config,
            cassandra_ips=CC.cassandra_ips,
            cassandra_port=CC.cassandra_port,
            username=CC.username,
            password=CC.password,
            img_format=img_format,
            keyspace=keyspace,
            table_suffix=table_suffix,
        )
    )


# parse arguments
if __name__ == "__main__":
    run(save_images)
