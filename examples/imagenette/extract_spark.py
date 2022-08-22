# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

### To insert in DB, run with, e.g.,
# /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files extract_common.py extract_spark.py /tmp/imagenette2-320 --img-format=JPEG --keyspace=imagenette

### To save files in a directory, run with, e.g.,
# /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=20 --py-files extract_common.py extract_spark.py /tmp/imagenette2-320 --img-format=JPEG --target-dir=/data/imagenette/224_jpg

from getpass import getpass
import extract_common
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from clize import run


def save_images(
    src_dir,
    *,
    img_format="JPEG",
    keyspace="imagenette",
    target_dir=None,
    split="train",
):
    """Save center-cropped images to Cassandra DB or directory

    :param src_dir: Input directory for Imagenette
    :param img_format: Format of output images
    :param keyspace: Name of dataset (for the Cassandra table)
    :param target_dir: Output directory (is savinf to filesystem)
    :param split: Subdir to be processed
    """
    splits = [split]
    jobs = extract_common.get_jobs(src_dir, splits)
    # run spark
    conf = SparkConf().setAppName("data-extract")
    # .setMaster("spark://spark-master:7077")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    par_jobs = sc.parallelize(jobs)
    if not target_dir:
        try:
            # Read Cassandra parameters
            from private_data import cassandra_ips, username, password
        except ImportError:
            cassandra_ip = getpass("Insert Cassandra's IP address: ")
            cassandra_ips = [cassandra_ip]
            username = getpass("Insert Cassandra user: ")
            password = getpass("Insert Cassandra password: ")

        par_jobs.foreachPartition(
            extract_common.send_images_to_db(
                cassandra_ips, username, password, img_format, keyspace
            )
        )
    else:
        par_jobs.foreachPartition(
            extract_common.save_images_to_dir(target_dir, img_format)
        )


# parse arguments
if __name__ == "__main__":
    run(save_images)
