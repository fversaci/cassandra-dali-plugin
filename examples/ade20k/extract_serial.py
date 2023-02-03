# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from getpass import getpass
import extract_common
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
    :param img_format: Format of output images
    :param keyspace: Name of dataset (for the Cassandra table)
    :param table_suffix: Suffix for table names
    """
    jobs = extract_common.get_jobs(src_dir, mask_dir)
    try:
        # Read Cassandra parameters
        from private_data import CassConf as CC
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    extract_common.send_images_to_db(
        cloud_config=CC.cloud_config,
        cassandra_ips=CC.cassandra_ips,
        cassandra_port=CC.cassandra_port,
        username=CC.username,
        password=CC.password,
        img_format=img_format,
        keyspace=keyspace,
        table_suffix=table_suffix,
    )(jobs)


# parse arguments
if __name__ == "__main__":
    run(save_images)
