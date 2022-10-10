# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

### To insert in DB, run with, e.g.,
# python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --keyspace=imagenette --split-subdir=train --table-suffix=train_224_jpg

### To save files in a directory, run with, e.g.,
# python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --split-subdir=train --target-dir=/data/imagenette/train_224_jpg


from getpass import getpass
import extract_common
from clize import run


def save_images(
    src_dir,
    mask_dir,
    *,
    img_format="UNCHANGED",
    keyspace="pascal2",
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
        from private_data import cassandra_ips, username, password
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    extract_common.send_images_to_db(
        cassandra_ips,
        username,
        password,
        img_format,
        keyspace,
        table_suffix,
    )(jobs)
    

# parse arguments
if __name__ == "__main__":
    run(save_images)
