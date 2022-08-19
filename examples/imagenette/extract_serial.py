# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from getpass import getpass
import extract_common
from clize import run


def save_images(src_dir, *, img_format="JPEG", dataset="imagenette", target_dir=None):
    """Save center-cropped images to Cassandra DB or directory

    :param src_dir: Input directory for Imagenette
    :param dataset: Name of dataset (for the Cassandra table)
    :param img_format: Format for image saving
    :param target_dir: Output directory for the cropped images
    """
    jobs = extract_common.get_jobs(src_dir)
    if not target_dir:
        try:
            # Read Cassandra parameters
            from private_data import cassandra_ips, username, password
        except ImportError:
            cassandra_ip = getpass("Insert Cassandra's IP address: ")
            cassandra_ips = [cassandra_ip]
            username = getpass("Insert Cassandra user: ")
            password = getpass("Insert Cassandra password: ")

        extract_common.send_images_to_db(
            cassandra_ips, username, password, img_format, dataset
        )(jobs)
    else:
        extract_common.save_images_to_dir(target_dir, img_format)(jobs)


# parse arguments
if __name__ == "__main__":
    run(save_images)
