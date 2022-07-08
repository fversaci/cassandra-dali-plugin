# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from getpass import getpass
import imagenette_common
from clize import run


def save_images(src_dir, img_format="JPEG", *, target_dir=None):
    """Save center-cropped images to Cassandra DB or directory

    :param src_dir: Input directory for Imagenette
    :param target_dir: Output directory for the cropped images
    """
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass("Insert Cassandra user: ")
        cass_pass = getpass("Insert Cassandra password: ")

    jobs = imagenette_common.get_jobs(src_dir)
    imagenette_common.send_images_to_db(
        cassandra_ip, cass_user, cass_pass, img_format)(jobs)
    # imagenette_common.save_images_to_dir(
    #     target_dir, img_format)(jobs)


# parse arguments
if __name__ == "__main__":
    run(save_images)
