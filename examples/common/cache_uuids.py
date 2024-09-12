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

import os
from clize import run
from crs4.cassandra_utils import MiniListManager
from private_data import cass_conf as CC


def cache_uuids(
    *,
    metadata_table,
    rows_fn,
    id_col="id",
):
    """Cache uuids from DB to local file (via pickle)

    :param metadata_table: Cassandra metadata table (i.e., keyspace.name_of_the_metadata_table)
    :param rows_fn: Filename of local copy of UUIDs
    :param id_col: Column containing the UUIDs
    """

    # Load list of uuids from Cassandra DB...
    lm = MiniListManager(
        cass_conf=CC,
    )
    conf = {
        "table": metadata_table,
        "id_col": id_col,
    }
    lm.set_config(conf)
    print("Loading list of uuids from DB... ", end="", flush=True)
    lm.read_rows_from_db()
    stuff = lm.get_rows()
    uuids = stuff["row_keys"]
    real_sz = len(uuids)
    print(f" {real_sz} images")
    lm.save_rows(rows_fn)
    print(f"Saved as {rows_fn}.")


# parse arguments
if __name__ == "__main__":
    run(cache_uuids)
