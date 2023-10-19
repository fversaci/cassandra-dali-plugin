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


from cassandra_reader import read_uuids
from crs4.cassandra_utils import get_shard
import json


def save_to_json(in_name="Reader"):
    uuids = read_uuids(
        keyspace="imagenette",
        table_suffix="train_256_jpg",
        ids_cache_dir="ids_cache",
    )
    uuids, real_sz = get_shard(
        uuids,
        batch_size=128,
        shard_id=0,
        num_shards=1,
    )
    l = list()
    j = dict()
    j["data"] = l
    for u in uuids:
        b = list()
        for p in u:
            p = p.tolist()
            d = dict()
            d[in_name] = p
            b.append(d)
        l.append(b)
    # save as json
    with open("uuids.json", "w") as f:
        json.dump(j, f)


# parse arguments
if __name__ == "__main__":
    save_to_json()
