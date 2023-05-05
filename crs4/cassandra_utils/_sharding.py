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

import random
import math
import numpy as np

def uuid2ints(uuid):
    # convert to CassUuid format
    i1 = int.from_bytes(uuid.bytes_le[:8], byteorder="little")
    i2 = int.from_bytes(uuid.bytes[8:], byteorder="big")
    return (i1, i2)


def uuids_as_tensors(uuids, bs):
    uuids = list(map(uuid2ints, uuids))  # convert uuids to ints
    uuids = np.array(uuids, dtype=np.uint64)
    uuids = np.pad(uuids, ((0, bs - len(uuids) % bs), (0, 0)), "edge")
    uuids = uuids.reshape([-1, bs, 2])
    return uuids


def get_shard(
    uuids,
    batch_size,
    epoch=0,
    shard_id=0,
    num_shards=1,
    seed=0,
):
    random.seed(seed + epoch)
    random.shuffle(uuids)
    real_sz = len(uuids)
    uuids = uuids_as_tensors(uuids, batch_size)
    pad_sz = uuids.size / 2  # padded size
    del_sz = pad_sz - real_sz
    num_batches = uuids.shape[0]
    shard_size = math.ceil(num_batches / num_shards)
    shard_begin = math.floor(shard_id * num_batches / num_shards)
    shard_end = shard_begin + shard_size
    shard_uuids = uuids[shard_begin:shard_end]
    shard_sz = shard_uuids.size / 2
    if shard_id == num_shards - 1:
        shard_sz -= del_sz

    return shard_uuids, shard_sz
