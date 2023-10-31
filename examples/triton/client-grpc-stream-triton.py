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


import sys
import gevent.ssl
import time
from functools import partial
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
from cassandra_reader import read_uuids
from crs4.cassandra_utils import get_shard
from tqdm import tqdm, trange
import queue
from IPython import embed


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def start_inferring():
    try:
        triton_client = grpcclient.InferenceServerClient(
            url="127.0.0.1:8001",
            verbose=False,
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "dali_cassandra"

    uuids = read_uuids(
        keyspace="imagenette",
        table_suffix="train_256_jpg",
        ids_cache_dir="ids_cache",
    )
    uuids, real_sz = get_shard(
        uuids,
        batch_size=1024,
        shard_id=0,
        num_shards=1,
    )
    user_data = UserData()
    triton_client.start_stream(callback=partial(callback, user_data))
    for _ in range(30):
        for raw_data in uuids:
            inputs = []
            infer = grpcclient.InferInput("Reader", raw_data.shape, "UINT64")
            infer.set_data_from_numpy(raw_data)
            inputs.append(infer)
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("DALI_OUTPUT_0"))
            outputs.append(grpcclient.InferRequestedOutput("DALI_OUTPUT_1"))

            # Infer with requested Outputs
            triton_client.async_stream_infer(
                model_name,
                inputs=inputs,
                outputs=outputs,
            )
        for raw_data in tqdm(uuids):
            data_item = user_data._completed_requests.get()
            # ten = data_item.as_numpy("DALI_OUTPUT_0")


# parse arguments
if __name__ == "__main__":
    start_inferring()
