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
from IPython import embed


def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


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
        batch_size=256,
        shard_id=0,
        num_shards=1,
    )
    for _ in range(1):
        user_data = []
        for raw_data in uuids:
            inputs = []
            infer = grpcclient.InferInput("Reader", raw_data.shape, "UINT64")
            infer.set_data_from_numpy(raw_data)
            inputs.append(infer)
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("DALI_OUTPUT_0"))
            # outputs.append(grpcclient.InferRequestedOutput("DALI_OUTPUT_1"))

            # Infer with requested Outputs
            triton_client.async_infer(
                model_name,
                inputs=inputs,
                callback=partial(callback, user_data),
                outputs=outputs,
                client_timeout=10,
            )
        for i in trange(len(uuids)):
            while len(user_data) == i:
                time.sleep(0.02)
        # print(user_data)


# parse arguments
if __name__ == "__main__":
    start_inferring()
