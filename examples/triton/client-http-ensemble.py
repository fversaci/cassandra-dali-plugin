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
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import numpy as np
from cassandra_reader_interactive import read_uuids
from crs4.cassandra_utils import get_shard
from tqdm import tqdm, trange
from IPython import embed


def start_inferring():
    try:
        triton_client = httpclient.InferenceServerClient(
            url="127.0.0.1:8000",
            verbose=False,
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "cass_to_inference"

    uuids = read_uuids(
        keyspace="imagenette",
        table_suffix="train_256_jpg",
        ids_cache_dir="ids_cache",
    )
    uuids, real_sz = get_shard(
        uuids,
        batch_size=64,
        shard_id=0,
        num_shards=1,
    )
    for _ in trange(3):
        async_requests = []
        for raw_data in uuids:
            inputs = []
            infer = httpclient.InferInput("UUID", raw_data.shape, "UINT64")
            infer.set_data_from_numpy(raw_data, binary_data=True)
            inputs.append(infer)

            # Infer with requested Outputs
            async_requests.append(
                triton_client.async_infer(
                    model_name,
                    inputs=inputs,
                )
            )
        for req in async_requests:
            result = req.get_result()


# parse arguments
if __name__ == "__main__":
    start_inferring()
