# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('./cpp/build/libcrs4cassandra.so')

from tqdm import trange, tqdm
import numpy as np
from time import sleep
import os
import torch
from torchvision import transforms

#help(fn.crs4.cassandra)

@pipeline_def(batch_size=3, num_threads=1, device_id=1)
def get_dali_pipeline():
    images = fn.crs4.cassandra(name="CassReader")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    return images

pl = get_dali_pipeline()
pl.build()

ddl = DALIGenericIterator(
   [pl], ['data'],
   reader_name='CassReader'
)

for epoch in range(4):
    print(f"Epoch {epoch}")
    for data in tqdm(ddl):
        x = data[0]['data']
    ddl.reset() # rewind data loader
