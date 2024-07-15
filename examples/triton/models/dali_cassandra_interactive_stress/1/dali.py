# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

import sys

print("Python Version:", sys.version)
print("Python Executable Path:", sys.executable)

# cassandra reader
from cassandra_reader_interactive import get_cassandra_reader, read_uuids
from nvidia.dali.plugin.triton import autoserialize
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def

max_batch_size = 256


@autoserialize
@pipeline_def(batch_size=max_batch_size, num_threads=16)
def create_dali_pipeline(
    data_table="imagenette.data_train_orig",
    crop=224,
    size=256,
    dali_cpu=False,
    prefetch_buffers=4,
    io_threads=4,
    comm_threads=2,
    copy_threads=3,
    wait_threads=4,
):
    cass_reader = get_cassandra_reader(
        data_table=data_table,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
        ooo=False,
        slow_start=1,
    )
    images, labels = cass_reader
    return images[:][0]
