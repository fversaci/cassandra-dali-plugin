# Adapted to use cassandra-dali-plugin, from
# https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
# (Apache License, Version 2.0)

import sys

print("Python Version:", sys.version)
print("Python Executable Path:", sys.executable)

# cassandra reader
from cassandra_reader_interactive import get_cassandra_reader
from nvidia.dali.plugin.triton import autoserialize
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def

max_batch_size = 256


@autoserialize
@pipeline_def(batch_size=max_batch_size, num_threads=16)
def create_dali_pipeline(
    data_table="imagenette.data_train",
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
    dali_device = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == "mixed" else 0
    preallocate_height_hint = 6430 if decoder_device == "mixed" else 0
    images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
    images = fn.resize(
        images,
        device=dali_device,
        size=size,
        mode="not_smaller",
        interp_type=types.INTERP_TRIANGULAR,
    )
    mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    # labels = labels.gpu()
    # return (images, labels)
    return images

    ## for testing with single output and reduced bw:
    # return images[:][0][0][0]
