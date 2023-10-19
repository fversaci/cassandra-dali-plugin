from nvidia.dali import fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.triton import autoserialize


@autoserialize
@pipeline_def(batch_size=512, num_threads=1, device_id=0)
def pipe():
    data = fn.external_source(device="cpu", name="DALI_INPUT_0")
    return data
