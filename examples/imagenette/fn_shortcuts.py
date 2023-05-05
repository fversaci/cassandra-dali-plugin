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

import nvidia.dali.fn as fn
import nvidia.dali.types as types


def fn_decode(images):
    return fn.decoders.image(
        images,
        device="mixed",
        output_type=types.RGB,
        # hybrid_huffman_threshold=100000,
        # memory_stats=True,
    )  # note: output is HWC (channels-last)


def fn_normalize(images):
    return fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )


def fn_image_random_crop(images):
    return fn.decoders.image_random_crop(
        images,
        device="mixed",
        output_type=types.RGB,
        hybrid_huffman_threshold=100000,
        random_aspect_ratio=[0.8, 1.25],
        random_area=[0.1, 1.0],
        # memory_stats=True,
    )  # note: output is HWC (channels-last)


def fn_resize(images):
    return fn.resize(
        images,
        resize_x=256,
        resize_y=256,
    )


def fn_crop_normalize(images):
    return fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
