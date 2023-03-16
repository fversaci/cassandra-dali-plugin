# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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
