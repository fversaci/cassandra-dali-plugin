# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tqdm import trange, tqdm
import numpy as np
from time import sleep
import os
import torch
from torchvision import datasets, transforms


aug = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(n_mean, n_std, inplace=True),
    # transforms.GaussianBlur(5),
])

suff="jpg"
src_dir = os.path.join("/data/imagenette-cropped/", suff)
ds = datasets.ImageFolder(src_dir, aug)
num_classes = len(ds.classes) # 10
bs = 128

dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                 # num_workers=4, prefetch_factor=16,
                                 # pin_memory=True,
                                 )

for _ in trange(10):
    for x,y in tqdm(dl):
        # x = x.to("cuda:1")
        # y = y.to("cuda:1")
        ...

