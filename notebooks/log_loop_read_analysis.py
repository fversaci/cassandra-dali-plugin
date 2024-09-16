# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pickle
import numpy as np
import matplotlib.pyplot as plt

fn = "../examples/imagenette/log_loop_reader.pickle"

# +
bs, tbatch_np, byteio_np = pickle.load(open(fn, "rb"))

byteio_rate_np = byteio_np / tbatch_np / 1e9 # In GB/s
tbatch_np = tbatch_np * 1e3 # in ms

batch_size = bs
n_epochs = tbatch_np.shape[0]
# -

# ## Batch Time

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs/n_cols))
subfig_size = 3
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*subfig_size,n_rows*subfig_size))

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r,c]
    data = tbatch_np[epoch]
    data_mean = np.mean(data)
    
    _ = ax.hist(data, bins=30)
    
    ax.axvline(data_mean, label=(f"mean={data_mean:.2} ms"), c='r', lw=2)
    
    ax.set_xlabel("batch time(ms)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs/n_cols))
subfig_size = 3
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*subfig_size,n_rows*subfig_size))

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r,c]
    data = tbatch_np[epoch]
    data_mean = np.mean(data)
    
    ax.plot(data)
    
    ax.axhline(data_mean, label=(f"mean={data_mean:.2} ms"), c='r', lw=2)
    
    ax.set_ylim(0,15)
    ax.set_ylabel("batch time(ms)")
    ax.set_xlabel("batch index")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()
# -

# ## Data Rate

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs/n_cols))
subfig_size = 3
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*subfig_size,n_rows*subfig_size))

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r,c]
    data = byteio_rate_np[epoch]
    data_mean = np.mean(data)
    
    _ = ax.hist(data, bins=30)
    
    ax.axvline(data_mean, label=(f"mean={data_mean:.2} GB/s"), c='r', lw=2)
    
    ax.set_xlabel("data rate(GB/s)")
    #ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs/n_cols))
subfig_size = 3
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*subfig_size,n_rows*subfig_size))

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r,c]
    data = byteio_rate_np[epoch]
    data_mean = np.mean(data)
    
    ax.plot(data)
    
    ax.axhline(data_mean, label=(f"mean={data_mean:.2} GB/s"), c='r', lw=2)
    
    ax.set_ylim(0,20)
    ax.set_ylabel("data rate(GB/s)")
    ax.set_xlabel("batch index")
    #ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()
# -

# ## Image rate


