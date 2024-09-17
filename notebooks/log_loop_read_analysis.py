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
from scipy.stats import hmean
import matplotlib.pyplot as plt

fn = "../examples/imagenette/log_loop_reader.pickle"

# +
bs, tbatch_np, byteio_np = pickle.load(open(fn, "rb"))

last_time_batch = tbatch_np[:, -1]

# Removed last batch data. It takes into account overhead due to reshuffle
tbatch_np = tbatch_np[:, 0:-1]
byteio_np = byteio_np[:, 0:-1]

tbatch_ms_np = tbatch_np * 1e3  # in milliseconds
byteio_per_epoch_np = (
    np.sum(byteio_np, axis=1) / np.sum(tbatch_np, axis=1) / 1e9
)  # GB/s

batch_size = bs
n_epochs = tbatch_np.shape[0]
# -

# ## Batch Time

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = tbatch_ms_np[epoch][:-1]
    data_mean = np.mean(data)

    _ = ax.hist(data, bins=30)

    ax.axvline(data_mean, label=(f"mean={data_mean:.2} ms"), c="r", lw=2)

    ax.set_xlabel("batch time(ms)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = tbatch_ms_np[epoch]
    data_mean = np.mean(data)

    ax.plot(data)

    ax.axhline(data_mean, label=(f"mean={data_mean:.2} ms"), c="r", lw=2)

    ax.set_ylim(0, 60)
    ax.set_ylabel("batch time (ms)")
    ax.set_xlabel("batch index")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
per_epoch_means = []
for epoch in range(n_epochs):
    data = tbatch_ms_np[epoch]  # ms
    data_mean = np.mean(data)
    per_epoch_means.append(data_mean)

_ = plt.hist(per_epoch_means, bins=n_epochs)
plt.axvline(
    np.mean(per_epoch_means),
    label=(f"mean={np.mean(per_epoch_means):.2} ms"),
    c="r",
    lw=2,
)
plt.legend()
# -

# ## DATA

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = byteio_np[epoch] / 1e9
    data_mean = np.mean(data)

    _ = ax.hist(data, bins=30)

    ax.axvline(data_mean, label=(f"mean={data_mean:.2} GB"), c="r", lw=2)

    ax.set_xlabel("data size(GB)")
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = byteio_np[epoch] / 1e9
    data_mean = np.mean(data)

    ax.plot(data)

    ax.axhline(data_mean, label=(f"mean={data_mean:.2} GB"), c="r", lw=2)

    ax.set_ylim(0, 0.15)
    ax.set_ylabel("data size(GB)")
    ax.set_xlabel("batch index")
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()
# -

# ## Data Rate

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

byteio_rate_np = byteio_np / tbatch_np / 1e9  # In GB/s

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = byteio_rate_np[epoch]

    _ = ax.hist(data, bins=30)

    ax.axvline(
        byteio_per_epoch_np[epoch],
        label=(f"mean={byteio_per_epoch_np[epoch]:.2} GB/s"),
        c="r",
        lw=2,
    )

    ax.set_xlabel("data rate(GB/s)")
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = byteio_rate_np[epoch]

    ax.plot(data)

    ax.axhline(
        byteio_per_epoch_np[epoch],
        label=(f"mean={byteio_per_epoch_np[epoch]:.2} GB/s"),
        c="r",
        lw=2,
    )

    ax.set_ylim(0, 15)
    ax.set_ylabel("data rate(GB/s)")
    ax.set_xlabel("batch index")
    # ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()
# -

_ = plt.hist(byteio_per_epoch_np, bins=n_epochs)
plt.axvline(
    np.mean(byteio_per_epoch_np),
    label=(f"mean={np.mean(byteio_per_epoch_np):.2} GB/s"),
    c="r",
    lw=2,
)
plt.legend()

# ## Image rate

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

image_rate_np = bs / tbatch_np  # Image rate in im/s

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = image_rate_np[epoch]
    data_mean = hmean(data)  # harmonic mean to get the right speed average

    _ = ax.hist(data, bins=30)

    ax.axvline(data_mean, label=(f"hmean={data_mean:.0f} im/s"), c="r", lw=2)

    ax.set_xlabel("image rate(im/s)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
n_cols = 5
n_rows = int(np.ceil(n_epochs / n_cols))
subfig_size = 3
f, axs = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
)

for epoch in range(n_epochs):
    r = epoch // n_cols
    c = epoch % n_cols
    ax = axs[r, c]
    data = image_rate_np[epoch]
    data_mean = hmean(data)

    ax.plot(data)

    ax.axhline(data_mean, label=(f"hmean={data_mean:.0f} im/s"), c="r", lw=2)

    ax.set_ylim(0, 8e4)
    ax.set_ylabel("image rate(im/s)")
    ax.set_xlabel("batch index")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(f"Epoch {epoch}")
    ax.legend()
plt.tight_layout()

# +
per_epoch_means = []
for epoch in range(n_epochs):
    data = image_rate_np[epoch]
    data_mean = hmean(data)
    per_epoch_means.append(data_mean)

_ = plt.hist(per_epoch_means, bins=n_epochs)
plt.axvline(
    np.mean(per_epoch_means),
    label=(f"mean={np.mean(per_epoch_means):.2} im/s"),
    c="r",
    lw=2,
)
plt.legend()
# -
