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

import os, sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

fnames = []

folder = "../examples/lightning/logs_csv/prova_1_GPU_NOIO_BS_1024/version_14"
hyperparams_fname = os.path.join(folder, "hparams.yaml")
log_fname = os.path.join(folder, "metrics.csv")

# ### Load hyperparams

# +
with open(hyperparams_fname, "r") as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

# Print the values as a dictionary
for hp in hyperparams:
    print(hp, hyperparams[hp])
# -

batch_size = hyperparams["batch_size"]
num_gpu = hyperparams["num_gpu"]

# ### Load data log and divide it in a train and a validation dataframe

# +
df = pd.read_csv(log_fname)

## Add time delta columns
df["train_batch_ts_diff"] = df["train_batch_ts"].diff()
df["val_batch_ts_diff"] = df["val_batch_ts"].diff()

df
# -

df

df["train_batch_ts_diff"]

df["val_batch_ts_diff"]

# ### Plots

# +
f, ax = plt.subplots(1, 2, figsize=(12, 4))

x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["train_batch_ts_diff"]) * num_gpu

y = df["train_batch_ts_diff"]

ax[0].plot(x, y)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("time (s)")
# ax[0].set_ylim(0.644,0.655)
# ax[0].set_xlim(0,20)

ax[0].legend()

x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["val_batch_ts_diff"]) * num_gpu
y = df["val_batch_ts_diff"]

ax[1].plot(x, y)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("time (s)")
# ax[1].set_xlim(40,47)


# +
f, ax = plt.subplots(1, 2, figsize=(12, 4))

x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["train_batch_ts_diff"]) * num_gpu

ax[0].plot(x, y)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("im/sec")
# ax[0].set_ylim(0.644,0.655)

ax[0].legend()

x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["val_batch_ts_diff"]) * num_gpu

ax[1].plot(x, y)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("im/sec")

# +
f, ax = plt.subplots(1, 2, figsize=(12, 4))

x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["train_batch_ts_diff"]) * num_gpu

y = df["train_batch_ts_diff"]

ax[0].plot(x, y)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("time (s)")
# ax[0].set_ylim(0.644,0.655)

ax[0].legend()

x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["val_batch_ts_diff"]) * num_gpu
y = df["val_batch_ts_diff"]

ax[0].plot(x, y)
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("time (s)")
# ax[1].set_xlim(40,47)


x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["train_batch_ts_diff"]) * num_gpu

ax[1].plot(x, y)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("im/sec")
# ax[1].set_ylim(0.644,0.655)

ax[0].legend()

x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
y = (batch_size / df["val_batch_ts_diff"]) * num_gpu

ax[1].plot(x, y)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("im/sec")
# -

# ### Comparison

# +
folders = [
    "../examples/lightning/logs_csv/KENTU400_1_GPU_NO_IO_BS_1024/version_1/",
    "../examples/lightning/logs_csv/KENTU400_1_GPU_TFR_BS_1024/version_3/",
    "../examples/lightning/logs_csv/KENTU400_1_GPU_SCYLLA_BS_1024/version_1/",
    "../examples/lightning/logs_csv/KENTU400_1_GPU_CASSANDRA_BS_1024/version_1/",
]

fnames = []

for folder in folders:
    hyperparams_fname = os.path.join(folder, "hparams.yaml")
    log_fname = os.path.join(folder, "metrics.csv")
    fnames.append((hyperparams_fname, log_fname))

fnames

# +


for hyperparams_fname, log_fname in fnames:
    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    print(hyperparams_fname, log_fname)

    batch_size = hyperparams["batch_size"]
    num_gpu = hyperparams["num_gpu"]

    test_name = "".join(log_fname.split("/")[-3].split("_")[1:])
    df = pd.read_csv(log_fname)

    df["train_batch_ts_diff"] = df["train_batch_ts"].diff()
    df["val_batch_ts_diff"] = df["val_batch_ts"].diff()

    x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
    y = (batch_size / df["train_batch_ts_diff"]) * num_gpu

    ax[0].plot(x, y, label=test_name)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("im/sec")

    ax[0].legend()

    x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
    y = (batch_size / df["val_batch_ts_diff"]) * num_gpu

    ax[1].plot(x, y, label=test_name)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("im/sec")

# -
