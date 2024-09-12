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

folder = "../examples/lightning/logs_csv/prova_tfr/version_2"
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

df = pd.read_csv(log_fname)
df

df["train_batch_ts_diff"] = df["train_batch_ts"].diff()
df["val_batch_ts_diff"] = df["val_batch_ts"].diff()

# +
train_timestamp = df["train_batch_ts"]
df_train = df[~np.isnan(train_timestamp)]
# df_train['train_batch_ts_diff'] = df_train['train_batch_ts'].diff()

val_timestamp = df["val_batch_ts"]
df_val = df[~np.isnan(val_timestamp)]
# df_val['val_batch_ts_diff'] = df_val['val_batch_ts'].diff()
# -

df_train

df_val

# ### Train data

# +
x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
y = df["train_im_sec_step"]
y2 = (batch_size / df["train_batch_ts_diff"]) * num_gpu

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("time (s)")
plt.ylabel("im/sec")
# -

plt.plot(df_train.index)

# +
x = df_train["train_batch_ts"] - df_train["train_batch_ts"].iloc[0]
y = df_train["train_im_sec_step"]
y2 = (batch_size / df_train["train_batch_ts_diff"]) * num_gpu

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("time (s)")
plt.ylabel("im/sec")
# -

# ### Val Data

# +
x = df["val_batch_ts"] - df["train_batch_ts"].iloc[0]
y = df["val_im_sec_step"]
y2 = (batch_size / df["val_batch_ts_diff"]) * num_gpu

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("time (s)")
plt.ylabel("im/sec")
# -

plt.plot(df_val.index)
plt.plot(df_train.index)

# +
x = df_val["val_batch_ts"] - df_train["train_batch_ts"].iloc[0]
y = df_val["val_im_sec_step"]
y2 = (batch_size / df_val["val_batch_ts_diff"]) * num_gpu

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("time (s)")
plt.ylabel("im/sec")
