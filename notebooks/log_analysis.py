# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: python3_stats-ML
#     language: python
#     name: python3_stats-ml
# ---

import os, sys, glob
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# ls '/home/giovanni/AWS_MAIN_TEST_HI_LAT/train/torch/training/'

fnames = []

#folder = "../examples/lightning/logs_csv/ACCA400_1_GPU_NO_IO_BS_1024/version_0/"
folder = "/home/giovanni/AWS_MAIN_TEST_HI_LAT/train/torch/training/AWS_1_GPU_NO_IO_BS_512/version_0/"
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
def read_torch_lightning_results(folder):
    hyperparams_fname = os.path.join(folder, "hparams.yaml")
    log_fname = os.path.join(folder, "metrics.csv")
    
    with open(hyperparams_fname, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    batch_size = hyperparams["batch_size"]
    num_gpu = hyperparams["num_gpu"]
    
    test_type = log_fname.split('AWS_MAIN_TEST_')[1].split('_')[0]
    if 'NO_IO' in log_fname:
        test_name = "_".join(log_fname.split("/")[-3].split("_")[1:-2])
    elif 'FILE' in log_fname or 'TFR' in log_fname:
        test_name = "_".join(log_fname.split("/")[-3].split("_")[3:-2])
    else:
        test_name = test_type + "_" + "_".join(log_fname.split("/")[-3].split("_")[3:-2])
    
    log_df = pd.read_csv(log_fname)
    log_df["train_batch_ts_diff"] = log_df["train_batch_ts"].diff()
    log_df["val_batch_ts_diff"] = log_df["val_batch_ts"].diff()
    log_df["train_im_sec"] =  (batch_size / log_df["train_batch_ts_diff"]) * num_gpu
    log_df["val_im_sec"] =  (batch_size / log_df["val_batch_ts_diff"]) * num_gpu
    
    return test_name, num_gpu, batch_size, hyperparams, log_df


def read_tf_results(log_fname, batch_size=512, num_gpu=8):
    test_type = log_fname.split('AWS_MAIN_TEST_')[1].split('_')[0]
    test_name = test_type + "_" + "_".join(log_fname.split("/")[-1].split("_")[2:-3])
    log_df = pd.read_csv(log_fname)
    log_df["train_batch_ts_diff"] = log_df["train_batch_ts"].diff()
    log_df["val_batch_ts_diff"] = log_df["val_batch_ts"].diff()
    log_df["train_im_sec"] =  (batch_size / log_df["train_batch_ts_diff"]) * num_gpu
    log_df["val_im_sec"] =  (batch_size / log_df["val_batch_ts_diff"]) * num_gpu
    
    return test_name, log_df


# +
parent = '/home/giovanni/AWS_MAIN_TEST_LOW_LAT/torch/training/'
folders = glob.glob(parent + '/*/version_0/')

parent = '/home/giovanni/AWS_MAIN_TEST_MED_LAT/torch/training/'
folders += glob.glob(parent + '/*/version_0/')

parent = '/home/giovanni/AWS_MAIN_TEST_HI_LAT/train/torch/training/'
folders += glob.glob(parent + '/*/version_0/')

torch_folders = sorted(folders)
# -

tf_files = ['/home/giovanni/AWS_MAIN_TEST_LOW_LAT/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512', 
            '/home/giovanni/AWS_MAIN_TEST_MED_LAT/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512',
            '/home/giovanni/AWS_MAIN_TEST_HI_LAT/train/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512']

# +
data_dict = {}

## Get torch data, num gpu and batch size
for folder in tqdm(torch_folders):
    test_name, num_gpu, batch_size, hyperparams, log_df = read_torch_lightning_results(folder)
    data_dict[test_name] = (log_df, batch_size, num_gpu)

## Get TF data
for fname in tqdm(tf_files):
    test_name, log_df = read_tf_results(fname)
    print (test_name)
    data_dict[test_name] = (log_df, 512, 8)
# -

data_dict.keys()

# +
n_subplots = len(data_dict.keys())
subplot_size = 4
f, axs = plt.subplots(n_subplots, 1, figsize=(subplot_size*2, n_subplots*subplot_size))

for i, test_name in enumerate(data_dict):    
    df, batch_size, num_gpu = data_dict[test_name]

    x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
    y = df["train_im_sec"]
    m = np.nanmean(y)
    
    ax = axs[i] 
    ax.plot(x, y, label=test_name)
    ax.axhline(m, c='r', ls='--', label=f"mean: {m:.1f}im/sec")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("im/sec")

    ax.set_ylim(0, 12000)
    ax.legend()

plt.tight_layout()

f.savefig("training_timeseries.png", dpi=300)
# -


for i, test_name in enumerate(data_dict):    
    df, batch_size, num_gpu = data_dict[test_name]

    x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
    y = df["train_im_sec"]
    m = np.nanmean(y)
    s = np.nanstd(y)


