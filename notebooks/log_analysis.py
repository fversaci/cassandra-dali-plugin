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
from scipy.stats import hmean

# ls './AWS_logs/AWS_MAIN_TEST_HI_LAT/train/torch/training/'

fnames = []

#folder = "../examples/lightning/logs_csv/ACCA400_1_GPU_NO_IO_BS_1024/version_0/"
folder = "./AWS_logs/AWS_MAIN_TEST_HI_LAT/train/torch/training/AWS_1_GPU_NO_IO_BS_512/version_0/"
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
    ## Remove all rows with nan epoch (end of epoch info)
    log_df  = log_df [~np.isnan(log_df .epoch)]
    log_df["train_batch_ts_diff"] = log_df["train_batch_ts"].diff()
    log_df["val_batch_ts_diff"] = log_df["val_batch_ts"].diff()
    log_df["train_im_sec"] =  (batch_size / log_df["train_batch_ts_diff"]) * num_gpu
    log_df["val_im_sec"] =  (batch_size / log_df["val_batch_ts_diff"]) * num_gpu
    
    
    log_df["epoch"] = log_df["epoch"].astype(np.int32) 
    return test_name, num_gpu, batch_size, hyperparams, log_df


def read_tf_results(log_fname, batch_size=512, num_gpu=8):
    
    if "NO_IO" in log_fname:
        test_name = "_".join(log_fname.split("/")[-1].split("_")[1:-2])
        num_gpu = int(log_fname.split("/")[-1].split("_")[1])
    else:
        test_type = log_fname.split('AWS_MAIN_TEST_')[1].split('_')[0]
        test_name = test_type + "_" + "_".join(log_fname.split("/")[-1].split("_")[2:-3])
    
    log_df = pd.read_csv(log_fname)
    ## Remove all rows with nan epoch (end of epoch info)
    log_df  = log_df [~np.isnan(log_df .epoch)]
    log_df["train_batch_ts_diff"] = log_df["train_batch_ts"].diff()
    log_df["val_batch_ts_diff"] = log_df["val_batch_ts"].diff()
    log_df["train_im_sec"] =  (batch_size / log_df["train_batch_ts_diff"]) * num_gpu
    log_df["val_im_sec"] =  (batch_size / log_df["val_batch_ts_diff"]) * num_gpu
    log_df["epoch"] = log_df["epoch"].astype(np.int32)
    return test_name, log_df


# +
parent = './AWS_logs/AWS_MAIN_TEST_LOW_LAT/torch/training/'
folders = glob.glob(parent + '/*/version_0/')

parent = './AWS_logs/AWS_MAIN_TEST_MED_LAT/torch/training/'
folders += glob.glob(parent + '/*/version_0/')

parent = './AWS_logs/AWS_MAIN_TEST_HI_LAT/train/torch/training/'
folders += glob.glob(parent + '/*/version_0/')

torch_folders = sorted(folders)
# -

tf_files = ['./AWS_logs/AWS_MAIN_TEST_LOW_LAT/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512', 
            './AWS_logs/AWS_MAIN_TEST_MED_LAT/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512',
            './AWS_logs/AWS_MAIN_TEST_HI_LAT/train/tensorflow/train/AWS_Train_TF_tfdataservice_tfr_BS_512', 
            './AWS_logs/AWS_MAIN_TEST_HI_LAT/train/tensorflow/train/AWS_1_GPU_TF_NO_IO_BS_512',]

# +
data_dict = {}

## Get torch data, num gpu and batch size
for folder in tqdm(torch_folders):
    test_name, num_gpu, batch_size, hyperparams, log_df = read_torch_lightning_results(folder)
    df_group = log_df.groupby("epoch")
    data_dict[test_name] = (log_df, batch_size, num_gpu, df_group)

## Get TF data
for fname in tqdm(tf_files):
    test_name, log_df = read_tf_results(fname)
    df_group = log_df.groupby("epoch")
    print (test_name)
    data_dict[test_name] = (log_df, 512, 8, df_group)
# -

data_dict.keys()

tmp_df = data_dict[list(data_dict.keys())[0]][0]
tmp_df

# +
n_subplots = len(data_dict.keys())
subplot_size = 4


for i, test_name in enumerate(data_dict):    
    df, batch_size, num_gpu, df_group = data_dict[test_name]
    
    n_epochs = len(df_group.groups)
    
    f, axs = plt.subplots(1, n_epochs, figsize=(subplot_size*n_epochs*2, subplot_size))
    
    for grp in df_group.groups:
        
        df = df_group.get_group(grp)
        if n_epochs == 1:
            ax = axs
        else:
            ax = axs[grp]

        x = df["train_batch_ts"] - df["train_batch_ts"].iloc[0]
        y = df["train_im_sec"]
        m = hmean(y[~np.isnan(y)])

        ax.plot(x, y, label=test_name)
        ax.axhline(m, c='r', ls='--', label=f"mean: {m:.1f}im/sec")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("im/sec")

        ax.set_ylim(0, 12000)
        ax.legend()

plt.tight_layout()

f.savefig("training_timeseries.png", dpi=300)
# -




# ### Fig 4

data_dict.keys()

barcolor_dict = {'HIlat': 'red',
             'MEDlat': 'green',
             'LOWlat': 'blue',
             '': 'k'
            }

test_name_list = [('8_GPU_NO_IO', 'NO_IO', 'k'),
                  ('HI_STREAMING', 'MosaicML SD', 'red'), ('MED_STREAMING', 'MosaicML SD', 'green'), ('LOW_STREAMING', 'MosaicML SD', 'blue'),
                  ('HI_SCYLLA', 'Cassandra-DALI', 'red'), ('MED_SCYLLA', 'Cassandra-DALI', 'green'), ('LOW_SCYLLA', 'Cassandra-DALI', 'blue')] 

# +
batch_time_dict = {}
data_size_dict = {}
data_rate_dict = {}
img_rate_dict = {}
df_dict = {}

ep = 1 # Epoch to take

x_bar = []
y_bar = []
x_tick_lab = []
x_color=[]

for i, (test_name, label, color) in enumerate(test_name_list):    
    df, batch_size, num_gpu, df_group = data_dict[test_name]

    #df = df[df["epoch"] >= 1] # decomment to exclude epoch 0
    #df = df_group.get_group(ep)
    
    y = df["train_im_sec"]
    m = hmean(y[~np.isnan(y)])

    x_bar.append(i)
    x_tick_lab.append(label)
    y_bar.append(m)
    x_color.append(color)
# -

y_bar_np

# +
x_bar_np = np.array(x_bar)
y_bar_np = np.array(y_bar)
x_color_np = np.array(x_color)
x_tick_lab_np = ['No IO', 'MosaicML SD', 'Cassandra-DALI']

x_bar_loc_index = np.array([0])
x_bar_hi_index = np.array([1, 4])
x_bar_med_index = np.array([2, 5])
x_bar_low_index = np.array([3, 6])
x_ticks_indexes = [0,2,5]
x_offset = 0.2

norm_factor = np.max(y_bar_np)

_ = plt.bar(x_bar_np[x_bar_loc_index], y_bar_np[x_bar_loc_index] / norm_factor, color=x_color_np[x_bar_loc_index], label="Loc", zorder=3)
_ = plt.bar(x_bar_np[x_bar_hi_index] + x_offset, y_bar_np[x_bar_hi_index] / norm_factor, color=x_color_np[x_bar_hi_index], label="Hi", zorder=3)
_ = plt.bar(x_bar_np[x_bar_med_index], y_bar_np[x_bar_med_index] / norm_factor, color=x_color_np[x_bar_med_index], label="Med", zorder=3)
_ = plt.bar(x_bar_np[x_bar_low_index] - x_offset, y_bar_np[x_bar_low_index] / norm_factor, color=x_color_np[x_bar_low_index], label="Low", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0)

plt.ylabel("Normalized image rate")
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.savefig("figures/normalized_train_rate.pdf", bbox_inches="tight")

# +
x_bar_np = np.array(x_bar)
y_bar_np = np.array(y_bar)
x_color_np = np.array(x_color)
x_tick_lab_np = ['No IO', 'MosaicML SD', 'Cassandra-DALI']

x_bar_loc_index = np.array([0])
x_bar_hi_index = np.array([1, 4])
x_bar_med_index = np.array([2, 5])
x_bar_low_index = np.array([3, 6])
x_ticks_indexes = [0,2,5]
x_offset = 0.2

norm_factor = 1.0

_ = plt.bar(x_bar_np[x_bar_loc_index], y_bar_np[x_bar_loc_index] / norm_factor, color=x_color_np[x_bar_loc_index], label="Loc", zorder=3)
_ = plt.bar(x_bar_np[x_bar_hi_index] + x_offset, y_bar_np[x_bar_hi_index] / norm_factor, color=x_color_np[x_bar_hi_index], label="Hi", zorder=3)
_ = plt.bar(x_bar_np[x_bar_med_index], y_bar_np[x_bar_med_index] / norm_factor, color=x_color_np[x_bar_med_index], label="Med", zorder=3)
_ = plt.bar(x_bar_np[x_bar_low_index] - x_offset, y_bar_np[x_bar_low_index] / norm_factor, color=x_color_np[x_bar_low_index], label="Low", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0)

plt.ylabel("image rate (img/sec)")
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.savefig("figures/train_rate.pdf", bbox_inches="tight")
# -


