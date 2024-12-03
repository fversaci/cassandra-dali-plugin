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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from scipy.stats import hmean
import matplotlib.pyplot as plt
import os, glob
import itertools
import pandas as pd


# #### Functions

# +
def plot_hist(n_epochs, data_grp_per_epoch, data_field='batch_time_ms', xlabel_str="batch time(ms)", y_label_str="occurencies"):
    n_cols = n_epochs+1
    n_rows = 1
    subfig_size = 3
    f, axs = plt.subplots(
        1, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
    )
    
    for epoch, data_group in enumerate(data_grp_per_epoch):
        ax = axs[epoch]
        # Remove the first and last batch because they have overhead due
        # to prefetch and reshuffle
        tmp_data = data_group[1].iloc[1:-1][data_field]
        
        tbatch_ms_mean = tmp_data.mean()
    
        _ = ax.hist(tmp_data, bins=30)
    
        ax.axvline(tbatch_ms_mean, label=(f"mean={tbatch_ms_mean:.1f} ms"), c="r", lw=1)
        
        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(y_label_str)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(f"Epoch {epoch}")
        ax.legend()
        
    plt.tight_layout()

def plot_line(n_epochs, data_grp_per_epoch, data_field='batch_time_ms', xlabel_str="batch_index", y_label_str="batch time(ms)"):
    n_cols = n_epochs+1
    n_rows = 1
    subfig_size = 3
    f, axs = plt.subplots(
        1, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
    )
    
    for epoch, data_group in enumerate(data_grp_per_epoch):
        ax = axs[epoch]
        # Remove the first and last batch because they have overhead due
        # to prefetch and reshuffle
        tmp_data = data_group[1].iloc[1:-1][data_field]
        batch_index_xx = np.arange(0, len(tmp_data))
        
        tbatch_ms_mean = tmp_data.mean()
    
        ax.plot(batch_index_xx, tmp_data)

        ax.axhline(tbatch_ms_mean, label=(f"mean={tbatch_ms_mean:.2} ms"), c="r", lw=2)

        ax.set_xlabel(xlabel_str)
        ax.set_ylabel(y_label_str)
        #ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(f"Epoch {epoch}")
        ax.legend()
        
    plt.tight_layout()


# -

# #### Log parent folder

parent = "./logs/TEST/"

# #### Read Dool as dataframe

# +
dool_fn = glob.glob(os.path.join(parent, "dool_loopread*"))[0]
dool_df = pd.read_csv(dool_fn, header=5, index_col=False)

# Add unix epoch column
dool_df['timestamp'] = pd.to_datetime(dool_df['time'],format="%b-%d %H:%M:%S").apply(lambda dt: dt.replace(year=2024).timestamp()).astype(int)
dool_df.columns
# -

dool_df

# #### Read loopread csv logs

# +

csv_folder_list = [os.path.join(parent, "tensorflow/loopread/"), os.path.join(parent, "torch/loopread/")]

csv_file_list = [glob.glob(i + "*.csv") for i in csv_folder_list]

## join list of lists
csv_file_list = list(itertools.chain.from_iterable(csv_file_list))
csv_file_list
# -

fn = csv_file_list[1]

# +
data_df = pd.read_csv(fn)

bs = 1024
n_epochs = data_df['epoch'].iloc[-1]

data_df['batch_time_ms'] = data_df['batch_time'] * 1e3  # in milliseconds
data_df['data_rate_GBs'] = data_df['batch_bytes'] / data_df['batch_time'] / 1e9
data_df['img_rate'] = bs / data_df['batch_time']

data_grp_per_epoch = data_df.groupby('epoch') 

data_df
# -

# ## Batch Time

plot_hist(n_epochs, data_grp_per_epoch, data_field='batch_time_ms', xlabel_str="batch time(ms)", y_label_str="occurencies")

plot_line(n_epochs, data_grp_per_epoch, data_field='batch_time_ms', xlabel_str="batch_index", y_label_str="batch time(ms)")

# ## DATA

plot_hist(n_epochs, data_grp_per_epoch, data_field='batch_bytes', xlabel_str="data size(GB)", y_label_str="occurencies")

plot_line(n_epochs, data_grp_per_epoch, data_field='batch_bytes', xlabel_str="batch_index", y_label_str="data size(GB)")

# ## Data Rate

plot_hist(n_epochs, data_grp_per_epoch, data_field='data_rate_GBs', xlabel_str="data rate(GB/s)", y_label_str="occurencies")

plot_line(n_epochs, data_grp_per_epoch, data_field='data_rate_GBs', xlabel_str="batch_index", y_label_str="data rate(GB/s)")

# ## Image rate

plot_hist(n_epochs, data_grp_per_epoch, data_field='img_rate', xlabel_str="image rate(img/s)", y_label_str="occurencies")

plot_line(n_epochs, data_grp_per_epoch, data_field='img_rate', xlabel_str="batch_index", y_label_str="image rate(img/s)")


