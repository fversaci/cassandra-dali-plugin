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
def load_loopread_csv(fn):
    try:
        data_df = pd.read_csv(fn)

        bs = data_df['bs'].iloc[0]
        n_epochs = data_df["epoch"].iloc[-1]

        data_df["batch_time_ms"] = data_df["batch_time"] * 1e3  # in milliseconds
        data_df["batch_bytes_MB"] = data_df["batch_bytes"] / 1e6  # batch size in MB
        data_df["data_rate_GBs"] = data_df["batch_bytes"] / data_df["batch_time"] / 1e9
        data_df["img_rate"] = bs / data_df["batch_time"]

        data_grp_per_epoch = data_df.groupby("epoch")
    except:
        print(f"cannot read csv file {fn}")
        n_epochs = None
        bs = None
        data_df = None
        data_grp_per_epoch = None

    return n_epochs, bs, data_df, data_grp_per_epoch


def plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="batch_time_ms",
    xlabel_str="batch time(ms)",
    y_label_str="occurencies",
    plot_type="hist",
    um="ms",
    mean_type='mean'
):
    n_cols = n_epochs + 1
    n_rows = 1
    subfig_size = 3
    f, axs = plt.subplots(
        1, n_cols, figsize=(n_cols * subfig_size, n_rows * subfig_size)
    )

    avg_list = []

    for epoch, data_group in enumerate(data_grp_per_epoch):
        if n_epochs == 0:
            ax = axs
        else:
            ax = axs[epoch]
        # Remove the first and last batch because they have overhead due
        # to prefetch and reshuffle
        tmp_data = data_group[1].iloc[1:-1][data_field]

        if mean_type == 'mean':
            tmp_data_mean = tmp_data.mean()
            avg_list.append(tmp_data_mean)
        elif mean_type == 'harmonic':
            tmp_data_mean = hmean(tmp_data)
            avg_list.append(tmp_data_mean)
            
        if plot_type == "hist":
            _ = ax.hist(tmp_data, bins=30)
            ax.axvline(
                tmp_data_mean, label=(f"mean={tmp_data_mean:.1f} {um}"), c="r", lw=1
            )
            ax.set_xlabel(xlabel_str)
            ax.set_ylabel(y_label_str)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        elif plot_type == "line":
            batch_index_xx = np.arange(0, len(tmp_data))
            ax.plot(batch_index_xx, tmp_data)
            ax.axhline(
                tmp_data_mean, label=(f"mean={tmp_data_mean:.2} {um}"), c="r", lw=2
            )
            ax.set_xlabel(xlabel_str)
            ax.set_ylabel(y_label_str)

        ax.set_title(f"Epoch {epoch}")
        ax.legend()

    plt.tight_layout()
    return avg_list


def get_per_epoch_avg_values(n_epochs, data_grp_per_epoch, data_field="batch_time_ms", mean_type='mean'):
    avg_list = []
    std_list = []

    for epoch, data_group in enumerate(data_grp_per_epoch):
        # Remove the first and last batch because they have overhead due
        # to prefetch and reshuffle
        tmp_data = data_group[1].iloc[1:-1][data_field]

        if mean_type == 'mean':
            tmp_data_mean = tmp_data.mean()
            tmp_data_std = tmp_data.std()
        elif mean_type == 'harmonic':
            tmp_data_mean = hmean(tmp_data)
            tmp_data_std = tmp_data.std()
            
        avg_list.append(tmp_data_mean)
        std_list.append(tmp_data_std)

    return avg_list, std_list


# -

# #### Log parent folder

parent_hi = "/home/giovanni/AWS_MAIN_TEST_HI_LAT/loopread/"
parent_low = "/home/giovanni/AWS_MAIN_TEST_LOW_LAT/"
parent_med = "/home/giovanni/AWS_MAIN_TEST_MED_LAT/"

# #### Read Dool as dataframe

# +
dool_fn = glob.glob(os.path.join(parent, "dool_loopread*"))[0]
dool_df = pd.read_csv(dool_fn, header=5, index_col=False)

# Add unix epoch column
dool_df["timestamp"] = (
    pd.to_datetime(dool_df["time"], format="%b-%d %H:%M:%S")
    .apply(lambda dt: dt.replace(year=2024).timestamp())
    .astype(int)
)
dool_df.columns
# -

dool_df

# #### Read loopread csv logs

# +
csv_folder_list = [
    os.path.join(parent_hi, "tensorflow/loopread/"),
    os.path.join(parent_low, "tensorflow/loopread/"),
    os.path.join(parent_med, "tensorflow/loopread/"),
    os.path.join(parent_hi, "torch/loopread/"),
    os.path.join(parent_low, "torch/loopread/"),
    os.path.join(parent_med, "torch/loopread/"),
]

csv_file_list = [glob.glob(i + "*.csv") for i in csv_folder_list]

## join list of lists
csv_file_list = list(itertools.chain.from_iterable(csv_file_list))
csv_file_list
# -



# ### Tests on single file

fn = csv_file_list[11]

n_epochs, bs, data_df, data_grp_per_epoch = load_loopread_csv(fn)
print (n_epochs, bs)
data_df

# ## Batch Time

avg_batch_time_list = plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="batch_time_ms",
    xlabel_str="batch time(ms)",
    y_label_str="occurencies",
    plot_type="hist",
    um="ms",
)
print(avg_batch_time_list)

plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="batch_time_ms",
    xlabel_str="batch_index",
    y_label_str="batch time(ms)",
    plot_type="line",
    um="ms",
)

# ## DATA

data_size_avg_list = plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="batch_bytes_MB",
    xlabel_str="data size(MB)",
    y_label_str="occurencies",
    plot_type="hist",
    um="MB",
)
print(data_size_avg_list)

plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="batch_bytes_MB",
    xlabel_str="batch_index",
    y_label_str="data size(MB)",
    plot_type="line",
    um="MB",
)

# ## Data Rate

data_rate_avg = plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="data_rate_GBs",
    xlabel_str="data rate(GB/s)",
    y_label_str="occurencies",
    plot_type="hist",
    um="GB/s",
    mean_type='harmonic'
)
print(data_rate_avg)

plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="data_rate_GBs",
    xlabel_str="batch_index",
    y_label_str="data rate(GB/s)",
    plot_type="line",
    um="GB/s",
    mean_type='harmonic'
)

# ## Image rate

img_rate_avg = plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="img_rate",
    xlabel_str="image rate(img/s)",
    y_label_str="occurencies",
    plot_type="hist",
    um="img/s",
    mean_type='harmonic'
)
print(img_rate_avg)

plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="img_rate",
    xlabel_str="batch_index",
    y_label_str="image rate(img/s)",
    plot_type="line",
    um="img/s",
    mean_type='harmonic'
)

# ### Tests on all files

# +
batch_time_dict = {}
data_size_dict = {}
data_rate_dict = {}
img_rate_dict = {}

ep = 0
x_bar = []
y_bt_bar = []
y_ds_bar = []
y_dr_bar = []
y_ir_bar = []
x_tick_lab = []

for i, fn in enumerate(csv_file_list):
    print(fn)
    n_epochs, bs, data_df, data_grp_per_epoch = load_loopread_csv(fn)
    if bs == None:
        continue
    name = "_".join(os.path.basename(fn).split("_")[3:-2])
    print(f"{name}")
    avg_batch_time_list, std_batch_time_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_time_ms"
    )
    avg_data_size_list, std_data_size_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_bytes_MB"
    )
    avg_data_rate_list, std_data_rate_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="data_rate_GBs", mean_type='harmonic'
    )
    avg_img_rate_list, std_img_rate_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="img_rate", mean_type='harmonic'
    )

    batch_time_dict[name] = (avg_batch_time_list, std_batch_time_list)
    data_size_dict[name] = (avg_data_size_list, std_data_size_list)
    data_rate_dict[name] = (avg_data_rate_list, std_data_rate_list)
    img_rate_dict[name] = (avg_img_rate_list, std_img_rate_list)

    x_bar.append(i)
    y_bt_bar.append(avg_batch_time_list[ep])
    y_ds_bar.append(avg_data_size_list[ep])
    y_dr_bar.append(avg_data_rate_list[ep])
    y_ir_bar.append(avg_img_rate_list[ep])
    x_tick_lab.append(name)
# -

_ = plt.bar(x_bar, y_bt_bar)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch time (ms)")

_ = plt.bar(x_bar, y_ds_bar)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch size (MB)")

_ = plt.bar(x_bar, y_dr_bar)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch data rate (GB/s)")

_ = plt.bar(x_bar, y_ir_bar)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch image rate (img/s)")






