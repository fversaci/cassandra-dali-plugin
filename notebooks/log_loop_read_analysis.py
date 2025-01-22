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

import numpy as np
from scipy.stats import hmean
import matplotlib.pyplot as plt
import os, glob
import itertools
import pandas as pd
import pickle


# #### Functions


# +
def load_loopread_csv(log_fn, test_name_dict):
    try:
        data_df = pd.read_csv(log_fn, dtype={'timestamp': np.float64})

        bs = data_df['bs'].iloc[0]
        n_epochs = data_df["epoch"].iloc[-1]

        data_df["batch_time_ms"] = data_df["batch_time"] * 1e3  # in milliseconds
        data_df["batch_bytes_MB"] = data_df["batch_bytes"] / 1e6  # batch size in MB
        data_df["data_rate_MBs"] = data_df["batch_bytes"] / data_df["batch_time"] / 1e6
        data_df["img_rate"] = bs / data_df["batch_time"]
        data_df["datetime"] = pd.to_datetime(data_df.timestamp, unit='s')
        
        data_grp_per_epoch = data_df.groupby("epoch")
    except:
        print(f"cannot read csv file {log_fn}")
        n_epochs = None
        bs = None
        data_df = None
        data_grp_per_epoch = None

    test_type = log_fn.split('AWS_MAIN_TEST_')[1].split('_')[0]
    
    basename = os.path.basename(log_fn)
    
    test_name, is_remote = test_name_dict[basename]
    
    if is_remote:
        test_name = test_type + "lat - " + test_name
    
    return n_epochs, bs, data_df, data_grp_per_epoch, test_name


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
        elif mean_type == 'integral':
            tmp_data_mean = tmp_data.sum() / data_group[1].iloc[1:-1]["batch_time"].sum()
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
                tmp_data_mean, label=(f"mean={tmp_data_mean:.1f} {um}"), c="r", lw=2
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
            tmp_data_std = None
        elif mean_type == 'integral':
            tmp_data_mean = tmp_data.sum() / data_group[1].iloc[1:-1]["batch_time"].sum()
            tmp_data_std = None
            
        avg_list.append(tmp_data_mean)
        std_list.append(tmp_data_std)

    return avg_list, std_list


def moving_average(a, n=10):
    a = a.values
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# -

# #### Log parent folder

parent_hi = "./AWS_logs/AWS_MAIN_TEST_HI_LAT/loopread/"
parent_low = "./AWS_logs/AWS_MAIN_TEST_LOW_LAT/"
parent_med = "./AWS_logs/AWS_MAIN_TEST_MED_LAT/"

# #### Read Dool as dataframe

# +
dool_fn = "./AWS_logs/AWS_MAIN_TEST_LOW_LAT/dool_loopread_2024-12-11.csv"
dool_df = pd.read_csv(dool_fn, header=5, index_col=False)

# Add unix epoch column
dool_df["timestamp"] = (
    pd.to_datetime(dool_df["time"], format="%b-%d %H:%M:%S")
    .apply(lambda dt: dt.replace(year=2024).timestamp())
    .astype(int)
)
dool_df.columns
# -

tmp = dool_df[dool_df.timestamp >= 1733881987.07862]
tmp = tmp[tmp.timestamp < 1733882012.9338305]
plt.plot(tmp['net/ens32:recv'])

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

for ix, i in enumerate(csv_file_list):
    print (ix, i)
# -

sel_csv_file_list = [csv_file_list[9], 
                     csv_file_list[5], csv_file_list[17], csv_file_list[15],
                     csv_file_list[1], csv_file_list[4], csv_file_list[3],
                     csv_file_list[8], csv_file_list[18], csv_file_list[16]]

test_name_dict = {
    "AWS_loop_read_TF_tfdata_tfr_BS_512.csv" : ("TFdata_tfr", 0), 
    "AWS_loop_read_TF_tfdataservice_tfr_BS_512.csv" : ("TFdataservice_tfr", 1),
    "AWS_loop_read_TF_tfdata_files_BS_512.csv" : ("TFdata files", 0),
    "AWS_loop_read_S3_Streaming_BS_512.csv" : ("S3_Streaming", 1),
    "AWS_loop_read_pytorch_files_BS_512.csv" : ("Pytorch Files", 0),
    "AWS_loop_read_cassandra_OOO_SLSTART_4_BS_512.csv" : ("Cassandra with OOO and SS=4", 1),
    "AWS_loop_read_scylla_OOO_SLSTART_4_BS_512.csv" : ("Scylla with OOO and SS=4", 1),
    "AWS_loop_read_scylla_OOO_SLSTART_0_BS_512.csv" : ("Scylla with OOO and SS=0", 1),
    "AWS_loop_read_scylla_no_OOO_SLSTART_0_BS_512.csv" : ("Scylla without OOO and SS=0", 1),
    "AWS_loop_read_DALI_tfrecord_BS_512.csv" : ("DALI tfr", 0),
    "AWS_loop_read_DALI_file_BS_512.csv" : ("DALI files", 0),
    "AWS_loop_read_S3_pytorch_files_BS_512.csv" : ("S3 Pytorch files", 1),
    "AWS_loop_read_S3_DALI_file_BS_512.csv" : ("S3 DALI files", 1),
}

# ### Tests on single file

fn = csv_file_list[5]

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(fn, test_name_dict)
print (test_name, n_epochs, bs)
data_df

print (f"Start: {data_df['datetime'].iloc[0]}")
print (f"Stop: {data_df['datetime'].iloc[-1]}")

data_df.describe()

# +
cesco_grp = data_df.groupby("epoch")

zz = np.empty(3)
for i in range(3):
    zz[i] = cesco_grp.get_group(i)['batch_time'].iloc[1:-1].sum()
    print (zz[i])

print(np.mean(zz))
print(np.std(zz))
# -

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

# +
tmp_list = []
for epoch, data_group in enumerate(data_grp_per_epoch):
    tmp = data_group[1]
    tmp_batch_tot_data = tmp['batch_bytes'].iloc[1:-1].sum() / 1e6
    tmp_batch_tot_time = tmp['batch_time'].iloc[1:-1].sum()
    tmp_list.append(tmp_batch_tot_data / tmp_batch_tot_time)

print (tmp_list)
print (np.mean(tmp_list))
# -

data_rate_avg = plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="data_rate_MBs",
    xlabel_str="data rate(MB/s)",
    y_label_str="occurencies",
    plot_type="hist",
    um="MB/s",
    mean_type='harmonic'
)
print(data_rate_avg)

plot_data(
    n_epochs,
    data_grp_per_epoch,
    data_field="data_rate_MBs",
    xlabel_str="batch_index",
    y_label_str="data rate(MB/s)",
    plot_type="line",
    um="MB/s",
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

# # Paper figures

# Data rates are assessed for each epoch, with an epoch representing a single "experiment" that measures the performance of transferring the entire dataset. The epoch data rate is determined by calculating the harmonic mean of the batch data rates. When applicable (i.e., for three or more epochs), the distribution of epoch data rates is analyzed to extract the mean and standard deviation, which are used to characterize the experiment.

barcolor_dict = {'HIlat': 'red',
             'MEDlat': 'green',
             'LOWlat': 'blue',
             '': 'k'
            }

# +
batch_time_dict = {}
data_size_dict = {}
data_rate_dict = {}
img_rate_dict = {}
df_dict = {}

data_rate_epoch_mean_std_dict = {}

ep = 0 # Epoch to take

x_bar = []
y_dr_bar = []
y_dr_yerr = []

x_tick_lab = []
x_color=[]

for i, fn in enumerate(sel_csv_file_list):
    print('-'*100)
    print(fn)
    n_epochs, bs, data_df, data_grp_per_epoch, name = load_loopread_csv(fn, test_name_dict)
    
    test_type = name.split('-')[0].rstrip()
    barcolor = "k"
    
    if test_type in list(barcolor_dict.keys()):
        barcolor = barcolor_dict[test_type]

    if bs == None:
        continue
    
    avg_batch_time_list, std_batch_time_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_time_ms"
    )
    avg_data_size_list, std_data_size_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_bytes_MB"
    )
    avg_data_rate_list, std_data_rate_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_bytes_MB", mean_type='integral'
    )
    avg_img_rate_list, std_img_rate_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="img_rate", mean_type='harmonic'
    )
    
    batch_time_dict[name] = (avg_batch_time_list, std_batch_time_list)
    data_size_dict[name] = (avg_data_size_list, std_data_size_list)
    data_rate_dict[name] = (avg_data_rate_list, std_data_rate_list)
    img_rate_dict[name] = (avg_img_rate_list, std_img_rate_list)
    df_dict[name] = data_df

    # Compute datarate average and std among epoch results if applicable 
    
    avg_list = np.array(data_rate_dict[name][0]) # it takes the list of epoch means
    avg_mean = np.round(np.mean(avg_list)) #MB/s
    avg_std = np.round(np.std(avg_list)) #MB/s
    data_rate_epoch_mean_std_dict[name]={'mean':avg_mean, 'std':avg_std}

    print(avg_list)
    
    # 

    # set x attributes
    x_bar.append(i)
    x_tick_lab.append(name)
    x_color.append(barcolor)

    y_dr_bar.append(avg_mean)
    y_dr_yerr.append(avg_std)

# -

pickle.dump(batch_time_dict, open("batch_time_dict.pickle", "wb"))

# #### Mean and std of data rate for table 3

data_rate_epoch_mean_std_dict

# ### Figure 3a

# +
x_bar_np = np.array(x_bar)
y_bar_np = np.array(y_dr_bar)
x_color_np = np.array(x_color)
x_tick_lab_np = ['DALI tfr', 'MosaicML SD', 'tf.data service', 'Cassandra-DALI']

x_bar_loc_index = np.array([0])
x_bar_hi_index = np.array([1, 4, 7])
x_bar_med_index = np.array([2, 5, 8])
x_bar_low_index = np.array([3, 6, 9])
x_ticks_indexes = [0,2,5,8]
x_offset = 0.2

norm_fact = np.max(y_bar_np)

_ = plt.bar(x_bar_np[x_bar_loc_index], y_bar_np[x_bar_loc_index] / norm_fact, color=x_color_np[x_bar_loc_index], label="Loc", zorder=3)
_ = plt.bar(x_bar_np[x_bar_hi_index] + x_offset, y_bar_np[x_bar_hi_index] / norm_fact, color=x_color_np[x_bar_hi_index], label="Hi", zorder=3)
_ = plt.bar(x_bar_np[x_bar_med_index], y_bar_np[x_bar_med_index] / norm_fact, color=x_color_np[x_bar_med_index], label="Med", zorder=3)
_ = plt.bar(x_bar_np[x_bar_low_index] - x_offset, y_bar_np[x_bar_low_index] / norm_fact, color=x_color_np[x_bar_low_index], label="Low", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0)

plt.ylabel("Normalized data rate")
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.savefig("figures/normalized_loopread.pdf", bbox_inches="tight")

# +
x_bar_np = np.array(x_bar)
y_bar_np = np.array(y_dr_bar)
x_color_np = np.array(x_color)
x_tick_lab_np = ['DALI tfr', 'MosaicML SD', 'tf.data service', 'Cassandra-DALI']

x_bar_loc_index = np.array([0])
x_bar_hi_index = np.array([1, 4, 7])
x_bar_med_index = np.array([2, 5, 8])
x_bar_low_index = np.array([3, 6, 9])
x_ticks_indexes = [0,2,5,8]
x_offset = 0.2

norm_fact = 1.0

_ = plt.bar(x_bar_np[x_bar_loc_index], y_bar_np[x_bar_loc_index] / norm_fact, color=x_color_np[x_bar_loc_index], label="Loc", zorder=3)
_ = plt.bar(x_bar_np[x_bar_hi_index] + x_offset, y_bar_np[x_bar_hi_index] / norm_fact, color=x_color_np[x_bar_hi_index], label="Hi", zorder=3)
_ = plt.bar(x_bar_np[x_bar_med_index], y_bar_np[x_bar_med_index] / norm_fact, color=x_color_np[x_bar_med_index], label="Med", zorder=3)
_ = plt.bar(x_bar_np[x_bar_low_index] - x_offset, y_bar_np[x_bar_low_index] / norm_fact, color=x_color_np[x_bar_low_index], label="Low", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0)

plt.ylabel("Data rate (GB/s)")
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.savefig("figures/loopread_rate.pdf", bbox_inches="tight")
# -

# ### Figure 4

# +
scylla_ooo_fn = csv_file_list[8]
scylla_no_ooo_fn = csv_file_list[12]
epoch = 1

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(scylla_ooo_fn, test_name_dict)
ooo_dr_series = data_grp_per_epoch.get_group(epoch)['batch_time_ms'].iloc[1:-1]
ooo_dr_series_smoothed = moving_average(ooo_dr_series, n=1)
print (ooo_dr_series.sum())
m_ooo = np.mean(ooo_dr_series)

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(scylla_no_ooo_fn, test_name_dict)
no_ooo_dr_series = data_grp_per_epoch.get_group(epoch)['batch_time_ms'].iloc[1:-1]
no_ooo_dr_series_smoothed = moving_average(no_ooo_dr_series, n=1)
print (no_ooo_dr_series.sum())
m_no_ooo = np.mean(no_ooo_dr_series)

f, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)
axs[0].plot(no_ooo_dr_series_smoothed, label=f"in-order", c='k', )
axs[0].axhline(m_no_ooo, ls='--', c='r', label= f"mean={m_no_ooo:.2f} ms")
axs[0].set_ylabel("milliseconds", fontsize=16)
axs[0].grid(True)
axs[0].legend(loc='upper center', fontsize=12)

axs[0].tick_params(axis='both', which='major', labelsize=11)

axs[1].set_xlabel("batch index", fontsize=16)
axs[1].plot(ooo_dr_series_smoothed, label=f"out-of-order", c='k')
axs[1].axhline(m_ooo, ls='--', c='r', label= f"mean={m_ooo:.2f} ms")
axs[1].set_ylabel("milliseconds", fontsize=16)
axs[1].grid(True)
axs[1].legend(loc='upper center', fontsize=12)

axs[1].tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()

plt.savefig("figures/batchtime_ooo_vs_noooo.pdf", bbox_inches="tight")
# -

plt.plot(no_ooo_dr_series_smoothed, label=f"no OOO", c='k', )
plt.xlim(115,1000)
for pippo in np.arange(119,1000,33):
    plt.axvline(pippo, c='r')
plt.grid()

# +
scylla_ooo_fn = csv_file_list[8]
scylla_no_ooo_fn = csv_file_list[12]
epoch = 2

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(scylla_ooo_fn, test_name_dict)
ooo_dr_series = data_grp_per_epoch.get_group(epoch)['data_rate_MBs'].iloc[1:-1]
ooo_dr_series_smoothed = moving_average(ooo_dr_series, n=1)
m_ooo = hmean(ooo_dr_series)

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(scylla_no_ooo_fn, test_name_dict)
no_ooo_dr_series = data_grp_per_epoch.get_group(epoch)['data_rate_MBs'].iloc[1:-1]
no_ooo_dr_series_smoothed = moving_average(no_ooo_dr_series, n=1)
m_no_ooo = hmean(no_ooo_dr_series)

f, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)
axs[0].plot(no_ooo_dr_series_smoothed, label="no OOO", c='k')
axs[0].axhline(m_no_ooo, ls='--', c='r', label= f"mean={m_no_ooo:.2f} MB/s")
axs[0].set_xlabel("batch index", fontsize=14)
axs[0].set_ylabel("MB/s", fontsize=14)
axs[0].grid(True)
axs[0].legend(loc='upper center')

axs[1].plot(ooo_dr_series_smoothed, label="OOO", c='k')
axs[1].axhline(m_ooo, ls='--', c='r', label= f"mean={m_ooo:.2f} MB/s")
axs[1].set_ylabel("MB/s", fontsize=14)
axs[1].grid(True)
axs[1].legend(loc='upper center')


plt.tight_layout()

plt.savefig("figures/throughput_ooo_vs_noooo.pdf", bbox_inches="tight")
# -

print(np.max(no_ooo_dr_series), np.min(no_ooo_dr_series))

# ### Figure 7

data_rate_dict['HIlat - Scylla with OOO and SS=4']

## Dool log to get server disk throughput 
fn = 'AWS_logs/server/stockholm/2024-12-10/general.csv'
df_log = pd.read_csv(fn, header=5, index_col=False)
df_log['time'] = '2024-' + df_log['time']
df_log = df_log.set_index('time')
# Convert index to datetime
df_log.index = pd.to_datetime(df_log.index)
#df_log.sort_index(inplace=True)
df_log.columns

# +
test_time_intervals_d = {'HI_scylla_ooo_start': '2024-12-10 16:03:31.660300032', 'HI_scylla_ooo_stop': '2024-12-10 16:05:58.601401856',
                        'HI_cassandra_ooo_start': '2024-12-10 16:22:40.195471104', 'HI_cassandra_ooo_stop': '2024-12-10 16:28:41.880232960'}

start_scylla = test_time_intervals_d['HI_scylla_ooo_start']
stop_scylla = test_time_intervals_d['HI_scylla_ooo_stop']

start_cass = test_time_intervals_d['HI_cassandra_ooo_start']
stop_cass = test_time_intervals_d['HI_cassandra_ooo_stop']

sel_epoch = 1
# -

# #### Scylla

# +
##### Disk

df_log_scylla = df_log.loc[start_scylla:stop_scylla].iloc[:]
df_log_scylla['disk_IO:read'] = df_log_scylla[["dsk/nvm11:read", "dsk/nvm31:read", "dsk/nvm31:read", "dsk/nvm41:read"]].sum(axis=1) / 8e9

scylla_disk_IO_mean = np.mean(df_log_scylla['disk_IO:read'].values)
print (scylla_disk_IO_mean)

#plt.plot(df_log_scylla[f"disk_IO:read"])

##### Loopread mean Data rate 
n_epochs, bs, data_df, data_grp_per_epoch, name = load_loopread_csv(csv_file_list[8], test_name_dict)
scylla_avg_data_rate_list, _ = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_bytes_MB", mean_type='integral',
    )
scylla_loopread_epoch_avg_MB = np.mean(scylla_avg_data_rate_list) / 1000. # in GB

print (scylla_avg_data_rate_list, scylla_loopread_epoch_avg_MB)

# -

# #### Cass ooo

# +
##### Disk
df_log_cass = df_log.loc[start_cass:stop_cass]
df_log_cass['disk_IO:read'] = df_log_cass[["dsk/nvm11:read", "dsk/nvm31:read", "dsk/nvm31:read", "dsk/nvm41:read"]].sum(axis=1) / 8e9

cass_disk_IO_mean = np.mean(df_log_cass['disk_IO:read'].values)

print (cass_disk_IO_mean)

#plt.plot(df_log_cass[f"disk_IO:read"])

##### Loopread mean Data rate
n_epochs, bs, data_df, data_grp_per_epoch, name = load_loopread_csv(csv_file_list[7], test_name_dict)
cass_avg_data_rate_list, _ = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="batch_bytes_MB", mean_type='integral',
    )
cass_loopread_epoch_avg_MB = np.mean(cass_avg_data_rate_list) / 1000. # in GB

print (cass_avg_data_rate_list, cass_loopread_epoch_avg_MB)

# +
plt.figure(figsize=(6,6))
x_bar_np = np.array([0,1,2,3])
y_bar_np = np.array([scylla_disk_IO_mean, scylla_loopread_epoch_avg_MB, 
                      cass_disk_IO_mean, cass_loopread_epoch_avg_MB])

x_color_np = np.array(['r','k', 'r', 'k'])
x_tick_lab_np = ['Scylla', 'Cassandra']

x_bar_disk_index = np.array([0, 2])
x_bar_reader_index = np.array([1, 3])
x_bar_low_index = np.array([3, 6, 9])
x_ticks_indexes = [0.5, 2.5]
x_offset = 0.1


#_ = plt.bar(x_bar_np[x_bar_disk_index], y_bar_np[x_bar_disk_index] / np.max(y_bar_np), color=x_color_np[x_bar_disk_index], label="Disk IO", zorder=3)
#_ = plt.bar(x_bar_np[x_bar_reader_index] + x_offset, y_bar_np[x_bar_reader_index] / np.max(y_bar_np), color=x_color_np[x_bar_reader_index], label="Reader throughput", zorder=3)

_ = plt.bar(x_bar_np[x_bar_disk_index] + x_offset, y_bar_np[x_bar_disk_index], color=x_color_np[x_bar_disk_index], label="Disk IO", zorder=3)
_ = plt.bar(x_bar_np[x_bar_reader_index] - x_offset, y_bar_np[x_bar_reader_index], color=x_color_np[x_bar_reader_index], label="Reader throughput", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0, fontsize=18)

plt.ylim(0,4.5)
plt.ylabel("Data rate (GB/s)",fontsize=18)
plt.legend(loc='upper right', fontsize=12)
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.yticks(fontsize=16)
plt.savefig("figures/scylla_vs_cassandra.pdf", bbox_inches="tight")
# -


