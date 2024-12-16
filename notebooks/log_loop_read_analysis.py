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


# #### Functions


# +
def get_conf_interval_hmean(data, interval=95):
    l_perc = (100.0-interval) / 2
    h_perc = 100 - ((100.0-interval) / 2)
    
    #print ("Percentile")
    
    data_l = np.percentile(data, l_perc)
    data_h = np.percentile(data, h_perc)
    
    #print (np.min(data), np.max(data), l_perc, h_perc, data_l, data_h)
    
    return data_l, data_h


def load_loopread_csv(log_fn, test_name_dict):
    try:
        data_df = pd.read_csv(log_fn, dtype={'timestamp': np.float64})

        bs = data_df['bs'].iloc[0]
        n_epochs = data_df["epoch"].iloc[-1]

        data_df["batch_time_ms"] = data_df["batch_time"] * 1e3  # in milliseconds
        data_df["batch_bytes_MB"] = data_df["batch_bytes"] / 1e6  # batch size in MB
        data_df["data_rate_GBs"] = data_df["batch_bytes"] / data_df["batch_time"] / 1e9
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
            tmp_data_std = get_conf_interval_hmean(tmp_data)
            
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

fn = csv_file_list[12]

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(fn, test_name_dict)
print (test_name, n_epochs, bs)
data_df

print (f"Start: {data_df['datetime'].iloc[0]}")
print (f"Stop: {data_df['datetime'].iloc[-1]}")

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

barcolor_dict = {'HIlat': 'r',
             'MEDlat': 'g',
             'LOWlat': 'b',
             '': 'k'
            }

# +
batch_time_dict = {}
data_size_dict = {}
data_rate_dict = {}
img_rate_dict = {}
df_dict = {}

ep = 0
x_bar = []
y_bt_bar = []
y_ds_bar = []
y_dr_bar = []
y_ir_bar = []
x_tick_lab = []
x_color=[]

for i, fn in enumerate(sel_csv_file_list):
    #print(fn)
    n_epochs, bs, data_df, data_grp_per_epoch, name = load_loopread_csv(fn, test_name_dict)
    
    test_type = name.split('-')[0].rstrip()
    barcolor = "k"
    
    if test_type in list(barcolor_dict.keys()):
        barcolor = barcolor_dict[test_type]
        
    print (test_type, list(barcolor_dict.keys()), barcolor)
    
    if bs == None:
        continue
        
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
    df_dict[name] = data_df

    
    x_bar.append(i)
    y_bt_bar.append(avg_batch_time_list[ep])
    y_ds_bar.append(avg_data_size_list[ep])
    y_dr_bar.append(avg_data_rate_list[ep])
    y_ir_bar.append(avg_img_rate_list[ep])
    x_tick_lab.append(name)
    x_color.append(barcolor)
# -

_ = plt.bar(x_bar, y_bt_bar / np.max(y_bt_bar), color=x_color)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch time (ms)")

_ = plt.bar(x_bar, y_ds_bar)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch size (MB)")

_ = plt.bar(x_bar, y_dr_bar / np.max(y_dr_bar), color=x_color)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch data rate (GB/s)")

_ = plt.bar(x_bar, y_ir_bar / np.max(y_ir_bar), color=x_color)
_ = plt.xticks(x_bar, x_tick_lab, rotation=90)
plt.ylabel("average batch image rate (img/s)")

# # Paper figures

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

ep = 0 # Epoch to take

x_bar = []
y_bt_bar = []
y_ds_bar = []
y_dr_bar = []
y_ir_bar = []
y_bt_yerr = []
y_ds_yerr = []
y_dr_yerr = []
y_ir_yerr = []
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
        n_epochs, data_grp_per_epoch, data_field="data_rate_GBs", mean_type='harmonic'
    )
    avg_img_rate_list, std_img_rate_list = get_per_epoch_avg_values(
        n_epochs, data_grp_per_epoch, data_field="img_rate", mean_type='harmonic'
    )
    
    batch_time_dict[name] = (avg_batch_time_list, std_batch_time_list)
    data_size_dict[name] = (avg_data_size_list, std_data_size_list)
    data_rate_dict[name] = (avg_data_rate_list, std_data_rate_list)
    img_rate_dict[name] = (avg_img_rate_list, std_img_rate_list)
    df_dict[name] = data_df

    
    x_bar.append(i)
    y_bt_bar.append(avg_batch_time_list[ep])
    y_ds_bar.append(avg_data_size_list[ep])
    y_dr_bar.append(avg_data_rate_list[ep])
    y_ir_bar.append(avg_img_rate_list[ep])
    
    y_bt_yerr.append(std_batch_time_list[ep])
    y_ds_yerr.append(std_data_size_list[ep])
    y_dr_yerr.append(std_data_rate_list[ep])
    y_ir_yerr.append(std_img_rate_list[ep])
    
    x_tick_lab.append(name)
    x_color.append(barcolor)
# -

y_bar_np



# ### Figure 3

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


_ = plt.bar(x_bar_np[x_bar_loc_index], y_bar_np[x_bar_loc_index] / np.max(y_bar_np), color=x_color_np[x_bar_loc_index], label="Loc", zorder=3)
_ = plt.bar(x_bar_np[x_bar_hi_index] + x_offset, y_bar_np[x_bar_hi_index] / np.max(y_bar_np), color=x_color_np[x_bar_hi_index], label="Hi", zorder=3)
_ = plt.bar(x_bar_np[x_bar_med_index], y_bar_np[x_bar_med_index] / np.max(y_bar_np), color=x_color_np[x_bar_med_index], label="Med", zorder=3)
_ = plt.bar(x_bar_np[x_bar_low_index] - x_offset, y_bar_np[x_bar_low_index] / np.max(y_bar_np), color=x_color_np[x_bar_low_index], label="Low", zorder=3)

_ = plt.xticks(x_ticks_indexes, x_tick_lab_np, rotation=0)

plt.ylabel("Normalized data rate")
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.8, zorder=0)

plt.savefig("figures/normalized_loopread.pdf", bbox_inches="tight")
# -

batch_time_dict

# ### Figure 5

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
axs[0].plot(ooo_dr_series_smoothed, label=f"OOO", c='k')
axs[0].axhline(m_ooo, ls='--', c='r', label= f"mean={m_ooo:.2f} ms")
axs[0].set_ylabel("milliseconds", fontsize=14)
axs[0].grid(True)
axs[0].legend(loc='upper center')

axs[1].plot(no_ooo_dr_series_smoothed, label=f"no OOO", c='k', )
axs[1].axhline(m_no_ooo, ls='--', c='r', label= f"mean={m_no_ooo:.2f} ms")
axs[1].set_xlabel("batch index", fontsize=14)
axs[1].set_ylabel("milliseconds", fontsize=14)
axs[1].grid(True)
axs[1].legend(loc='upper center')
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
ooo_dr_series = data_grp_per_epoch.get_group(epoch)['data_rate_GBs'].iloc[1:-1]
ooo_dr_series_smoothed = moving_average(ooo_dr_series, n=1)
m_ooo = hmean(ooo_dr_series)

n_epochs, bs, data_df, data_grp_per_epoch, test_name = load_loopread_csv(scylla_no_ooo_fn, test_name_dict)
no_ooo_dr_series = data_grp_per_epoch.get_group(epoch)['data_rate_GBs'].iloc[1:-1]
no_ooo_dr_series_smoothed = moving_average(no_ooo_dr_series, n=1)
m_no_ooo = hmean(no_ooo_dr_series)

f, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)
axs[0].plot(ooo_dr_series_smoothed, label="OOO", c='k')
axs[0].axhline(m_ooo, ls='--', c='r', label= f"mean={m_ooo:.2f} GB/s")
axs[0].set_ylabel("GB/s", fontsize=14)
axs[0].grid(True)
axs[0].legend(loc='upper center')

axs[1].plot(no_ooo_dr_series_smoothed, label="no OOO", c='k')
axs[1].axhline(m_no_ooo, ls='--', c='r', label= f"mean={m_no_ooo:.2f} GB/s")
axs[1].set_xlabel("batch index", fontsize=14)
axs[1].set_ylabel("GB/s", fontsize=14)
axs[1].grid(True)
axs[1].legend(loc='upper center')
plt.tight_layout()

plt.savefig("figures/throughput_ooo_vs_noooo.pdf", bbox_inches="tight")
# -

# ### Figure 8

data_rate_dict['HIlat - Scylla with OOO and SS=4']



fn = 'AWS_logs/server/stockholm/2024-12-10/general.csv'
df_log = pd.read_csv(fn, header=5, index_col=False)
df_log['time'] = '2024-' + df_log['time']
df_log = df_log.set_index('time')
# Convert index to datetime
df_log.index = pd.to_datetime(df_log.index)
#df_log.sort_index(inplace=True)
df_log.columns

len(df_log.columns)

# +
test_time_intervals_d = {'HI_scylla_ooo_start': '2024-12-10 16:03:31.660300032', 'HI_scylla_ooo_stop': '2024-12-10 16:05:58.601401856',
                        'HI_cassandra_ooo_start': '2024-12-10 16:22:40.195471104', 'HI_cassandra_ooo_stop': '2024-12-10 16:28:41.880232960'}

start_scylla = test_time_intervals_d['HI_scylla_ooo_start']
stop_scylla = test_time_intervals_d['HI_scylla_ooo_stop']

start_cass = test_time_intervals_d['HI_cassandra_ooo_start']
stop_cass = test_time_intervals_d['HI_cassandra_ooo_stop']
# -

# #### Scylla

# +
##### Disk

df_log_scylla = df_log.loc[start_scylla:stop_scylla].iloc[:]
df_log_scylla['disk_IO:read'] = df_log_scylla[["dsk/nvm11:read", "dsk/nvm31:read", "dsk/nvm31:read", "dsk/nvm41:read"]].sum(axis=1) / 8e9

scylla_disk_IO_mean = hmean(df_log_scylla['disk_IO:read'].values)
scylla_disk_IO_interval = get_conf_interval_hmean(df_log_scylla['disk_IO:read'].values)

print (scylla_disk_IO_mean, scylla_disk_IO_interval)

#plt.plot(df_log_scylla[f"disk_IO:read"])

##### Loopread Data rate 

# -

# #### Cass

# +
##### Disk
df_log_cass = df_log.loc[start_cass:stop_cass]
df_log_cass['disk_IO:read'] = df_log_cass[["dsk/nvm11:read", "dsk/nvm31:read", "dsk/nvm31:read", "dsk/nvm41:read"]].sum(axis=1) / 8e9

cass_disk_IO_mean = hmean(df_log_cass['disk_IO:read'].values)
cass_disk_IO_interval = get_conf_interval_hmean(df_log_cass['disk_IO:read'].values)

print (cass_disk_IO_mean, cass_disk_IO_interval)

#plt.plot(df_log_cass[f"disk_IO:read"])

##### Loopread Data rate
# -


