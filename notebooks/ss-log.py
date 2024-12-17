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

# +
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Set the maximum number of columns to be displayed
pd.set_option('display.max_columns', None)

## Time series comparison
from scipy.stats import pearsonr
#from dtaidistance import dtw


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# -

def moving_average(a, n=10):
    a = a.values
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# ### SS Logs

# +
# fn = 'ss2.csv.gz'
# df_log = pd.read_csv(fn, compression='gzip')

fn = 'AWS_logs/server/stockholm/2024-12-10/csv/ss-scylla.csv'
df_log = pd.read_csv(fn)

## Make some adjustment 
df_log = df_log.set_index('datetime')

df_log['ip_addr'] = [i.split(':')[0] for i in df_log.peer_addr] ## Only IP Address without port
df_log['port'] = [i.split(':')[1] for i in df_log.peer_addr] ## Only port Address without port

# Convert index to datetime
df_log.index = pd.to_datetime(df_log.index)
df_log.sort_index(inplace=True)

df_log.columns

## Columns of interest
col_sel = ['busy', 'bytes_acked', 'bytes_received',
       'bytes_retrans', 'bytes_sent', 'data_segs_in',
       'data_segs_out', 'delivered', 'delivery_rate', 
       'lastack', 'lastrcv', 'lastsnd', 'loc_addr', 
       'lost', 'notsent',
       'pacing_rate', 'peer_addr', 'recvq',
       'retrans', 'segs_in', 
       'segs_out', 'send', 'sendq', 'state',
       'unacked', 'ip_addr', 'port']
# -

# ## Create some time intervals of interest and select the working dataframes (Different training test)

# +
test_time_intervals_d = {'HI_scylla_ooo_start': '2024-12-10 16:03:31.660300032', 'HI_scylla_ooo_stop': '2024-12-10 16:05:58.601401856',
                        'HI_scylla_no_ooo_start': '2024-12-10 16:09:02.259324672', 'HI_scylla_no_ooo_stop': '2024-12-10 16:21:29.428230912'}
#test_time_intervals_d = {tag: datetime.datetime.fromisoformat(f'{date}T{test_time_intervals_d[tag]}+00:00') for tag in test_time_intervals_d} 

start = test_time_intervals_d['HI_scylla_ooo_start']
stop = test_time_intervals_d['HI_scylla_ooo_stop']

#start = test_time_intervals_d['HI_scylla_no_ooo_start']
#stop = test_time_intervals_d['HI_scylla_no_ooo_stop']

## Dataframe selection
df_log_sel = df_log.loc[start:stop]
# -

# ## Measures of interest of main connections (automatically detected)

# +
### Grouping by port
df_log_sel_grp = df_log_sel.groupby('port')
grps = df_log_sel_grp.groups
grps_key_l = list(grps.keys())

measures_l = ['busy', 'bytes_acked', 'bytes_received',
       'bytes_retrans', 'bytes_sent']

# Create a dictionary with: 
## key: measure of interest
## Value: A dataframe associated with the measure of interest, wherein each connection's contribution has its own column.
df_measure_d = {}

for measure in measures_l:
    df_measure = df_log_sel_grp.get_group(grps_key_l[0])[measure]
    df_measure.name = f'{df_measure.name}_{grps_key_l[0]}'

    for key in grps_key_l[1:]:
        g = df_log_sel_grp.get_group(key)[measure]
        g.name = f'{g.name}_{key}'

        df_measure = pd.merge(df_measure, g, left_index=True, right_index=True)
    
    df_measure_d[measure] = df_measure
    
## Add finite difference columns for each measure and communication port
col_list = []
for k in df_measure_d:
    print (k)
    df_tmp = df_measure_d[k]
    df_tmp_diff = df_tmp.diff() # Dataframe of finite difference
    df_tmp_diff.columns = [f'{c}_diff' for c in df_tmp.columns] # Rename columns of finite diff
     
    df_measure_d[k] = pd.concat([df_tmp, df_tmp_diff], axis=1) # Concat two dataframe horizontally

### MAIN CONNECTION DETECTION
### Select the main connections by using a threshold on maximum transmitted data
df_measure = df_measure_d['bytes_sent']
threshold = 0.1

data = df_measure.max() 
tmp = data / data.max() > threshold
selected_connection_port_l = [i.split('_')[2] for i in tmp[tmp == True].index.to_list()]
not_selected_connection_port_l = [i.split('_')[2] for i in tmp[tmp == False].index.to_list() if 'diff' not in i]

## Create a new df_dataframe_main_d with only the main connection
df_measure_mainconn_d = {}
for k in df_measure_d:
    scols = [f'{k}_{i}' for i in  selected_connection_port_l]
    df_measure_mainconn_d[k] = df_measure_d[k][scols]
    df_measure_mainconn_d[k].columns = [i.split('_')[-1] for i in df_measure_mainconn_d[k].columns]
    
df_measure_mainconn_diff_d = {}
for k in df_measure_d:
    scols = [f'{k}_{i}_diff' for i in  selected_connection_port_l]
    df_measure_mainconn_diff_d[k] = df_measure_d[k][scols]
    df_measure_mainconn_diff_d[k].columns = [i.split('_')[-2] for i in df_measure_mainconn_diff_d[k].columns]
    
    
## Create a new df_dataframe_main_d with only the residual connection
df_measure_residualconn_d = {}
for k in df_measure_d:
    scols = [f'{k}_{i}' for i in  not_selected_connection_port_l]
    df_measure_residualconn_d[k] = df_measure_d[k][scols]
    df_measure_residualconn_d[k].columns = [i.split('_')[-1] for i in df_measure_residualconn_d[k].columns]
    
df_measure_residualconn_diff_d = {}
for k in df_measure_d:
    scols = [f'{k}_{i}_diff' for i in  not_selected_connection_port_l]
    df_measure_residualconn_diff_d[k] = df_measure_d[k][scols]
    df_measure_residualconn_diff_d[k].columns = [i.split('_')[-2] for i in df_measure_residualconn_diff_d[k]]
    
### Adding some dataframe of derived measures
df_measure_mainconn_d['bytes_retrans_sent_ratio'] = df_measure_mainconn_d['bytes_retrans'] / df_measure_mainconn_d['bytes_sent'] 
df_measure_mainconn_diff_d['bytes_retrans_sent_ratio'] = df_measure_mainconn_diff_d['bytes_retrans'] / df_measure_mainconn_diff_d['bytes_sent'] 

measures_l.append('bytes_retrans_sent_ratio')


# -

def get_data(df_log_sel):

    ### Grouping by port
    df_log_sel_grp = df_log_sel.groupby('port')
    grps = df_log_sel_grp.groups
    grps_key_l = list(grps.keys())

    measures_l = ['busy', 'bytes_acked', 'bytes_received',
           'bytes_retrans', 'bytes_sent']

    # Create a dictionary with: 
    ## key: measure of interest
    ## Value: A dataframe associated with the measure of interest, wherein each connection's contribution has its own column.
    df_measure_d = {}

    for measure in measures_l:
        df_measure = df_log_sel_grp.get_group(grps_key_l[0])[measure]
        df_measure.name = f'{df_measure.name}_{grps_key_l[0]}'

        for key in grps_key_l[1:]:
            g = df_log_sel_grp.get_group(key)[measure]
            g.name = f'{g.name}_{key}'

            df_measure = pd.merge(df_measure, g, left_index=True, right_index=True)

        df_measure_d[measure] = df_measure

    ## Add finite difference columns for each measure and communication port
    col_list = []
    for k in df_measure_d:
        print (k)
        df_tmp = df_measure_d[k]
        df_tmp_diff = df_tmp.diff() # Dataframe of finite difference
        df_tmp_diff.columns = [f'{c}_diff' for c in df_tmp.columns] # Rename columns of finite diff

        df_measure_d[k] = pd.concat([df_tmp, df_tmp_diff], axis=1) # Concat two dataframe horizontally

    ### MAIN CONNECTION DETECTION
    ### Select the main connections by using a threshold on maximum transmitted data
    
    df_measure = df_measure_d['bytes_sent']
    threshold = 0.1

    data = df_measure.max() 
    tmp = data / data.max() > threshold
    selected_connection_port_l = [i.split('_')[2] for i in tmp[tmp == True].index.to_list()]
    not_selected_connection_port_l = [i.split('_')[2] for i in tmp[tmp == False].index.to_list() if 'diff' not in i]

    ## Create a new df_dataframe_main_d with only the main connection
    df_measure_mainconn_d = {}
    for k in df_measure_d:
        scols = [f'{k}_{i}' for i in  selected_connection_port_l]
        df_measure_mainconn_d[k] = df_measure_d[k][scols]
        df_measure_mainconn_d[k].columns = [i.split('_')[-1] for i in df_measure_mainconn_d[k].columns]

    df_measure_mainconn_diff_d = {}
    for k in df_measure_d:
        scols = [f'{k}_{i}_diff' for i in  selected_connection_port_l]
        df_measure_mainconn_diff_d[k] = df_measure_d[k][scols]
        df_measure_mainconn_diff_d[k].columns = [i.split('_')[-2] for i in df_measure_mainconn_diff_d[k].columns]


    ## Create a new df_dataframe_main_d with only the residual connection
    df_measure_residualconn_d = {}
    for k in df_measure_d:
        scols = [f'{k}_{i}' for i in  not_selected_connection_port_l]
        df_measure_residualconn_d[k] = df_measure_d[k][scols]
        df_measure_residualconn_d[k].columns = [i.split('_')[-1] for i in df_measure_residualconn_d[k].columns]

    df_measure_residualconn_diff_d = {}
    for k in df_measure_d:
        scols = [f'{k}_{i}_diff' for i in  not_selected_connection_port_l]
        df_measure_residualconn_diff_d[k] = df_measure_d[k][scols]
        df_measure_residualconn_diff_d[k].columns = [i.split('_')[-2] for i in df_measure_residualconn_diff_d[k]]

    ### Adding some dataframe of derived measures
    df_measure_mainconn_d['bytes_retrans_sent_ratio'] = df_measure_mainconn_d['bytes_retrans'] / df_measure_mainconn_d['bytes_sent'] 
    df_measure_mainconn_diff_d['bytes_retrans_sent_ratio'] = df_measure_mainconn_diff_d['bytes_retrans'] / df_measure_mainconn_diff_d['bytes_sent'] 

    measures_l.append('bytes_retrans_sent_ratio')
    
    return df_measure_mainconn_d, df_measure_mainconn_diff_d, df_measure_residualconn_d, df_measure_residualconn_diff_d, measures_l

# ### Plot of measure of interest for each main connection

smooth_factor = 2

# +
df_tmp = df_measure_residualconn_diff_d['bytes_sent']
n_plots = len(df_tmp.columns)
dim = int(np.ceil(np.sqrt(n_plots)))

_ = df_tmp.plot(subplots=True, layout=(dim,dim), figsize=(dim*2, dim*2), legend=False, sharex=True, sharey=True, title=f'{measure}')
# -

# ## Fig 5 and 6

# ### Fig 5

# +
test_time_intervals_d = {'HI_scylla_ooo_start': '2024-12-10 16:03:31.660300032', 'HI_scylla_ooo_stop': '2024-12-10 16:05:58.601401856',
                        'HI_scylla_no_ooo_start': '2024-12-10 16:09:02.259324672', 'HI_scylla_no_ooo_stop': '2024-12-10 16:21:29.428230912'}

start = test_time_intervals_d['HI_scylla_ooo_start']
stop = test_time_intervals_d['HI_scylla_ooo_stop']

## Dataframe selection
df_log_sel = df_log.loc[start:stop]

df_measure_mainconn_d, df_measure_mainconn_diff_d, df_measure_residualconn_d, df_measure_residualconn_diff_d, measures_l = get_data(df_log_sel)

df_tmp = df_measure_mainconn_diff_d['bytes_sent'][1:] / 1e6 # Skip the initial nan and convert to MB/s
n_plots = len(df_tmp.columns)

df_tmp_smoothed = df_tmp.apply(lambda x: moving_average(x, n=smooth_factor))

dim1 = int(np.ceil(np.sqrt(n_plots)))
dim2 = int(np.ceil(np.sqrt(n_plots)))

dim1 = 4
dim2 = 8

plt.rcParams.update({'font.size': 12}) 

_ = df_tmp_smoothed.plot(subplots=True, layout=(dim1,dim2), figsize=(dim2*2.5, dim1*2), lw=2, legend=False, sharex=True, sharey=True, xlabel='seconds', ylabel='MB/s')

plt.tight_layout()
plt.savefig("figures/scylla_OOO_sentbytes.pdf", bbox_inches="tight")
# -

# ### Fig 6

# +
start = test_time_intervals_d['HI_scylla_no_ooo_start']
stop = test_time_intervals_d['HI_scylla_no_ooo_stop']

## Dataframe selection
df_log_sel = df_log.loc[start:stop]

df_measure_mainconn_d, df_measure_mainconn_diff_d, df_measure_residualconn_d, df_measure_residualconn_diff_d, measures_l = get_data(df_log_sel)

df_tmp = df_measure_mainconn_diff_d['bytes_sent'][1:] / 1e6 # Skip the initial nan and convert to MB/s
n_plots = len(df_tmp.columns)

df_tmp_smoothed = df_tmp.apply(lambda x: moving_average(x, n=10))

dim1 = int(np.ceil(np.sqrt(n_plots)))
dim2 = int(np.ceil(np.sqrt(n_plots)))

dim1 = 4
dim2 = 8

plt.rcParams.update({'font.size': 12}) 

_ = df_tmp_smoothed.plot(subplots=True, layout=(dim1,dim2), figsize=(dim2*2.5, dim1*2), legend=False, sharex=True, sharey=True, xlabel='seconds', ylabel='MB/s')


plt.tight_layout()
plt.savefig("figures/scylla_NO_OOO_sentbytes.pdf", bbox_inches="tight")
# -

#
# ### Old stuff to look at the other measures

for measure in measures_l:
    df_tmp = df_measure_mainconn_d[measure]
    _ = df_tmp.plot(subplots=True, layout=(dim,dim), figsize=(dim*2, dim*2), legend=False, sharex=True, sharey=True, title=f'{measure}')

for measure in measures_l:
    df_tmp = df_measure_mainconn_diff_d[measure]
    _ = df_tmp.plot(subplots=True, layout=(dim,dim), figsize=(dim*2, dim*2), legend=False, sharex=True, sharey=True,title=f'{measure}_diff')
