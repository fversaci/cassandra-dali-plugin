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

folder = "../examples/lightning/logs_csv/imagenet_noio_4GPU/version_0/"
hyperparams_fname = os.path.join(folder, 'hparams.yaml')
log_fname = os.path.join(folder, 'metrics.csv')

# ### Load hyperparams

# +
with open(hyperparams_fname, 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
     
# Print the values as a dictionary
for hp in hyperparams:
    print(hp, hyperparams[hp])
# -

# ### Load data log and divide it in a train and a validation dataframe

df = pd.read_csv(log_fname)
df

# +
train_timestamp = df['train_batch_ts']
df_train = df[~np.isnan(train_timestamp)]

val_timestamp = df['val_batch_ts']
df_val = df[~np.isnan(val_timestamp)]
# -

df_train

df_val

# ### Train data

# +
x = df_train['train_batch_ts'] - df_train['train_batch_ts'].iloc[0]
y = df_train['train_im_sec_step'] 

plt.plot(x,y)
plt.xlabel('time (s)')
plt.ylabel('im/sec')
# -

df_train['train_im_sec_step'] 

df_train.iloc[150:200]

# ### Val Data

# +
x = df_val['val_batch_ts'] - df_train['train_batch_ts'].iloc[0]
y = df_val['val_im_sec_step'] 

plt.plot(x,y)
plt.xlabel('time (s)')
plt.ylabel('im/sec')
# -

df_val.iloc[250:300]


