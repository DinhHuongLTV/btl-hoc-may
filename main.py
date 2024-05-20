import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import re
#..
pip install ucimlrepo
#..
from ucimlrepo import fetch_ucirepo

# fetch dataset
air_quality = fetch_ucirepo(id=360) # <---- is in Json format

# data (as pandas dataframes)
X = air_quality.data.features
y = air_quality.data.targets

# metadata
# print(air_quality.metadata)

# variable information
# print(air_quality.variables)

#..
%mkdir UntitledFolder

#..
df = air_quality.data.features
df
#..
new_col = []
for col in df.columns:
  new_col.append(re.sub("[\(\[].*?[\)\]]", "", col))
cols = df.columns
for j in cols:
    for i in range(0,len(df)):
       if df[j][i] == -200:
           df[j][i] = df[j][i-1]
  #df.rename(columns = {'col':'new_col'}, inplace = True)
print(new_col)
df.columns = new_col
df

#..
data_col = new_col[2:]
plt.figure(figsize=(len(data_col),len(data_col)/2))
for i, col in enumerate(data_col):
  plt.plot(df.head(150).index, df.head(150)[col], label = str(col))
plt.legend()

#..
group_1 = ['PT08.S1', 'PT08.S2', 'PT08.S3', 'PT08.S4', 'PT08.S5']
group_2 = ['CO', 'NOx',  'NO2', 'NMHC']
group_3 = ['T', 'RH', 'AH']
# Still C6H6 left
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 8))
for col in group_1:
  axes[0].plot(df.head(150).index, df.head(150)[col], label = str(col))
axes[0].legend()
for col in group_2:
  axes[1].plot(df.head(150).index, df.head(150)[col], label = str(col))
axes[1].legend()
for col in group_3:
  axes[2].plot(df.head(150).index, df.head(150)[col], label = str(col))
axes[2].legend()

#..
df1 = df[group_1]
df1
# TODO: Implementation of dataset and dataloader class
# class
class customDataset(Dataset):
  def __init__(self, data):

  def __getitem__(self, item):

  def __len__(self):

# function
def process(data)

# function
def seq1(dataframe):

# function
def train(dataframe):

# function
def test(dataframe):

# function
def itr_merge(*itrs):

# function
def predict(model, data):

# class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):

    def forget(self, x, h):

    def input(self, x, h):

    def mem(self, i, f, x, h, c_prev):

    def out(self, x, h):

    def forward(self, seq):