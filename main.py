import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import re

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