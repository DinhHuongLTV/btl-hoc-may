import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import re
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
class customDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, item):
    return self.data[item]

  def __len__(self):
    return len(self.data)
  
output_size = 30
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 4
hidden_size = 128
num_layers = 1
lr = 0.003
weight_decay = 1e-4
path = '/content/UntitledFolder/test.pt'
epochs = 50

# function
def process(data):
  load = data[data.columns[0]].tolist()
  data = data.values.tolist()
  max, min = np.max(load), np.min(load)
  load = (load-min) / (max-min)
  seq = []

  for i in range(0, len(data) - 24 - output_size, output_size):
    train_seq = []
    train_label = []

    for j in range(i, i + 24):
      x = [load[j]]

      for c in range(1, 4):
        x.append(data[j][c])

      train_seq.append(x)

    for j in range(i + 24, i + 24 + output_size):
      train_label.append(load[j])

    train_seq = torch.FloatTensor(train_seq)
    train_label = torch.FloatTensor(train_label).view(-1)

    seq.append((train_seq, train_label))

  seq = customDataset(seq)
  seq = DataLoader(dataset=seq, batch_size = BATCH_SIZE, shuffle=False)
  data = seq, [max, min]
  return data

# function
def seq1(dataframe):
  data = dataframe
  # split train test
  train = data[:int(len(data)*0.7)]
  test = data[int(len(data)*0.7):len(data)]

  data_train = process(train)
  data_test = process(test)
  return data_train, data_test

# class
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layer, output_size):
    super().__init__()
    self.input_size = input_size
    self.num_layer = num_layer
    self.hidden_size = hidden_size
    self.output_size = output_size

    # forget
    self.ln_forget_w1 = nn.Linear(self.input_size, self.hidden_size, bias= True)
    self.ln_forget_r1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.sigmoid_forget = nn.Sigmoid()

    # input gate
    self.ln_input_w2 = nn.Linear(self.input_size, self.hidden_size, bias= True)
    self.ln_input_r2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.sigmoid_in = nn.Sigmoid()

    # cell memory
    self.ln_mem_w3 = nn.Linear(self.input_size, self.hidden_size, bias= True)
    self.ln_mem_r3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.activation_gate = nn.Tanh()

    # out gate
    self.ln_out_w4 = nn.Linear(self.input_size, self.hidden_size, bias= True)
    self.ln_out_r4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.sigmoid_out = nn.Sigmoid()

    # final activation
    self.activation = nn.Tanh()

    # output
    self.linear_out = nn.Linear(self.hidden_size, self.output_size)

  def forget(self, x, h):
    x = self.ln_forget_w1(x)
    h = self.ln_forget_r1(h)
    return self.sigmoid_forget(x+h)

  def input(self, x, h):
    x = self.ln_input_w2(x)
    h = self.ln_input_r2(h)
    return self.sigmoid_in(x+h)

  def mem(self, i, f, x, h, c_prev):
    x = self.ln_mem_w3(x)
    h = self.ln_mem_r3(h)
    k = self.activation_gate(x+h)
    g = k * i
    c = f * c_prev
    c_next = g + c
    return c_next

  def out(self, x, h):
    x = self.ln_out_w4(x)
    h = self.ln_out_r4(h)
    return self.sigmoid_out(x + h)

  def forward(self, seq):
    batch_size, len_seq, _ = seq.size()
    h = torch.randn(batch_size, self.hidden_size).to(seq.device)
    c = torch.randn(batch_size, self.hidden_size).to(seq.device)
    pred = []
    for i in range(len_seq):
      x = seq[:,i,:]
      i = self.input(x, h)
      f = self.forget(x, h)
      m = self.mem(i, f, x, h, c)
      o = self.out(x, h)
      h = o * self.activation(m)
      pred.append(h.unsqueeze(1))
    pred = torch.cat(pred, dim=1)
    pred = self.linear_out(pred)
    pred = pred[:, -1, :]
    return pred
  # function
#from scipy.interpolate import make_interp_spline
def train(dataframe):
  dtr, dte = seq1(dataframe)
  d1, l1 = dtr
  d2, l2 = dte
  model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
  loss_function = nn.MSELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  losses = []
  loss = 0
  for i in range(epochs):
    cnt = 0
    epochs_loss = 0.0
    for (seq, label) in d1:
      cnt += 1
      seq = seq.to(device)
      label = label.to(device)
      y_pred = model(seq)
      loss = loss_function(y_pred, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epochs_loss += loss.item()
    print('epoch', i, ':', loss.item())
    avg_loss = epochs_loss / cnt
    losses.append(avg_loss)
  plt.plot(losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
  state = {'models': model.state_dict(), 'optimizer': optimizer.state_dict()}
  torch.save(state, path)

def test(dataframe):
    dtr, dte = seq1(dataframe)
    d1, l1 = dtr
    d2, l2 = dte
    pred = []
    y = []
    print('loading models...')
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in d2:
      target = [item for sublist in target.data.tolist() for item in sublist]
      y.extend(target)

      seq = seq.to(device)
      with torch.no_grad():
          y_pred = model(seq)
          y_pred = [item for sublist in y_pred.data.tolist() for item in sublist]
          pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    m, n = l2[0], l2[1]
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', np.mean(np.abs((y - pred) / y)))
    # plot
    x = np.arange(len(y))
    plt.plot(x, y, c='green', marker='*', ms=1, alpha=0.5, label='true')
    plt.plot(x, pred, c='red', marker='o', ms=1, alpha=0.5, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
#train
train(df1)

model = LSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load(path)['models'])
model.eval()

#test
test(df1.head(1000))
df1.head(215).tail(70).head(30)
# function
def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v
def predict(model, data):
    pred = []
    d, l = data
    for (seq, target) in d:
          with torch.no_grad():
              y_pred = model(seq)
              y_pred = [item for sublist in y_pred.data.tolist() for item in sublist]
              pred.extend(y_pred)
    return pred
#loaded
loaded_model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load(path)['models'])
loaded_model.eval()

dt = process(df1.head(1000))
d, l = dt
m, n = l[0], l[1]

target = np.array(df1[df1.columns[0]].head(100).tolist())

pred = predict(loaded_model, dt)


arr_pred = np.array(pred)[:output_size]



arr_pred = ((m - n) * arr_pred + n)

upper = target[-1]
lower = arr_pred[0]
print(m, n)
alpha = upper / lower

arr_pred = alpha * arr_pred

# Plot input data
x_input = np.arange(len(target))
plt.plot(x_input, target, c='blue', marker='o', ms=1, alpha=0.5, label='input')

# Plot connected predictions
x_pred = np.arange(len(arr_pred)) + len(x_input)
plt.plot(x_pred, arr_pred, c='red', marker='o', ms=1, alpha=0.5, label='pred')

plt.grid(axis='y')
plt.legend()
plt.show()

#pred2 = predict(loaded_model, df1)
#print(len(pred), len(pred2))
#for i, e in enumerate(pred2):
#  print(pred2[i], pred[i])