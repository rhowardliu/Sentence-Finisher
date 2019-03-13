import csv
import json
import torch
import numpy as np
import pdb
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_data(infile):
  with open(infile, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    data = [[int(word)for word in row] for row in csv_reader]
    return data

def split_input_output(sentences, min_input, min_pred):
  data_x = []
  data_y = []
  for sentence in sentences:
    sen_len = len(sentence)
    if sen_len <= min_input or sen_len-min_input< min_pred: continue
    for i in range(min_pred,sen_len-min_input+1):
      x = sentence[:-i]
      y = sentence[-i:]
      data_x.append(x)
      data_y.append(y)
  return data_x,data_y



def pad_data(data, sort=None):
  sent_len = torch.LongTensor([len(row) for row in data])
  pdb.set_trace()
  longest_sent = max(sent_len)
  batch_size = len(data)
  padded_data = torch.zeros((batch_size, longest_sent)).long()
  for i, x_len in enumerate(sent_len):
    sequence = torch.LongTensor(data[i])
    padded_data[i, :x_len] = sequence[:x_len]
  if sort is None:
    sent_len, sort = sent_len.sort(0, descending=True)
  else:
    sent_len=sent_len[sort]
  padded_data = padded_data[sort]
  return padded_data, sent_len, sort


class SentenceFinisher(nn.Module):
  """docstring for SentenceFinisher"""
  def __init__(self, nb_vocab, embed_dims, nb_layers, nb_hidden):
    super(SentenceFinisher, self).__init__()
    self.word_embedding = nn.Embedding(nb_vocab, embed_dims, padding_idx = 0)
    self.lstm = nn.LSTM(input_size = embed_dims, hidden_size = nb_hidden, num_layers=nb_layers, batch_first = True)
    self.linear = nn.Linear(nb_hidden, embed_dims)

  def forward(self, x, sent_len):
    #bs x seq x 1 -> bs x seq x embed
    x = self.word_embedding(x)
    #bs x seq x embed -> bs x seq x hidden
    packed_input = nn.utils.rnn.pack_padded_sequence(x, sent_len, batch_first=True)
    packed_output, (h_n,c_n) = self.lstm(packed_input)
    pdb.set_trace()
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    pdb.set_trace()
    output = output.contiguous()
    bs_seq_shape = output.shape[:2]
    #bs x seq x hidden -> (bs*seq) x hidden
    output = output.view(-1, output.shape[2])
    pdb.set_trace()
    #(bs*seq) x hidden -> (bs*seq) x embed
    output = self.linear(output)
    pdb.set_trace()
    #(bs*seq) x embed -> bs x seq x embed
    output = output.view(*bs_seq_shape, -1)
    pdb.set_trace()
    return output, self.word_embedding


def loss_mse(output, target, embed_fn):
  with torch.no_grad():
    pdb.set_trace()
    return F.mse_loss(embed_fn(output), embed_fn(target))

def fit(epochs, loss, train_dl, model, opt):
  for epoch in range(epochs):
    model.train()
    step = 0
    for xb, sent_len, yb in train_dl:
      #add data into gpu
      xb = xb.to(dev)
      sent_len = sent_len.to(dev)
      yb = yb.to(dev)

      print('step', step+1)
      output, embed = model(xb, sent_len)
      loss = loss(xb, yb, embed)
      print(loss)

      loss.backward()
      opt.step()
      opt.zero_grad()
      step+=1
  return model

if __name__ == '__main__':
  #files
  infile = './data2.csv'
  dicfile = './word_dic2.json'

  #data
  data = load_data(infile)
  pdb.set_trace()
  data_x, data_y = split_input_output(data, min_input=5, min_pred=3)
  print('Split data into x and y')
  data_x, sent_len_x, perm_idx = pad_data(data_x)
  data_y, sent_len_y, _ = pad_data(data_y, perm_idx)
  print('Padded data')

  #hyperparameters
  nb_vocab = 5845  
  bs = 64
  embed = 15
  nb_layers = max(sent_len_y)
  nb_hidden = 20
  lr = 0.0001
  epochs = 1

  #dataloaders
  train_ds = TensorDataset(data_x, sent_len_x, data_y)
  train_dl = DataLoader(train_ds, bs)

  #model
  model = SentenceFinisher(nb_vocab = nb_vocab, embed_dims = embed, nb_layers = nb_layers, nb_hidden = nb_hidden)
  model.to(dev)
  opt = optim.Adam(model.parameters(), lr)
  print('Initialised model')

  #train
  model = fit(epochs = epochs, loss=loss_mse, train_dl=train_dl, model=model, opt=opt)
  torch.save(model.state_dict(), './sentence_model.pth')




