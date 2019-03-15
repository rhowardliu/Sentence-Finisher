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

def split_input_output(sentences, min_input, min_pred, max_len):
  data_x = []
  data_y = []
  for sentence in sentences:
    sen_len = len(sentence)
    if sen_len>max_len: continue
    if sen_len <= min_input or sen_len-min_input< min_pred: continue
    for i in range(min_pred,sen_len-min_input+1):
      x = sentence[:-i]
      y = sentence[-i:]
      data_x.append(x)
      data_y.append(y)
  return data_x,data_y



def pad_data(data, nb_seq=None, sort=None):
  sent_len = torch.LongTensor([len(row) for row in data])
  if nb_seq is None:
    nb_seq = max(sent_len)
  batch_size = len(data)
  padded_data = torch.zeros((batch_size, nb_seq)).long()
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
  def __init__(self, nb_vocab, embed_dims, nb_layers, hidden_lstm, hidden_fcnn):
    self.nb_layers = nb_layers
    super(SentenceFinisher, self).__init__()
    self.word_embedding = nn.Embedding(nb_vocab, embed_dims, padding_idx = 0)
    self.lstm = nn.LSTM(input_size = embed_dims, hidden_size = hidden_lstm, num_layers=nb_layers, batch_first = True)
    self.linear1 = nn.Linear(hidden_lstm, hidden_fcnn)
    self.linear2 = nn.Linear(hidden_fcnn, nb_vocab)

  def forward(self, x, sent_len):
    #bs x seq x 1 -> bs x seq x embed
    x = self.word_embedding(x)
    #bs x seq x embed -> bs x seq x hidden_lstm
    packed_input = nn.utils.rnn.pack_padded_sequence(x, sent_len, batch_first=True)
    packed_output, (h_n,c_n) = self.lstm(packed_input)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=self.nb_layers)
    output = output.contiguous()
    # bs_seq_shape = output.shape[:2]
    #bs x seq x hidden_lstm -> (bs*seq) x hidden_lstm
    output = output.view(-1, output.shape[2])
    #(bs*seq) x hidden_lstm -> (bs*seq) x hidden_linear
    output = self.linear1(output)
    F.relu_(output)
    #(bs*seq) x hidden_linear -> (bs*seq) x vocab
    output=self.linear2(output)
    return output

def loss_mse(output, target, embed_fn):
  with torch.no_grad():
    embed_output = embed_fn(output)
    embed_target = embed_fn(target)
  return F.mse_loss(embed_output, embed_target)

def adjust_target(dataset):
  dataset = dataset.view(-1).type(torch.int64)
  return dataset

def fit(epochs, loss_fn, train_dl, model, opt):
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, sent_len, yb in train_dl:
      #add data into gpu
      xb = xb.to(dev)
      sent_len = sent_len.to(dev)
      yb = yb.to(dev)

      yb = adjust_target(yb)
      output = model(xb, sent_len)
      loss = loss_fn(output, yb)
      epoch_loss = loss

      loss.backward()
      opt.step()
      opt.zero_grad()
    print('Epoch: {} Loss: {}'.format(epoch, epoch_loss))      
  return model

if __name__ == '__main__':
  #files
  infile = './data2.csv'
  dicfile = './word_dic2.json'

  #hyperparameters
  nb_vocab = 5845  
  bs = 128
  embed = 50
  hidden_lstm = 100
  hidden_linear = 100
  lr = 1
  epochs = 100
  seq_len = 35


  #data
  data = load_data(infile)
  data_x, data_y = split_input_output(data, min_input=2, min_pred=5, max_len=seq_len)
  print('Split data into x and y')
  data_x, sent_len_x, perm_idx = pad_data(data_x, nb_seq = seq_len)
  data_y, sent_len_y, _ = pad_data(data_y, seq_len, perm_idx)
  print('Padded data')


  #dataloaders
  train_ds = TensorDataset(data_x, sent_len_x, data_y)
  train_dl = DataLoader(train_ds, bs)

  #model
  model = SentenceFinisher(nb_vocab = nb_vocab, embed_dims = embed, nb_layers = seq_len, hidden_lstm = hidden_lstm, hidden_fcnn=hidden_linear)
  model.to(dev)
  opt = optim.Adam(model.parameters(), lr)
  loss_fn = F.cross_entropy
  print('Initialised model')

  #train
  model = fit(epochs = epochs, loss_fn=loss_fn, train_dl=train_dl, model=model, opt=opt)
  torch.save(model.state_dict(), './sentence_model.pth')




