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
  # if sort is None:
  #   sent_len, sort = sent_len.sort(0, descending=True)
  # else:
  #   sent_len=sent_len[sort]
  # padded_data = padded_data[sort]
  return padded_data, sent_len


class WrappedLoader(object):
  """docstring for WrappedLoader"""
  def __init__(self, dl):
    super(WrappedLoader, self).__init__()
    self.dl = dl

  def sort_data(self, sent_len):
    sent_len, sort = sent_len.sort(0, descending=True)
    return sort

  def __len__(self):
    return len(self.dl)

  def __iter__(self):
    batches = iter(self.dl)
    for b in batches:
      sort = self.sort_data(b[1])
      b = [a.to(dev) for a in b]
      b = [a[sort] for a in b]
      yield b



class SentenceFinisher(nn.Module):
  """docstring for SentenceFinisher"""
  def __init__(self, nb_vocab, embed_dims, nb_layers, hidden_lstm, hidden_fc, bs, dropout_lstm, dropout_fc):
    self.nb_layers = nb_layers
    self.hidden_lstm = hidden_lstm
    super(SentenceFinisher, self).__init__()
    self.word_embedding = nn.Embedding(nb_vocab, embed_dims, padding_idx = 0)
    self.lstm = nn.LSTM(input_size = embed_dims, hidden_size = hidden_lstm, num_layers=nb_layers, batch_first = True, dropout = dropout_lstm)
    # self.linear1 = nn.Linear(hidden_lstm, hidden_fc)
    # self.dropout_fc = nn.Dropout(p=dropout_fc)
    self.linear2 = nn.Linear(hidden_fc, nb_vocab)

  def init_lstm_states(self, bs):
    hidden = torch.randn(self.nb_layers, bs, self.hidden_lstm, requires_grad=True).to(dev)
    cell = torch.randn(self.nb_layers, bs, self.hidden_lstm, requires_grad = True).to(dev)
    return hidden, cell

  def forward(self, x, sent_len):
    #initialise hidden and cell state
    bs_len, seq_len = x.shape
    self.hidden = self.init_lstm_states(bs_len)

    #bs x seq x 1 -> bs x seq x embed   
    x = self.word_embedding(x)
    #bs x seq x embed -> bs x seq x hidden_lstm
    packed_input = nn.utils.rnn.pack_padded_sequence(x, sent_len, batch_first=True)
    packed_output, self.hidden = self.lstm(packed_input, self.hidden)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
    # bs_seq_shape = output.shape[:2]
    #bs x seq x hidden_lstm -> (bs*seq) x hidden_lstm
    output = output.contiguous()    
    output = output.view(-1, output.shape[2])
    #(bs*seq) x hidden_lstm -> (bs*seq) x hidden_linear
    # output = self.linear1(output)
    # output = F.relu(output)
    # output = self.dropout_fc(output)
    #(bs*seq) x hidden_linear -> (bs*seq) x vocab
    output=self.linear2(output)
    return output

def loss_mse(output, target, embed_fn):
  with torch.no_grad():
    embed_output = embed_fn(output)
    embed_target = embed_fn(target)
  return F.mse_loss(embed_output, embed_target)

def accuracy(out, yb):
  preds = torch.argmax(out, dim=1)
  return (preds == yb).float().mean()

def adjust_target(dataset):
  dataset = dataset.view(-1).type(torch.int64)
  return dataset

def fit(epochs, loss_fn, train_dl, valid_dl, model, opt):
  for epoch in range(epochs):
    model.train()
    training_stats = []
    for xb, sent_len, yb in train_dl:

      yb = adjust_target(yb)
      output = model(xb, sent_len)
      loss = loss_fn(output, yb, ignore_index=0)
      epoch_loss = loss

      loss.backward()
      opt.step()
      opt.zero_grad()

    model.eval()
    with torch.no_grad():
      for xb, sent_len, yb in valid_dl:
        yb = adjust_target(yb)
        output = model(xb, sent_len)
        training_stats.append([loss_fn(output, yb),accuracy(output, yb)])
    training_stats = np.asarray(training_stats)
    epoch_loss, epoch_accuracy = np.mean(training_stats, axis=0)
    print('Epoch: {} Loss: {} Acc: {}'.format(epoch, epoch_loss, epoch_accuracy))
  return model

if __name__ == '__main__':
  #files
  infile = './data2.csv'
  dicfile = './word_dic2.json'

  #hyperparameters
  nb_vocab = 5845  
  bs = 128
  embed = 30
  lstm_layers = 2
  hidden_lstm = 100
  hidden_linear = 100
  lr = 0.0001
  epochs = 40
  seq_len = 50
  dropout_lstm = 0.1
  dropout_fc = 0.1


  #data
  data = load_data(infile)
  data_x, data_y = split_input_output(data, min_input=5, min_pred=10, max_len=seq_len)
  print('Split data into x and y')
  data_x, sent_len_x = pad_data(data_x, nb_seq = seq_len)
  data_y, _ = pad_data(data_y, nb_seq = seq_len)
  print('Padded data')


  #dataloaders
  train_ds = TensorDataset(data_x, sent_len_x, data_y)
  train_dl = DataLoader(train_ds, bs)
  train_dl = WrappedLoader(train_dl)
  valid_dl = DataLoader(train_ds, bs*2)
  valid_dl = WrappedLoader(valid_dl)
  pdb.set_trace()

  #model
  model = SentenceFinisher(nb_vocab = nb_vocab, embed_dims = embed, nb_layers = lstm_layers, hidden_lstm = hidden_lstm, hidden_fc=hidden_linear, bs=bs, dropout_lstm=dropout_lstm, dropout_fc=dropout_fc)
  model.to(dev)
  opt = optim.Adam(model.parameters(), lr)
  loss_fn = F.cross_entropy
  print('Initialised model')

  #train
  model = fit(epochs = epochs, loss_fn=loss_fn, train_dl=train_dl, valid_dl=valid_dl, model=model, opt=opt)
  torch.save(model.state_dict(), './sentence_model.pth')




