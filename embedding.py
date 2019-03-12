import torch
from torch import nn
# from torch import DataLoader
from torch.utils.data import TensorDataset
from data_processing import load_doc
from torch import optim
import torch.nn.functional as F
import numpy as np
import json
from nltk.corpus import stopwords
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_dic(infile, data):
  json.dump(data, open(infile,'w'))

def rem_stop_words(text):
  stop_words = set(stopwords.words('english'))
  return [word for word in text if word not in stop_words]

def getVocab(sentences):
  flatten = [word for word in sentences]
  return set(flatten)

def word2int(vocab):
  word_dic = {}
  for i, token in enumerate(vocab):
    word_dic[token] = i
  return word_dic

def one_hot_encoding(vocab_size, data_index):
  vector = torch.zeros((vocab_size))
  vector[data_index] = 1
  return vector

def parse_to_data(sentences, context_size, word_dic):
  data = []
  for idx, word in enumerate(sentences):
      for neighbor in sentences[max(idx - context_size, 0) : min(idx + context_size, len(sentences)) + 1] : 
          if neighbor != word:
              data.append([word_dic[word], word_dic[neighbor]])
  return data



def text_to_data(text, context_size):
  vocab = getVocab(text)
  word_dic = word2int(vocab)
  data = parse_to_data(text, context_size, word_dic)
  return vocab, word_dic, data


def word_to_onehot(word, word_dic):
  vocab_size = len(word_dic)
  index = word_dic[word]
  return one_hot_encoding(vocab_size, index)


class Word_Embed(nn.Module):
  def __init__(self, bs, dim, vocab):
    super().__init__()
    self.lin1 = nn.Linear(vocab, dim)
    self.lin2 = nn.Linear(dim, vocab)

  def forward(self, x):
    x = self.lin1(x)
    x = self.lin2(x)
    return x

def fit(model, opt, loss_func, epochs, bs):
  for epoch in range(epochs):
    for i in range((n-1)//bs+1):
      start_i = i * bs
      end_i = start_i + bs
      x_bs = data[start_i:end_i,0]
      y_bs = data[start_i:end_i,1].to(dev)
      xb = torch.stack([one_hot_encoding(vocab_size, x) for x in x_bs]).to(dev)
      # yb = torch.stack([one_hot_encoding(vocab_size, y) for y in y_bs]).squeeze().to(dev)
      model.train()
      pred = model(xb)
      loss = loss_func(pred, y_bs)
      acc = accuracy(pred, y_bs)
      print('EPOCH: {} step {} loss: {} accuracy: {}'.format(epoch+1, i+1, loss, acc))
      
      loss.backward()
      opt.step()
      opt.zero_grad()
  torch.save(model.state_dict(), './embed_model.pth')


def accuracy(out, yb):
  preds = torch.argmax(out, dim=1)
  return (preds==yb).float().mean()

if __name__ == '__main__':
  in_file = 'species_sentences.txt'
  text = load_doc(in_file)
  text = text.replace('\n', ' ')
  text = text.split()
  text = rem_stop_words(text)
  vocab, word_dic, data = text_to_data(text, 3)
  save_dic('./vocab_dic.json',word_dic)
  data = torch.Tensor(data)
  data = data.type(torch.int64)
  data_numpy = data.numpy()  
  np.savetxt('data.csv',data_numpy,delimiter=',')
  vocab_size = len(vocab)
  n=len(data)
  epochs=40
  lr = 0.0001
  bs = 4096
  dim=40
  model = Word_Embed(bs, dim, vocab_size)
  model = model.to(dev)
  opt = optim.Adam(model.parameters(), lr=lr)
  loss_func = F.cross_entropy
  fit(model, opt, loss_func, epochs, bs)
