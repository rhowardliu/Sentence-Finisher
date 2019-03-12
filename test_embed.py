import json
import torch
from embedding import Word_Embed
from embedding import one_hot_encoding

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def id_dic(dic):
  int2word = {}
  for word, vector_id in dic.items():
    int2word[vector_id] = word
  return int2word



if __name__ == '__main__':
  model_PATH = './embed_model.pth'
  word2int = json.load(open('./vocab_dic.json'))
  int2word = id_dic(word2int)
  bs = 4096
  dim=40
  vocab_size = len(word2int)  
  model = Word_Embed(bs, dim, vocab_size)
  model.load_state_dict(torch.load(model_PATH))
  model.to(dev)
  model.eval()
  with torch.no_grad():
    while(1):
      print('input word: ')
      word2test = input()
      xb = one_hot_encoding(vocab_size, word2int[word2test]).to(dev)
      out_int = torch.argmax(model(xb))
      print(int2word[int(out_int)])
