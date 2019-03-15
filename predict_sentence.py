import torch
from torch.nn import functional as F
import json
from lstm import SentenceFinisher
from data_processing import sentence_to_tokens, change_to_int
import pdb
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def pad_sentence(sentence, seq_len):
  padded_data = torch.zeros(1,seq_len).long()
  sequence = torch.LongTensor(sentence)
  padded_data[0,:len(sentence)] = sequence[:]
  return padded_data

def construct_sentence(int_seq, int_dic):
  sentence = ''
  for i in int_seq:
    if i in int_dic: sentence += int_dic[i] + ' '
  sentence += '.'
  return sentence


def reverse_dic(word_dic):
  reverse = {}
  for word, idx in word_dic.items():
    reverse[idx] = word
  return reverse

if __name__ == '__main__':
  model_path = './sentence_model.pth'
  word2int = json.load(open('./word_dic2.json'))
  int2word = reverse_dic(word2int)

  #hyperparameters
  nb_vocab = 5845  
  bs = 128
  embed = 30
  nb_hidden = 50
  lr = 0.001
  epochs = 300
  nb_seq = 25

  model = SentenceFinisher(nb_vocab = nb_vocab, embed_dims = embed, nb_layers = nb_seq, nb_hidden = nb_hidden)
  model.load_state_dict(torch.load(model_path))
  model.to(dev)
  model.eval()
  with torch.no_grad():  
    while(1):
      print('Give the start of a sentence')
      sent_start = input()
      tokens = sentence_to_tokens(sent_start)
      try:
        xb = [word2int[word] for word in tokens]
      except Exception as e:
        print('Some word is not found in the dictionary. Please try another sentence')
        continue
      xb = pad_sentence(xb,nb_seq).to(dev)
      out = model(xb,[len(tokens)])
      out = torch.argmax(out, dim=1)
      out_seq = out.tolist()
      sentence = construct_sentence(out_seq, int2word)
      print(sentence)
      pdb.set_trace()


