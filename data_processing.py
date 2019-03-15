import string
from nltk import tokenize
import numpy as np
import csv
import json

SEQUENCE_LENGTH = 4

# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# turn a doc into clean tokens
def clean_doc(doc):
  # replace '--' with a space ' '
  doc = doc.replace('--', ' ')
  # split into tokens by white space
  tokens = doc.split()
  # remove punctuation from each token
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # make lower case
  tokens = [word.lower() for word in tokens]
  return tokens

def save_dic(data, infile):
  json.dump(data, open(infile,'w'))


def sentence_to_tokens(sentence):
    tokens = sentence.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


def clean_sentences(sentences):
  new_list = []
  for sentence in sentences:
    tokens = sentence_to_tokens(sentence)
    new_list.append(tokens)
  return new_list

def getSentences(doc):
  sentences = tokenize.sent_tokenize(doc)
  sentences = [extrasplit for sent in sentences for extrasplit in sent.split(';')]
  return sentences

def getVocab(sentences):
  flatten = [word for sentence in sentences for word in sentence]
  return set(flatten)

def word2int(vocab):
  word_dic = {}
  for i, token in enumerate(vocab):
    word_dic[token] = i+1
  return word_dic

 

def change_to_int(sentences, word_dic):
  new_sentences = []
  for sentence in sentences:
    new_sentence = [word_dic[word] for word in sentence]
    new_sentences.append(new_sentence)
  return new_sentences



def organise_data(tokens, seq_len):
  length = seq_len + 1
  sequences = list()
  for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
  return sequences


# save tokens to file, one dialog per line
def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

def write_csv(data, filename):
  with open(filename,'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
  f.close()

def combine_to_sentence(tokens):
  sequences = []
  for token in tokens:
    line = ' '.join(token)
    sequences.append(line)
  return sequences

if __name__ == '__main__':

  in_filename = 'origin_of_species'
  out_filename = 'species_sentences'
  doc = load_doc(in_filename)
  sentences = getSentences(doc)
  sentences = clean_sentences(sentences)
  vocab = getVocab(sentences)
  word_dic = word2int(vocab)
  save_dic(word_dic, 'word_dic2.json')
  save_doc(combine_to_sentence(sentences), out_filename)
  data = change_to_int(sentences, word_dic)
  write_csv(data, 'data2.csv')
  
