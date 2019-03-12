import string
from nltk import tokenize
import numpy as np

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

def clean_sentences(sentences):
  new_list = []
  for sentence in sentences:
    tokens = sentence.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    new_list.append(tokens)
  return new_list

def getSentences(doc):
  sentences = tokenize.sent_tokenize(doc)
  return sentences



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

def combine_to_sentence(tokens):
  sequences = []
  for token in tokens:
    line = ' '.join(token)
    sequences.append(line)
  return sequences

if __name__ == '__main__':

  in_filename = 'origin_of_species.txt'
  out_filename = 'species_sentences.txt'
  doc = load_doc(in_filename)
  sentences = getSentences(doc)
  cleaned = clean_sentences(sentences)
  save_doc(combine_to_sentence(cleaned), out_filename)
