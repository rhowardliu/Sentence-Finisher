import unittest
from data_processing import *

text = 'I am a sick man... I am a spiteful man. I think my liver is diseased.'

class DataParseTest(unittest.TestCase):

  def test_parse(self):
    test_sentences = [['i','am','a', 'sick','man'],['i', 'am','a','spiteful','man'], ['i','think','my','liver','is','diseased']]
    sentences = getSentences(text)
    cleaned = clean_sentences(sentences)
    self.assertEqual(test_sentences, cleaned)
    return

  def test_vocab(self):
    test_sentences = [['i','am','a', 'sick','man'],['i', 'am','a','spiteful','man'], ['i','think','my','liver','is','diseased']]
    test_vocab = ['i','am','a','sick','man','spiteful','think','my','liver','is','diseased']
    vocab = getVocab(test_sentences)
    self.assertCountEqual(vocab, test_vocab)
    
  def test_data(self):
    test_sentences = [['i','am','a', 'sick','man'],['i', 'am','a','spiteful','man'], ['i','think','my','liver','is','diseased']]
    test_vocab = ['i','am','a','sick','man','spiteful','think','my','liver','is','diseased']
    word_dic = word2int(test_vocab)
    data = parse_to_data(test_sentences, 3)
    data = convert_to_vector(data, word_dic)
    print(data)
    return


if __name__== '__main__':
  unittest.main()