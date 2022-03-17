#!/usr/bin/env python
# coding:utf-8


import numpy as np
from tensorflow.contrib import learn

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]
        positive_examples = positive_examples[:int(options['dataset_rate']*len(positive_examples))]


        max_document_length = max([len(x.split(",")) for x in positive_examples])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping


    def load_generator_data(self, sample_size):
        text = self.text
        mod_size = len(text) - len(text)%sample_size
        text = text[0:mod_size]
        text = text.reshape(-1, sample_size)
        return text, self.vocab_indexed


    def string_to_indices(self, sentence, vocab):
        indices = [ vocab[s] for s in sentence.split(',') ]
        return indices

    def inidices_to_string(self, sentence, vocab):
        id_ch = { vocab[ch] : ch for ch in vocab } 
        sent = []
        for c in sentence:
            if id_ch[c] == 'eol':
                break
            sent += id_ch[c]

        return "".join(sent)

   
