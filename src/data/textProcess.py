"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""


class TextFileProcess:

    def __init__(self, textFile):
        self.textFile = textFile

    def create_vocab(self):
        with open(self.textFile, 'r') as f:
            content = f.read()
        vocab = set(content.split())
        word_to_ix = {word: ix for ix, word in enumerate(vocab)}
        ix_to_word = {ix: word for ix, word in enumerate(vocab)}
        return vocab, word_to_ix, ix_to_word, content.split()

    def noise_removal(self):
        pass
