#!/usr/bin/python
# coding=utf8

from __future__ import print_function

from exceptions import *
from options import Options as opt
from word import Word

import os
import re
import tensorflow as tf
import pickle as pk


class Vocab(object):
    """Vocabulary of the train corpus, used for embedding lookup and sense number lookup"""

    # vocab = {
    #   'wordToken': {
    #     'wordCount': Integer,
    #     'senseCount': Integer,
    #     'index': Integer
    #     /* 'senses': [
    #       'mean': tf.Variable,
    #       'sigma': tf.Variable
    #     ] */
    #   }, ...
    # }

    def __init__(self, file = None):
        # type: (String) -> Vocab
        self._wordSeparator = re.compile('\\s|(\\s*,\\s*)|(\\s*.\\s*)')
        self._vocab = {}
        self._idx2word = []
        self.size = 0
        self.totalWordCount = 0
        self.totalSenseCount = 0

        with open('data/coarse-grained-all-words/senseNumberDict.pk') as f:
            self._senseNum = pk.load(f)

        if file is not None:
            self.parse(file)

    def _parseLine(self, line):
        words = self._wordSeparator.split(line)
        for word in words:
            self.totalWordCount += 1

            if word in self._vocab.keys():
                self._vocab[word].count += 1
            else:
                if word in self._senseNum.keys():
                    self.totalSenseCount += self._senseNum[word]
                    self._vocab[word] = Word(word, self.size, self._senseNum[word]).initSenses()
                else:
                    self.totalSenseCount += 1
                    self._vocab[word] = Word(word, self.size).initSenses()

                self._idx2word.append(self._vocab[word])
                self.size += 1

                if self.size % 100 == 0:
                    print('Added', self.size, 'words totally.')
                # {
                #     'wordCount': 1,
                #     'senseCount': 1,
                #     'index': self.size
                # }

    def parse(self, file):
        if os.path.isfile(file):
            self.corpus = file

            with open(file) as f:
                for line in f.readlines():
                    self._parseLine(line)
        else:
            raise NotAFileException(file)

    def getWord(self, word):
        if isinstance(word, str):
            try:
                return self._vacob[word]
            except KeyError:
                return None
        elif isinstance(word, int):
            if word < self.size and word >= 0:
                return self._idx2word[word]
        else:
            return None

    def getSenseCount(self, word):
        try:
            return self._vacob[word].senseCount
        except KeyError:
            return 0

    def getWordCount(self, word):
        try:
            return self._vacob[word].count
        except KeyError:
            return 0

    def getWordFreq(self, word):
        return float(self.getWordCount()) / self.totalWordCount

    def save(self, file):
        try:
            with open(file, 'wb') as f:
                pk.dump(self, f)
        except:
            return False

    def load(self, file):
        try:
            if os.path.isfile(file):
                with open(file) as f:
                    self = pk.load(f)
                    return True
        except:
            return False

# if __name__ == '__main__':
#     opt.embSize = 100;

#     w1 = Word('w', 1)
#     w1.initSenses()

#     print(w1.means[0].get_shape())
