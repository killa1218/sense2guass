#!/usr/bin/python
#coding=utf8

from __future__ import print_function

from exceptions import NotAFileException

import os
import tensorflow as tf
import word


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
        self._wordSeparator = re.compile('\\s|(\\s*,\\s*)|(\\s*.\\s*)')
        self._vocab = {}
        self._idx2word = []
        self.size = 0
        self.totalWordCount = 0
        self.totalSenseCount = 0

        if file != None:
            self.parse(file)


    def _parseLine(self, line):
        words = self._wordSeparator.split(line)
        for word in words:
            self.totalWordCount ++

            try:
                self._vocab[word].count ++
            except KeyError:
                self._vocab[word] = Word(word, self.size)
                self._idx2word[size] = self._vocab[word]
                self.size ++
                # {
                #     'wordCount': 1,
                #     'senseCount': 1,
                #     'index': self.size
                # }

                # self.totalSenseCount += 1


    def parse(self, file):
        if os.path.isfile(file):
            self.corpus = file

            with open(file) as f:
                if os.path.getsize(self.corpus) > 2000000000:
                    for line in f.readline():
                        self._parseLine(self, line)
                else:
                    for line in f.readlines():
                        self._parseLine(self, line)
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



