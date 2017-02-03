#!/usr/bin/python
# coding=utf8

from __future__ import print_function

from exceptions import *
from word import Word

import os
import sys
import re
import threading
import tensorflow as tf
import pickle as pk
from cStringIO import StringIO
from threadpool import *


class Vocab(object):
    """ Vocabulary of the train corpus, used for embedding lookup and sense number lookup. """

    def __init__(self, file=None):
        self._wordSeparator = re.compile('\\s|(\\s*,\\s*)|(\\s*\\.\\s*)')
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


    def _parseLineThread(self, line):
        words = self._wordSeparator.split(line)
        for word in words:
            if self.mutex.acquire(1):
                self.totalWordCount += 1

                if word in self._vocab.keys():
                    self._vocab[word].count += 1
                else:
                    if word in self._senseNum.keys():
                        self.totalSenseCount += self._senseNum[word]
                        self._vocab[word] = Word(word, self.size, self._senseNum[word])
                    else:
                        self.totalSenseCount += 1
                        self._vocab[word] = Word(word, self.size)

                    self._idx2word.append(self._vocab[word])
                    self.size += 1

                    if self.size % 100 == 0:
                        sys.stdout.write('\rAdded %d words totally.' % self.size)
                        sys.stdout.flush()
                self.mutex.release()


    def parse(self, file, threadNum=10, parseUnitLength=100):
        if os.path.isfile(file):
            self.corpus = file
            stc = StringIO()
            wordCount = 0
            tp = ThreadPool(threadNum)
            self.mutex = threading.Lock()

            with open(file) as f:
                c = f.read(1)
                while c != '':
                    if self._wordSeparator.search(c) != None:
                        wordCount += 1

                    if wordCount == parseUnitLength:
                        requests = makeRequests(self._parseLineThread, [stc.getvalue()])

                        for req in requests:
                            tp.putRequest(req)

                        wordCount = 0
                        del(stc)
                        stc = StringIO()
                        continue

                    stc.write(c)
                    c = f.read(1)
            tp.wait()
            print('\nParse Finished.')
        else:
            raise NotAFileException(file)


    def getWord(self, word):
        if isinstance(word, str):
            try:
                return self._vocab[word]
            except KeyError:
                return None
        elif isinstance(word, int):
            if word < self.size and word >= 0:
                return self._idx2word[word]
        else:
            return None


    def getSenseCount(self, word):
        try:
            return self._vocab[word].senseCount
        except KeyError:
            return 0


    def getWordCount(self, word):
        try:
            return self._vocab[word].count
        except KeyError:
            return 0


    def getWordFreq(self, word):
        return float(self.getWordCount()) / self.totalWordCount


    def save(self, file):
        try:
            l = []

            for i in self._idx2word:
                l.append((i.token, i.senseNum, i.count, i.index ))
            with open(file, 'wb') as f:
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount}, f)
                return True
        except:
            return False


    def load(self, file):
        try:
            if os.path.isfile(file):
                with open(file) as f:
                    data = pk.load(f)

                    self.totalSenseCount = data['tsc']
                    self.totalWordCount = data['twc']

                    for i in data['words']:
                        w = Word(i[0], i[3], i[1], i[2])
                        w.initSenses()
                        self._idx2word.append(w)
                        self._vocab[i[0]] = self._idx2word[-1]

                    self.size = len(data['words'])
                    return True
        except:
            return False
