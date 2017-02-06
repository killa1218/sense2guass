#!/usr/bin/python
# coding=utf8

from __future__ import print_function

import os
import pickle as pk
import sys
import threading
import tensorflow as tf

from threadpool import *
from exceptions import *
from tqdm import *
from word import Word
from options import Options as opt
from utils.fileIO import fetchSentences
from utils.common import is_number

class Vocab(object):
    """ Vocabulary of the train corpus, used for embedding lookup and sense number lookup. """

    def __init__(self, file=None):
        self._wordSeparator = opt.wordSeparator
        self._vocab = {}
        self._idx2word = []
        self.size = 0
        self.totalWordCount = 0
        self.totalSenseCount = 0

        try:
            with open('data/coarse-grained-all-words/senseNumberDict.pk', 'rb') as f:
                self._senseNum = pk.load(f)
        except Exception:
            with open('data/coarse-grained-all-words/senseNumberDict.pk3', 'rb') as f:
                self._senseNum = pk.load(f)

        if file is not None:
            self.parse(file)

    def _parseLineThread(self, wordsList):
        for word in wordsList:
            if word != None and word != '' and not is_number(word) and self.mutex.acquire(1):
                word = word.strip()
                word = word.lower()
                self.totalWordCount += 1

                if word in self._vocab.keys():
                    self._vocab[word].count += 1
                else:
                    if word in self._senseNum.keys():
                        self.totalSenseCount += self._senseNum[word]
                        self._vocab[word] = Word(word, sNum=self._senseNum[word])
                    else:
                        self.totalSenseCount += 1
                        self._vocab[word] = Word(word)

                    # self._idx2word.append(self._vocab[word])
                    self.size += 1

                    if self.size % 100 == 0:
                        sys.stdout.write('\r\t\t\t\t\t\t\tAdded %d words totally using %i threads.' % (self.size, threading.activeCount() - 1))
                        sys.stdout.flush()
                self.mutex.release()


    def parse(self, file, maxThreadNum=500, buffer=200000, parseUnitLength=1000):
        if os.path.isfile(file):
            self.corpus = file
            tp = ThreadPool(10)
            self.mutex = threading.Lock()

            with open(file) as f:
                for stc in fetchSentences(f, buffer, parseUnitLength):
                    requests = makeRequests(self._parseLineThread, [stc])

                    for req in requests:
                        tp.putRequest(req)

            print('\nRead Finished, Waiting...')
            tp.createWorkers(maxThreadNum - 10)
            tp.wait()

            print('\nParse Finished.')

            self.reduce()
        else:
            raise NotAFileException(file)


    def reduce(self):
        tmp = 0
        toBeDel = []

        for i in self._vocab:
            w = self._vocab[i]

            if w.count <= opt.minCount:
                toBeDel.append(i)
            else:
                w.index = tmp
                self._idx2word.append(w)
                tmp += 1

        self.size = tmp

        for i in toBeDel:
            del(self._vocab[i])

        print('Reduce Finished. %d Words Encountered.' % self.size)


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
                with open(file, 'rb') as f:
                    print('Loading vocab from file:', file)
                    data = pk.load(f)

                    self.totalSenseCount = data['tsc']
                    self.totalWordCount = data['twc']

                    for i in tqdm(data['words']):
                        w = Word(i[0], i[3], i[1], i[2])
                        w.initSenses()
                        self._idx2word.append(w)
                        self._vocab[i[0]] = self._idx2word[-1]

                    self.size = len(data['words'])
                    print('Vocab load finished.')

                    # self.initAllSenses()
                    return True
        except:
            return False


    def initAllSenses(self):
        print('Initializing all senses.')
        for word in tqdm(self._idx2word):
            word.initSenses()
        # self._idx2word[0].initSenses()
        # tf.variables_initializer([self._idx2word[0].means]).run(session=tf.Session())
        # print(tf.Session().run(self._idx2word[0].means))
        print('Finished initializing senses.')


    def saveEmbeddings(self, file, buffer=1000000):
        print('Saving embeddings to file:', file)
        with open(file, 'w', buffer) as f, tf.Session() as sess:
            for word in self._idx2word:
                means = sess.run(word.means)
                sigmas = sess.run(word.sigmas)

                f.write(str({'w': '"' + word.token + '"', 'm': means, 's': sigmas}) + '\n')
        print('Finished saving.')
