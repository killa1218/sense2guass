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
        self.means = None
        self.sigmas = None

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
                        self.totalSenseCount += self._senseNum[word] if self._senseNum[word] < opt.maxSensePerWord else opt.maxSensePerWord
                        self._vocab[word] = Word(word, sNum=self._senseNum[word] if self._senseNum[word] < opt.maxSensePerWord else opt.maxSensePerWord)
                    else:
                        self.totalSenseCount += 1
                        self._vocab[word] = Word(word)

                    self.size += 1

                    if self.size % 100 == 0:
                        sys.stdout.write('\r\t\t\t\t\t\t\tAdded %d words totally using %i threads.' % (self.size, threading.activeCount() - 1))
                        sys.stdout.flush()
                self.mutex.release()


    def parse(self, file, maxThreadNum=2, buffer=200000, parseUnitLength=1000):
        if os.path.isfile(file):
            self.corpus = file
            tp = ThreadPool(2)
            self.mutex = threading.Lock()

            with open(file) as f:
                for stc in fetchSentences(f, buffer, parseUnitLength):
                    requests = makeRequests(self._parseLineThread, [stc])

                    for req in requests:
                        tp.putRequest(req)

            print('\nRead Finished, Waiting...')
            tp.createWorkers(maxThreadNum - 2)
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
            w = self._vocab[i]

            self.totalWordCount -= w.count
            self.totalSenseCount -= w.senseNum
            del(self._vocab[i])

        tmp = 0
        for i in self._idx2word:
            i.senseStart = tmp
            tmp += i.senseNum

        assert tmp == self.totalSenseCount
        assert len(self._idx2word) == len(self._vocab)
        assert len(self._idx2word) == self.size

        print('Reduce Finished. %d Words Encountered.' % self.size)

        self.initAllSenses()


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


    def saveVocabWithEmbeddings(self, file, sess):
        l = []
        try:
            for i in self._idx2word:
                l.append((i.token, i.senseNum, i.count, i.index, i.senseStart))
            with open(file, 'wb') as f:
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount, 'means': sess.run(self.means), 'sigmas': sess.run(self.sigmas)}, f)
                return True
        except Exception:
            tf.global_variables_initializer().run(session=sess)
            with open(file, 'wb') as f:
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount, 'means': sess.run(self.means), 'sigmas': sess.run(self.sigmas)}, f)
                return True


    def saveVocab(self, file):
        l = []
        try:
            for i in self._idx2word:
                l.append((i.token, i.senseNum, i.count, i.index, i.senseStart))
            with open(file, 'wb') as f:
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount}, f)
                return True
        except Exception:
            return True


    def load(self, file):
        try:
            if os.path.isfile(file):
                with open(file, 'rb') as f:
                    print('Loading vocab from file:', file)
                    data = pk.load(f)
                    curSenseCount = 0

                    self.totalSenseCount = data['tsc']
                    self.totalWordCount = data['twc']

                    try:
                        self.means = tf.Variable(data['means'], dtype=tf.float64)
                        self.sigmas = tf.Variable(data['sigmas'], dtype=tf.float64)
                    except KeyError:
                        print('Using old style vocab file.')

                    for i in tqdm(data['words']):
                        senseStart = None

                        try:
                            senseStart = i[4]
                        except IndexError:
                            senseStart = curSenseCount

                        w = Word(i[0], i[3], i[1] if i[1] < opt.maxSensePerWord else opt.maxSensePerWord, i[2], senseStart)
                        curSenseCount += i[1]
                        self._idx2word.append(w)
                        self._vocab[i[0]] = self._idx2word[-1]

                    self.size = len(data['words'])
                    print('Vocab load finished.')

                    if not self.means or not self.sigmas:
                        self.initAllSenses()
                    return True
        except Exception:
            print(Exception)
            return False


    def initAllSenses(self):
        print('Initializing all senses.')

        sNum = self.totalSenseCount
        eSize = opt.embSize
        iWidth = opt.initWidth

        self.means = tf.Variable(
            tf.random_uniform(
                [sNum, eSize],
                -iWidth,
                iWidth,
                dtype=tf.float64
            ),
            dtype=tf.float64,
            name="means"
        )

        if opt.covarShape == 'normal':
            self.sigmas = tf.clip_by_value(tf.Variable(
                tf.random_uniform(
                    [sNum, eSize, eSize],
                    0,
                    iWidth,
                    dtype=tf.float64
                ),
                dtype=tf.float64,
                name="sigmas"
            ), 0.01, float('inf'))
        elif opt.covarShape == 'diagnal':
            self.sigmas = tf.clip_by_value(tf.Variable(
                tf.random_uniform(
                    [sNum, eSize],
                    0,
                    iWidth,
                    dtype=tf.float64
                ),
                dtype=tf.float64,
                name="sigmas"
            ), 0.01, float('inf'))
        else:
            self.sigmas = None

        for w in self._idx2word:
            w.setMeans(self.means)
            w.setSigmas(self.sigmas)

        print('Finished initializing senses.')


    def saveEmbeddings(self, file, buffer=1000000):
        print('Saving embeddings to file:', file)
        with open(file, 'w', buffer) as f, tf.Session() as sess:
            for word in self._idx2word:
                means = sess.run(word.means)
                sigmas = sess.run(word.sigmas)

                f.write(str({'w': '"' + word.token + '"', 'm': means, 's': sigmas}) + '\n')
        print('Finished saving.')
