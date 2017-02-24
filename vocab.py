#!/usr/bin/python
# coding=utf8

from __future__ import print_function

import os
import pickle as pk
import sys
import multiprocessing
import tensorflow as tf

from tqdm import *
from word import Word
from options import Options as opt
from utils.common import is_number

curDir = os.path.dirname(os.path.abspath(__file__))

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
            with open(os.path.join(curDir, 'data/coarse-grained-all-words/senseNumberDict.pk'), 'rb') as f:
                self._senseNum = pk.load(f)
        except Exception:
            with open(os.path.join(curDir, 'data/coarse-grained-all-words/senseNumberDict.pk3'), 'rb') as f:
                self._senseNum = pk.load(f)

        if file is not None:
            self.parse(file)


    def _parseThread(self, filePath, start, end, chunkSize = 1048576):
        with open(filePath, 'r') as f:
            f.seek(start)
            d = {}

            while f.read(1) != ' ':
                pass

            start = f.tell()
            while start < end:
                if end - start > chunkSize:
                    chunk = f.read(chunkSize)
                else:
                    chunk = f.read(end - start)

                c = f.read(1)
                while c and c != ' ' and c != '\n' and c != '\t':
                    chunk += c
                    c = f.read(1)

                start = f.tell()

                for i in chunk.split(' '):
                    if i in d.keys():
                        d[i] += 1
                    else:
                        d[i] = 1

            return d


    def _helper(self, arg):
        return self._parseThread(*arg)


    def parse(self, file, buffer = 1048576, chunkNum = multiprocessing.cpu_count()):
        import math
        opt.minCount = 5
        pool = multiprocessing.Pool()
        fileSize = os.path.getsize(file)
        chunkSize = math.ceil(fileSize / chunkNum)
        task = []
        d = {}

        for i in range(chunkNum):
            start = i * chunkSize
            end = start + chunkSize if start + chunkSize < fileSize else fileSize
            task.append((file, start, end, buffer))

        dList = pool.imap_unordered(self._helper, task)

        for i in dList:
            for j in i:
                if not is_number(j):
                    if j in d.keys():
                        d[j] += i[j]
                    else:
                        d[j] = i[j]

        for i in d:
            if d[i] > opt.minCount:
                if i in self._vocab.keys():
                    self._vocab[i].count += d[i]
                else:
                    if i in self._senseNum.keys():
                        self._vocab[i] = Word(i, c=d[i], sNum=self._senseNum[i] if self._senseNum[i] < opt.maxSensePerWord else opt.maxSensePerWord)
                        self._vocab[i].senseStart = self.totalSenseCount
                        self.totalSenseCount += self._senseNum[i] if self._senseNum[i] < opt.maxSensePerWord else opt.maxSensePerWord
                    else:
                        self._vocab[i] = Word(i, c=d[i])
                        self._vocab[i].senseStart = self.totalSenseCount
                        self.totalSenseCount += 1

                    self.size += 1
                    self._vocab[i].index = len(self._idx2word)
                    self._idx2word.append(self._vocab[i])

                    if self.size % 100 == 0:
                        sys.stdout.write('\r%d words found, %d words encountered using %i threads.' % (len(d), self.size, multiprocessing.cpu_count()))
                        sys.stdout.flush()
        print('')
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


    def getWordBySenseId(self, sId):
        wId = self.size - 1 if sId > self.size - 1 else sId
        while wId > 0 and self._idx2word[wId].senseStart > sId:
            wId -= 1

        return self._idx2word[wId]


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
        '''
        {
            words: [
                (String, Integer, Integer, Integer, Integer),
                ...
            ]
            twc: Integer,
            tsc: Integer,
            means: [[Double, ...], ...],
            sigmas: [[Double, ...], ...]
        }
        '''
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
                    print('Vocab load finished. %d words and %d senses are encountered' % (self.totalWordCount, self.totalSenseCount))

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

if __name__ == '__main__':
    vocab = Vocab('data/text8')
