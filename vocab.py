# coding=utf8

from __future__ import print_function

import os
import pickle as pk
import sys
import time
import multiprocessing
import tensorflow as tf

from tqdm import *
from word import Word
from options import Options as opt
from utils.common import is_number

curDir = os.path.dirname(os.path.abspath(__file__))
dataType = opt.dType

class Vocab(object):
    """ Vocabulary of the train corpus, used for embedding lookup and sense number lookup. """

    def __init__(self, file=None):
        self._wordSeparator = opt.wordSeparator
        self._vocab = {}
        self._idx2word = []
        self._sidx2count = []
        self.size = 0
        self.totalWordCount = 0
        self.totalSenseCount = 0
        self.means = None
        self.sigmas = None

        try:
            with open(os.path.join(curDir, 'data/coarse-grained-all-words/senseNumberDict.pkl'), 'rb') as f:
                self._senseNum = pk.load(f)
        except Exception:
            with open(os.path.join(curDir, 'data/coarse-grained-all-words/senseNumberDict.pkl3'), 'rb') as f:
                self._senseNum = pk.load(f)

        if file is not None:
            self.parse(file)


    def _parseThread(self, filePath, start, end, chunkSize = 524288):
        with open(filePath, 'r') as f:
            f.seek(start)
            d = {}
            s = start

            while f.read(1) != ' ':
                pass

            start = f.tell()
            st = time.time()
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

                readSize = float(f.tell() - s)
                if (int(readSize) // chunkSize) % 10 == 1:
                    t = time.time()
                    sys.stdout.write('\r%.3f%% parsed. Speed: %.3fMB/s' % (readSize * 100 / (end - s), readSize * multiprocessing.cpu_count() / (t - st) / 1000000))
                    sys.stdout.flush()

            return d


    def _helper(self, arg):
        return self._parseThread(*arg)


    def parse(self, file, buffer = 524288, chunkNum = multiprocessing.cpu_count()):
        import math
        pool = multiprocessing.Pool()
        fileSize = os.path.getsize(file)
        chunkSize = math.ceil(fileSize / chunkNum)
        task = []
        d = {}
        tokenNum = 0

        for i in range(chunkNum):
            start = i * chunkSize
            end = start + chunkSize if start + chunkSize < fileSize else fileSize
            task.append((file, start, end, buffer))

        for i in pool.map(self._helper, task):
            sys.stdout.write('\rMerging results from processes...')
            sys.stdout.flush()
            for j in i:
                if not is_number(j):
                    if j in d.keys():
                        d[j] += i[j]
                    else:
                        d[j] = i[j]

        for i in d:
            tokenNum += d[i]

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

                    word = self._vocab[i]
                    wordCount = word.count
                    for _ in range(word.senseNum):
                        self._sidx2count.append(int(wordCount / word.senseNum))

                    if self.size % 100 == 0:
                        sys.stdout.write('\rCorpus contains %d tokens, %d words found, %d words and %d senses encountered using %i processes.' % (tokenNum, len(d), self.size, self.totalSenseCount, multiprocessing.cpu_count()))
                        sys.stdout.flush()

        self.totalWordCount = len(d)
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
        return float(self.getWordCount(word)) / self.totalWordCount


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
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount, 'means': sess.run(self.means), 'sigmas': sess.run(self.sigmas) if self.sigmas != None else None}, f)
                return True
        except Exception:
            tf.global_variables_initializer().run(session=sess)
            with open(file, 'wb') as f:
                pk.dump({'words': l, 'twc': self.totalWordCount, 'tsc': self.totalSenseCount, 'means': sess.run(self.means), 'sigmas': sess.run(self.sigmas) if self.sigmas != None else None}, f)
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
                        self.means = tf.Variable(data['means'], dtype=dataType)
                        self.sigmas = tf.Variable(data['sigmas'], dtype=dataType) if data['sigmas'] is not None else None
                    except KeyError:
                        print('Using old style vocab file.')

                    for i in tqdm(data['words']):
                        senseStart = curSenseCount

                        w = Word(word = i[0], index  = i[3], sNum = i[1] if i[1] < opt.maxSensePerWord else opt.maxSensePerWord, c = i[2], sStart = senseStart)
                        curSenseCount += w.senseNum
                        self._idx2word.append(w)
                        self._vocab[i[0]] = self._idx2word[-1]

                        wordCount = w.count
                        for _ in range(w.senseNum):
                            self._sidx2count.append(int(wordCount / w.senseNum))

                    self.size = len(data['words'])
                    self.totalSenseCount = curSenseCount
                    print('Vocab load finished. %d words and %d senses are encountered' % (self.size, self.totalSenseCount))

                    if not self.means and not self.sigmas:
                        self.initAllSenses()
                    return True
        except Exception:
            print("VOCAB LOAD", Exception.message)
            return False


    def initAllSenses(self):
        print('Initializing all senses.')

        sNum = self.totalSenseCount
        eSize = opt.embSize

        if opt.energy == 'IP': # When use inner product, clip the length of means
            iWidth = opt.initWidth / opt.embSize
            with tf.name_scope("Word2Vec_Vector"):
                self.means = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        -iWidth,
                        iWidth,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="means"
                )

                self.outputMeans = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        -iWidth,
                        iWidth,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="outputMeans"
                )
        else:
            iWidth = opt.initWidth
            with tf.name_scope("Means"):
                self.means = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        -iWidth,
                        iWidth,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="means"
                )

                self.outputMeans = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        -iWidth,
                        iWidth,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="outputMeans"
                )

        if opt.covarShape == 'normal':
            self.sigmas = tf.clip_by_value(tf.Variable(
                tf.random_uniform(
                    [sNum, eSize, eSize],
                    0.4,
                    0.6,
                    dtype=dataType
                ),
                dtype=dataType,
                name="sigmas"
            ), 0.00001, float('inf'))
        elif opt.covarShape == 'diagnal':
            with tf.name_scope("Diagnal_Cov"):
                self.trainableSigmas = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        0.4,
                        0.6,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="sigmas"
                )
                self.sigmas = tf.clip_by_value(self.trainableSigmas, opt.covMin, opt.covMax)

                self.trainableOutputSigmas = tf.Variable(
                    tf.random_uniform(
                        [sNum, eSize],
                        0.4,
                        0.6,
                        dtype=dataType
                    ),
                    dtype=dataType,
                    name="outputSigmas"
                )
                self.outputSigmas = tf.clip_by_value(self.trainableOutputSigmas, opt.covMin, opt.covMax)
        else:
            self.sigmas = None

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
