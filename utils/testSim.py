# coding=utf8

import sys
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import windowLossGraph
from e_step.cinference import batchDPInference
from vocab import Vocab
# from utils.distance import diagKL as dist
from utils.distance import diagEL as dist
from options import Options as opt


data = None
vocab = None

result = None
scoreList = None

date = '0407'
condition = ''

# with open('../data/SCWS/testData.pk3', 'rb') as f:
with open('/mnt/dataset/sense2gauss/data/SCWS/testData.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('/mnt/dataset/sense2gauss/data/gauss.EL.' + date + '_w3_b50_m100' + condition + '.pkl3')

with tf.Session() as sess:
    windowLossGraph, window = windowLossGraph(vocab)

    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    minValue = tf.argmin(distance, 0)
    maxValue = tf.argmax(distance, 0)
    labelPlaceholder = tf.placeholder(dtype=tf.float64)
    difference = 10 - labelPlaceholder - distance

    tf.global_variables_initializer().run()
    wordPairList = []
    scoreList = []

    for i in data:
        stcW = []
        word1 = i['w1']
        if vocab.getWord(word1):
            stcW.append(vocab.getWord(word1))
            try:
                w1sIdx = i['c1'].index(word1)
            except:
                print(i)
            j = 1

            while len(stcW) < opt.sentenceLength:
                if len(stcW) < opt.sentenceLength and w1sIdx - j >= 0 and vocab.getWord(i['c1'][w1sIdx - j]):
                    stcW.insert(0, vocab.getWord(i['c1'][w1sIdx - j]))
                if len(stcW) < opt.sentenceLength and w1sIdx + j < len(i['c1']) and vocab.getWord(i['c1'][w1sIdx + j]):
                    stcW.append(vocab.getWord(i['c1'][w1sIdx + j]))
                j += 1

            for k in range(len(stcW)):
                if stcW[k].token == word1:
                    w1sIdx = k
                    break

            assign1 = batchDPInference([stcW], sess, windowLossGraph, window)

            stcW = []
            word2 = i['w2']
            if vocab.getWord(word2):
                stcW.append(vocab.getWord(word2))
                w2sIdx = i['c2'].index(word2)
                j = 1

                while len(stcW) < opt.sentenceLength:
                    if len(stcW) < opt.sentenceLength and w2sIdx - j >= 0 and vocab.getWord(i['c2'][w2sIdx - j]):
                        stcW.insert(0, vocab.getWord(i['c2'][w2sIdx - j]))
                    if len(stcW) < opt.sentenceLength and w2sIdx + j < len(i['c2']) and vocab.getWord(i['c2'][w2sIdx + j]):
                        stcW.append(vocab.getWord(i['c2'][w2sIdx + j]))
                    j += 1

                for k in range(len(stcW)):
                    if stcW[k].token == word2:
                        w2sIdx = k
                        break

                assign2 = batchDPInference([stcW], sess, windowLossGraph, window)

                wordPairList.append([assign1[0][w1sIdx], assign2[0][w2sIdx]])
                scoreList.append(i['r'])

    result = sess.run(distance, feed_dict={sensePlaceholder: wordPairList})
    min = result[sess.run(minValue, feed_dict={sensePlaceholder: wordPairList})]
    max = result[sess.run(maxValue, feed_dict={sensePlaceholder: wordPairList})]
    idx = sess.run(tf.argmin(distance, 0), feed_dict={sensePlaceholder: wordPairList})

    print('Data size:', len(data), 'Data covered:', len(wordPairList), 'Recall:', float(len(wordPairList)) / len(data))
    print('minValue:', min)
    print('maxValue:', max)

with open('../data/SCWS/ELResult_' + date + condition + '.txt', 'w') as f:
    for i in range(len(wordPairList)):
        f.write(vocab.getWordBySenseId(wordPairList[i][0]).token)
        f.write('  ')
        f.write(vocab.getWordBySenseId(wordPairList[i][1]).token)
        f.write('\t')
        f.write(str(result[i]))
        f.write('\t')
        f.write(str(scoreList[i]))
        f.write('\n')
