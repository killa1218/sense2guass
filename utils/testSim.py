# coding=utf8

import sys
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import batchSentenceLossGraph
from e_step.inference import violentInference
from vocab import Vocab
from utils.distance import diagKL
from options import Options as opt


data = None
vocab = None

with open('../data/SCWS/testData.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('../data/vec.txt')

with tf.Session() as sess:
    lossGraph, placeholder = batchSentenceLossGraph(vocab)
    minLossIdxGraph = tf.argmin(lossGraph, 0)

    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = diagKL(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    labelPlaceholder = tf.placeholder(dtype=tf.float64)
    difference = 10 - labelPlaceholder - distance

    tf.global_variables_initializer().run()
    wordPairList = []
    scoreList = []
    # inferenceList = []

    for i in data:
        stcW = []
        word1 = i['w1']
        w1sIdx = i['c1'].index(word1)
        j = 1

        while len(stcW) < opt.sentenceLength:
            if w1sIdx - j >= 0 and vocab.getWord(i['c1'][w1sIdx - j]):
                stcW.append(vocab.getWord(i['c1'][w1sIdx - j]))
            if w1sIdx + j < len(i['c1']) and vocab.getWord(i['c1'][w1sIdx - j]):
                stcW.append(vocab.getWord(i['c1'][w1sIdx + j]))
        w1sIdx = stcW.index(word1)

        assign1 = violentInference(stcW, sess, minLossIdxGraph, placeholder)

        stcW = []
        word2 = i['w2']
        w2sIdx = i['c2'].index(word2)
        j = 1

        while len(stcW) < opt.sentenceLength:
            if w2sIdx - j >= 0 and vocab.getWord(i['c2'][w2sIdx - j]):
                stcW.append(vocab.getWord(i['c2'][w2sIdx - j]))
            if w2sIdx + j < len(i['c2']) and vocab.getWord(i['c2'][w2sIdx - j]):
                stcW.append(vocab.getWord(i['c2'][w2sIdx + j]))
        w2sIdx = stcW.index(word2)

        assign2 = violentInference(stcW, sess, minLossIdxGraph, placeholder)

        wordPairList.append([assign1[w1sIdx], assign2[w2sIdx]])
        scoreList.append(i['r'])

    print(sess.run(distance, feed_dict={sensePlaceholder: wordPairList}))

