# coding=utf8

from __future__ import print_function
from __future__ import division

from options import Options as opt
import tensorflow as tf
# from tensorflow.nn import relu as act
# from tensorflow import sigmoid as act
# from tensorflow import tanh as act
from utils.distance import dist
import time

import random
import math

act = tf.nn.relu

def skipGramWindowLoss(stc, sLabel, mid):
    start = time.time()
    l = []

    for i in range(1, opt.windowSize + 1):
        if mid - i > -1:
            l.append(
                dist(stc[mid], sLabel[mid], stc[mid - i], sLabel[mid - i])
            )
        if mid + i < len(stc):
            l.append(
                dist(stc[mid], sLabel[mid], stc[mid + i], sLabel[mid + i])
            )

    res = tf.clip_by_value(tf.add_n(l), tf.float64.min, tf.float64.max)

    end = time.time()
    print('skipGramWindowLoss time:', end - start)
    return res

def skipGramWindowKLLossGraph():
    from utils.distance import diagKL
    global vocabulary

    mid = tf.placeholder(dtype=tf.int32, name='mid')
    other = tf.placeholder(dtype=tf.int32, name='other')

    midMean = tf.nn.embedding_lookup(vocabulary.means, mid)
    midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid)
    l = []

    for i in range(opt.windowSize):
        l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, other[i]), tf.nn.embedding_lookup(vocabulary.sigmas, other[i]), midMean, midSigma))
        l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, other[opt.windowSize - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, other[opt.windowSize - i - 1]), midMean, midSigma))

    res = tf.clip_by_value(tf.add_n(l), tf.float64.min, tf.float64.max)

    return res


def skipGramSepWindowLoss(stc, sLabel, mid):
    res = []

    for i in range(1, opt.windowSize + 1):
        if mid - i > -1:
            res.insert(0, act(dist((stc[mid], sLabel[mid]), (stc[mid - i], sLabel[mid - i])), name="loss-ActivationFunction"))
        if mid + i < len(stc):
            res.append(act(dist((stc[mid], sLabel[mid]), (stc[mid + i], sLabel[mid + i])), name="loss-ActivationFunction"))

    return res


def skipGramLoss(stc, sLabel):
    '''Return the Tensorflow graph of skip-gram score of the sentence, the sentence is an array of Word()'''
    # totalSum = tf.constant(0., dtype=tf.float64)
    l = []

    for i in range(len(stc)):
        # tmpSum = tf.constant(0., dtype=tf.float64)

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                # tmpSum += act(dist(stc[i], sLabel[i], stc[i - offset], sLabel[i - offset]), name="loss-ActivationFunction")
                l.append(act(dist(stc[i], sLabel[i], stc[i - offset], sLabel[i - offset]), name="loss-ActivationFunction"))
            if i + offset < len(stc):
                # tmpSum += act(dist(stc[i], sLabel[i], stc[i + offset], sLabel[i + offset]), name="loss-ActivationFunction")
                l.append(act(dist(stc[i], sLabel[i], stc[i + offset], sLabel[i + offset]), name="loss-ActivationFunction"))

        # totalSum += tmpSum / (((i + opt.windowSize) if i + opt.windowSize < len(stc) else (len(stc) - 1)) - ((i - opt.windowSize) if i - opt.windowSize > -1 else 0))   # Full window
        # totalSum += tmpSum / ((i if i == 0 else i) if i < opt.windowSize else opt.windowSize)   # Half window

    return tf.add_n(l)


def skipGramKLLossGraph():
    '''Return the Tensorflow graph of skip-gram score of the sentence, the sentence is an array of Word()'''
    from utils.distance import diagKL
    global vocabulary

    senseIdx = tf.placeholder(dtype=tf.int32, shape=[opt.sentenceLength])
    l = []

    for i in range(opt.sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdx[i])
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdx[i])
        l = []

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                l.append(diagKL(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdx[i - offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdx[i - offset])))
            if i + offset < opt.sentenceLength:
                l.append(diagKL(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdx[i + offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdx[i + offset])))

    res = tf.clip_by_value(tf.add_n(l), tf.float64.min, tf.float64.max)

    return res

def avgSkipGramLoss(stc, sLabel):
    return skipGramLoss(stc, sLabel) / len(stc)


def skipGramNCELoss(stc, sLabel, vocab):
    '''Return the Tensorflow graph of skip-gram score with negative sampling of the sentence, the sentence is an array of Word()'''
    assert len(sLabel) == len(stc)

    totalSum = tf.constant(0., dtype=tf.float64)
    margin = tf.constant(opt.margin, dtype=tf.float64)

    for i in range(len(stc)):
        for offset in range(1, opt.windowSize + 1):
            try:
                if i - offset > -1:
                    sampleWord = stc[i]
                    while sampleWord is stc[i]:
                        sampleWord = vocab.getWord(random.randint(0, vocab.size - 1))

                    totalSum += tf.nn.relu(margin - act(dist(stc[i], sLabel[i], stc[i - offset], sLabel[i - offset])) + act(dist(stc[i], sLabel[i], sampleWord, 0)), name="loss-NCEMarginLoss")

                if i + offset < len(stc):
                    sampleWord = stc[i]
                    while sampleWord is stc[i]:
                        sampleWord = vocab.getWord(random.randint(0, vocab.size - 1))

                    totalSum += tf.nn.relu(margin - act(dist(stc[i], sLabel[i], stc[i + offset], sLabel[i + offset])) + act(dist(stc[i], sLabel[i], sampleWord, 0)), name="loss-NCEMarginLoss")
            except IndexError:
                print('ERROR:', 'len-', len(stc), 'i-', i, 'offset-', offset, 'stack-', sLabel)


        # totalSum += tmpSum / (((i + opt.windowSize) if i + opt.windowSize < len(stc) else (len(stc) - 1)) - ((i - opt.windowSize) if i - opt.windowSize > -1 else 0))   # Full window
        # totalSum += tmpSum / ((i if i == 0 else i) if i < opt.windowSize else opt.windowSize)   # Half window

    return tf.clip_by_value(totalSum, tf.float64.min, tf.float64.max)
