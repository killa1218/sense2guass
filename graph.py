# coding=utf8

import tensorflow as tf
from utils.distance import diagKL
from options import Options as opt


def batchSentenceLossGraph(vocabulary):
    senseIdxPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None, opt.sentenceLength])

    l = []
    for i in range(opt.sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i])
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i])

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                l.append(diagKL(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i - offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i - offset])))
            if i + offset < opt.sentenceLength:
                l.append(diagKL(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i + offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i + offset])))

    return tf.add_n(l), (senseIdxPlaceholder)


def windowLossGraph(vocabulary):
    mid = tf.placeholder(dtype=tf.int32, name='mid')
    others = tf.placeholder(dtype=tf.int32, name='others')

    midMean = tf.nn.embedding_lookup(vocabulary.means, mid)
    midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid)
    l = []

    for i in range(opt.windowSize * 2):
        l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, others[i]), tf.nn.embedding_lookup(vocabulary.sigmas, others[i]), midMean, midSigma))
        l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, others[opt.windowSize * 2 - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, others[opt.windowSize * 2 - i - 1]), midMean, midSigma))

    return tf.add_n(l), (mid, others)
