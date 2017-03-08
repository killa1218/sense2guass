# coding=utf8

import tensorflow as tf
from options import Options as opt

if opt.EL:
    from utils.distance import diagEL as dist
else:
    from utils.distance import diagKL as dist


def batchSentenceLossGraph(vocabulary, sentenceLength=opt.sentenceLength):
    senseIdxPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None, opt.sentenceLength])

    l = []
    for i in range(sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i])
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i])

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                l.append(dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i - offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i - offset])))
            if i + offset < sentenceLength:
                l.append(dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i + offset]), tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i + offset])))

    return tf.add_n(l), (senseIdxPlaceholder)


# def batchInferenceGraph(vocabulary, sentenceLength = opt.sentenceLength):
#     batchSenseIdxPlaceholder = tf.placeholder(dtype = tf.int32, shape = [opt.batchSize, None, sentenceLength])
#
#     res = []
#     for j in range(opt.batchSize):
#         l = []
#         for i in range(sentenceLength):
#             midMean = tf.nn.embedding_lookup(vocabulary.means, batchSenseIdxPlaceholder[j, :, i])
#             midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, batchSenseIdxPlaceholder[j, :, i])
#
#             for offset in range(1, opt.windowSize + 1):
#                 if i - offset > -1:
#                     l.append(dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, batchSenseIdxPlaceholder[j, :, i - offset]), tf.nn.embedding_lookup(vocabulary.sigmas, batchSenseIdxPlaceholder[j, :, i - offset])))
#                 if i + offset < sentenceLength:
#                     l.append(dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, batchSenseIdxPlaceholder[j, :, i + offset]), tf.nn.embedding_lookup(vocabulary.sigmas, batchSenseIdxPlaceholder[j, :, i + offset])))
#
#         res.append(tf.argmin(tf.add_n(l)))
#
#     return tf.stack(res), batchSenseIdxPlaceholder

# def windowLossGraph(vocabulary):
#     mid = tf.placeholder(dtype=tf.int32, name='mid')
#     others = tf.placeholder(dtype=tf.int32, name='others', shape=[None, opt.windowSize * 2])
#
#     midMean = tf.nn.embedding_lookup(vocabulary.means, mid)
#     midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid)
#     l = []
#
#     for i in range(opt.windowSize * 2):
#         l.append(dist(tf.nn.embedding_lookup(vocabulary.means, others[:, i]), tf.nn.embedding_lookup(vocabulary.sigmas, others[:, i]), midMean, midSigma))
#         l.append(dist(tf.nn.embedding_lookup(vocabulary.means, others[:, opt.windowSize * 2 - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, others[:, opt.windowSize * 2 - i - 1]), midMean, midSigma))
#
#     return tf.add_n(l), (mid, others)


def windowLossGraph(vocabulary):
    window = tf.placeholder(dtype = tf.int32, shape = [None, opt.windowSize * 2 + 1])

    midMean = tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize])
    midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, window[:, opt.windowSize])
    l = []

    for i in range(opt.windowSize * 2):
        l.append(dist(tf.nn.embedding_lookup(vocabulary.means, window[:, i]), tf.nn.embedding_lookup(vocabulary.sigmas, window[:, i]), midMean, midSigma))
        l.append(dist(tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize * 2 - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, window[:, opt.windowSize * 2 - i - 1]), midMean, midSigma))

    return tf.add_n(l), window


def negativeLossGraph(vocabulary):
    mid = tf.placeholder(dtype=tf.int32, name='mid')
    negSamples = tf.placeholder(dtype=tf.int32, name='others', shape=[None, opt.sentenceLength, opt.negative])

    l = []
    for i in range(opt.sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, mid[:, i])
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid[:, i])
        negSample = negSamples[:, i]

        for j in range(opt.negative):
            l.append(dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, negSample[:, j]), tf.nn.embedding_lookup(vocabulary.sigmas, negSample[:, j])))

    return tf.add_n(l), (mid, negSamples)
