# coding=utf8

import tensorflow as tf
from options import Options as opt

if opt.energy == 'KL':
    print("Using KL Energy")
    from utils.distance import diagKL as dist
elif opt.energy == 'CE':
    print("Using CE Energy")
    from utils.distance import diagCE as dist
elif opt.energy == 'IP':
    print("Using IP Energy")
    from utils.distance import meanDist as dist
else:
    print("Using EL Energy")
    from utils.distance import diagEL as dist


def batchSentenceLossGraph(vocabulary, sentenceLength=opt.sentenceLength):
    senseIdxPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None, opt.sentenceLength], name="Observation")

    l = []
    for i in range(sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i], name="midMean")
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i], name="midSigma") if opt.covarShape != 'none' else None

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                l.append(
                    dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i - offset], name="outputSigma-" + str(i) + "_" + str(i - offset)))
                    if opt.covarShape != 'none' else
                    dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), None)
                )
            if i + offset < sentenceLength:
                l.append(
                    dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i + offset],  name = "outputSigma-" + str(i) + "_" + str(i + offset)))
                    if opt.covarShape != 'none' else
                    dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), None)
                )

    return tf.add_n(l, name = "ADD_ALL"), senseIdxPlaceholder, l

def batchNCELossGraph(vocabulary, sentenceLength=opt.sentenceLength):
    senseIdxPlaceholder = tf.placeholder(dtype=tf.int32, shape=[opt.batchSize, opt.sentenceLength], name="Observation")
    negSamples = tf.placeholder(dtype = tf.int32, shape = [opt.batchSize, opt.sentenceLength, opt.negative], name = "Negative_Samples")

    posList = []
    negList = []
    for i in range(sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i], name="midMean")
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i], name="midSigma") if opt.covarShape != 'none' else None
        negMeans = tf.nn.embedding_lookup(vocabulary.means, negSamples[:, i, :], name="negMeans")
        negSigmas = tf.nn.embedding_lookup(vocabulary.sigmas, negSamples[:, i, :], name="negSigmas") if opt.covarShape != 'none' else None

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                tmpNegList = []
                posList.append(
                    dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i - offset], name="outputSigma-" + str(i) + "_" + str(i - offset)))
                    if opt.covarShape != 'none' else
                    dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), None)
                )

                for j in range(opt.negative):
                    tmpNegList.append(
                        dist(negMeans[:, j], negSigmas[:, j],
                             tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset],
                                                    name = "outputMean-" + str(i) + "_" + str(i - offset)),
                             tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i - offset],
                                                    name = "outputSigma-" + str(i) + "_" + str(i - offset)))
                        if opt.covarShape != 'none' else
                        dist(negMeans[:, j], None,
                             tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset],
                                                    name = "outputMean-" + str(i) + "_" + str(i - offset)), None)
                    )

                negList.append(tf.add_n(tmpNegList) / opt.negative)
            if i + offset < sentenceLength:
                tmpNegList = []
                posList.append(
                    dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i + offset],  name = "outputSigma-" + str(i) + "_" + str(i + offset)))
                    if opt.covarShape != 'none' else
                    dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), None)
                )

                for j in range(opt.negative):
                    tmpNegList.append(
                        dist(negMeans[:, j], negSigmas[:, j],
                             tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset],
                                                    name = "outputMean-" + str(i) + "_" + str(i + offset)),
                             tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i + offset],
                                                    name = "outputSigma-" + str(i) + "_" + str(i + offset)))
                        if opt.covarShape != 'none' else
                        dist(negMeans[:, j], None,
                             tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset],
                                                    name = "outputMean-" + str(i) + "_" + str(i + offset)), None)
                    )

                negList.append(tf.add_n(tmpNegList) / opt.negative)

    posLoss = tf.add_n(posList, name="Positive_Loss")
    negLoss = tf.add_n(negList, name="Negative_Loss")
    return posLoss - negLoss, posLoss, negLoss, senseIdxPlaceholder, negSamples

def windowLossGraph(vocabulary):
    window = tf.placeholder(dtype = tf.int32, shape = [None, opt.windowSize * 2 + 1])

    midMean = tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize])
    midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, window[:, opt.windowSize]) if opt.covarShape != 'none' else None
    l = []

    for i in range(opt.windowSize * 2):
        l.append(
            dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, window[:, i]), tf.nn.embedding_lookup(vocabulary.sigmas, window[:, i]))
            if opt.covarShape != 'none' else
            dist(midMean, None, tf.nn.embedding_lookup(vocabulary.means, window[:, i]), None)
        )
        l.append(
            dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize * 2 - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, window[:, opt.windowSize * 2 - i - 1]))
            if opt.covarShape != 'none' else
            dist(midMean, None, tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize * 2 - i - 1]), None)
        )

    return tf.add_n(l), window


def negativeLossGraph(vocabulary):
    mid = tf.placeholder(dtype=tf.int32, name='mid')
    negSamples = tf.placeholder(dtype=tf.int32, name='others', shape=[None, opt.sentenceLength, opt.negative])

    l = []
    for i in range(opt.sentenceLength):
        midMean = tf.nn.embedding_lookup(vocabulary.means, mid[:, i])
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid[:, i]) if opt.covarShape != 'none' else None

        negSample = negSamples[:, i, :]

        for j in range(opt.negative):
            l.append(
                dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, negSample[:, j]), tf.nn.embedding_lookup(vocabulary.outputSigmas, negSample[:, j]))
                if opt.covarShape != 'none' else
                dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, negSample[:, j]), None)
            )

    return tf.add_n(l), (mid, negSamples)

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

