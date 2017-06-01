# coding=utf8

import tensorflow as tf
from options import Options as opt
from tqdm import tqdm
import time

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


# def batchSentenceLossGraph(vocabulary, sentenceLength=opt.sentenceLength):
#     senseIdxPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None, opt.sentenceLength], name="Observation")
#     inputMeans = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder)
#     inputSigmas = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder)
#     outputMeans = tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder)
#     outputSigmas = tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder)
#
#     l = []
#     for i in range(sentenceLength):
#         # midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i], name="midMean")
#         # midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i], name="midSigma") if opt.covarShape != 'none' else None
#         midMean = inputMeans[:, i]
#         midSigma = inputSigmas[:, i] if opt.covarShape != 'none' else None
#
#         for offset in range(1, opt.windowSize + 1):
#             if i - offset > -1:
#                 l.append(
#                     dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i - offset], name="outputSigma-" + str(i) + "_" + str(i - offset)))
#                     if opt.covarShape != 'none' else
#                     dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), None)
#                 )
#             if i + offset < sentenceLength:
#                 l.append(
#                     dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i + offset],  name = "outputSigma-" + str(i) + "_" + str(i + offset)))
#                     if opt.covarShape != 'none' else
#                     dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), None)
#                 )
#
#     return tf.add_n(l, name = "ADD_ALL"), senseIdxPlaceholder, l

def batchNCELossGraph(vocabulary, sentenceLength=opt.sentenceLength):
    with tf.name_scope("NCE_Loss_Graph"):
        senseIdxPlaceholder = tf.placeholder(dtype=tf.int64, shape=[opt.batchSize, opt.sentenceLength], name="Observation")
        tf.add_to_collection('POS_PHDR', senseIdxPlaceholder)

        inputMeans = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder) # [opt.batchSize, opt.sentenceLength, opt.embSize]
        inputSigmas = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder)
        outputMeans = tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder)
        outputSigmas = tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder)

        tf.add_to_collection('POS_VAR', inputMeans)
        tf.add_to_collection('POS_VAR', inputSigmas)
        tf.add_to_collection('POS_VAR', outputMeans)
        tf.add_to_collection('POS_VAR', outputSigmas)

        # negSamples = tf.placeholder(dtype = tf.int32, shape = [opt.batchSize, opt.sentenceLength, opt.negative], name = "Negative_Samples")
        negSamples = tf.reshape(tf.nn.fixed_unigram_candidate_sampler(true_classes = senseIdxPlaceholder,
                                                           num_true = opt.sentenceLength,
                                                           num_sampled = opt.negative * opt.sentenceLength * opt.batchSize,
                                                           unique = True,
                                                           range_max = vocabulary.totalSenseCount,
                                                           distortion = 0.75,
                                                           unigrams = vocabulary._sidx2count,
                                                           seed = time.time()
                                                           )[0], shape = [opt.batchSize, opt.sentenceLength, opt.negative], name = "Negative_Samples")
        negMeans = tf.nn.embedding_lookup(vocabulary.outputMeans, negSamples) # [opt.batchSize, opt.sentenceLength, opt.negative, opt.embSize]
        negSigmas = tf.nn.embedding_lookup(vocabulary.outputSigmas, negSamples)

        tf.add_to_collection('NEG_VAR', negMeans)
        tf.add_to_collection('NEG_VAR', negSigmas)

        negLoss = dist(tf.expand_dims(inputMeans, 2), tf.expand_dims(inputSigmas, 2), negMeans, negSigmas, pos = False, dim = 3) # [opt.batchSize, opt.sentenceLength, opt.negative]
        tf.add_to_collection('NEG_LOSS', negLoss)

        # posList = []
        # negList = []
        lossList = []
        for i in tqdm(range(sentenceLength)):
            # midMean = tf.nn.embedding_lookup(vocabulary.means, senseIdxPlaceholder[:, i], name="midMean")
            # midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, senseIdxPlaceholder[:, i], name="midSigma") if opt.covarShape != 'none' else None
            # wordNegMeans = tf.nn.embedding_lookup(vocabulary.means, negSamples[:, i, :], name="negMeans")
            # wordNegSigmas = tf.nn.embedding_lookup(vocabulary.sigmas, negSamples[:, i, :], name="negSigmas") if opt.covarShape != 'none' else None

            midMean = inputMeans[:, i] # [opt.batchSize, opt.embSize]
            midSigma = inputSigmas[:, i]

            neg = negLoss[:, i]

            for offset in range(1, opt.windowSize + 1):
                if i - offset > -1:
                    pos = tf.expand_dims(dist(midMean, midSigma, outputMeans[:, i - offset], outputSigmas[:, i - offset], pos = True), 1)
                    tf.add_to_collection('POS_LOSS', pos)
                    lossList.append(tf.nn.relu(opt.margin - neg + pos))
                    # posList.append(
                    #     # dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i - offset], name="outputSigma-" + str(i) + "_" + str(i - offset)))
                    #     tf.expand_dims(dist(midMean, midSigma, outputMeans[:, i - offset], outputSigmas[:, i - offset], pos = True), 1)
                    #     # if opt.covarShape != 'none' else
                    #     # dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i - offset], name = "outputMean-" + str(i) + "_" + str(i - offset)), None)
                    #     # dist(midMean, None, outputMeans[:, i - offset], None, pos = True)
                    # )

                if i + offset < sentenceLength:
                    pos = tf.expand_dims(dist(midMean, midSigma, outputMeans[:, i + offset], outputSigmas[:, i + offset], pos = True), 1)
                    tf.add_to_collection('POS_LOSS', pos)
                    lossList.append(tf.nn.relu(opt.margin - neg + pos))
                    # posList.append(
                    #     # dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), tf.nn.embedding_lookup(vocabulary.outputSigmas, senseIdxPlaceholder[:, i + offset],  name = "outputSigma-" + str(i) + "_" + str(i + offset)))
                    #     tf.expand_dims(dist(midMean, midSigma, outputMeans[:, i + offset], outputSigmas[:, i + offset], pos = True), 1)
                    #     # if opt.covarShape != 'none' else
                    #     # dist(midMean, None, outputMeans[:, i + offset], None, pos = True)
                    #     # dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, senseIdxPlaceholder[:, i + offset], name = "outputMean-" + str(i) + "_" + str(i + offset)), None)
                    # )

            # for j in range(opt.negative):
            #     negList.append(
            #         dist(midMean, midSigma, wordNegMeans[:, j], wordNegSigmas[:, j], pos = False)
            #         # if opt.covarShape != 'none' else
            #         # dist(midMean, None, wordNegMeans[:, j], None, pos = False)
            #     )

        # with tf.name_scope("Positive_Loss_Graph"):
        #     posLoss = tf.div(tf.add_n(posList), len(posList), name="Positive_Loss")
        # with tf.name_scope("Negative_Loss_Graph"):
        #     negLoss = tf.div(tf.add_n(negList), len(negList), name="Negative_Loss")
        # return posLoss - negLoss, posLoss, negLoss, senseIdxPlaceholder, negSamples

        # with tf.name_scope("Positive_Loss_Graph"):
        #     posLoss = posList
        # with tf.name_scope("Negative_Loss_Graph"):
        #     negLoss = negList
        # return posLoss, negLoss, senseIdxPlaceholder, negSamples
        return lossList

def windowLossGraph(vocabulary):
    with tf.name_scope("Window_Graph"):
        window = tf.placeholder(dtype = tf.int32, shape = [None, opt.windowSize * 2 + 1], name = "window_placeholder")
        tf.add_to_collection('WIN_PHDR', window)

        midMean = tf.nn.embedding_lookup(vocabulary.means, window[:, opt.windowSize], name = "Middle_Word_Mean")
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, window[:, opt.windowSize], name = "Middle_Word_Sigma") if opt.covarShape != 'none' else None
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


# def negativeLossGraph(vocabulary):
#     mid = tf.placeholder(dtype=tf.int32, name='mid')
#     negSamples = tf.placeholder(dtype=tf.int32, name='others', shape=[None, opt.sentenceLength, opt.negative])
#
#     l = []
#     for i in range(opt.sentenceLength):
#         midMean = tf.nn.embedding_lookup(vocabulary.means, mid[:, i])
#         midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid[:, i]) if opt.covarShape != 'none' else None
#
#         negSample = negSamples[:, i, :]
#
#         for j in range(opt.negative):
#             l.append(
#                 dist(midMean, midSigma, tf.nn.embedding_lookup(vocabulary.outputMeans, negSample[:, j]), tf.nn.embedding_lookup(vocabulary.outputSigmas, negSample[:, j]))
#                 if opt.covarShape != 'none' else
#                 dist(midMean, None, tf.nn.embedding_lookup(vocabulary.outputMeans, negSample[:, j]), None)
#             )
#
#     return tf.add_n(l), (mid, negSamples)

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

