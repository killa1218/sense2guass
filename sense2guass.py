#!/usr/local/bin/python3
# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random

# from six.moves import xrange    # pylint: disable=redefined-builtin


from tqdm import tqdm
from vocab import Vocab as V
from options import Options as opt
from utils.fileIO import fetchSentencesAsWords
import tensorflow as tf
import numpy as np

random.seed(time.time())

flags = tf.app.flags

flags.DEFINE_string("output", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train", None, "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("vocab", None, "The vocabulary file path.")
flags.DEFINE_string("save_vocab", None, "If not None, save the vocabulary to this path.")
flags.DEFINE_string("covariance", "diagnal", "Shape of covariance matrix, default is diagnal. Possible value is 'diagnal' or ")
flags.DEFINE_integer("size", 50, "The embedding dimension size. Default is 100.")
flags.DEFINE_integer("window", 3, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("negative", 5, "Negative samples per sense. Default is 5.")
flags.DEFINE_integer("iter", 10, "Number of iterations to train. Each iteration processes the training data once completely. Default is 15.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary. Default is 5.")
flags.DEFINE_integer("sentence_length", 20, "The length of one sentence.")
flags.DEFINE_integer("max_sense_per_word", 5, "The maximum number of one word.")
flags.DEFINE_integer("batch_size", 50, "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_float("alpha", 0.001, "Initial learning rate. Default is 0.001.")
flags.DEFINE_float("margin", 100, "Margin between positive and negative training pairs. Default is 100.")
# flags.DEFINE_boolean("gpu", False, "If true, use GPU instead of CPU.")
flags.DEFINE_string("energy", "EL", "What energy function is used, default is EL.")

FLAGS = flags.FLAGS

opt.savePath = FLAGS.output
opt.train = FLAGS.train
opt.vocab = FLAGS.vocab
opt.saveVocab = FLAGS.save_vocab
opt.covarShape = FLAGS.covariance
opt.embSize = FLAGS.size
opt.windowSize = FLAGS.window
opt.negative = FLAGS.negative
opt.iter = FLAGS.iter
opt.minCount = FLAGS.min_count
opt.sentenceLength = FLAGS.sentence_length
opt.maxSensePerWord = FLAGS.max_sense_per_word
opt.batchSize = FLAGS.batch_size
opt.alpha = FLAGS.alpha
opt.margin = FLAGS.margin
# opt.gpu = FLAGS.gpu
opt.energy = FLAGS.energy

vocabulary = None
e_step = True
m_step = True
opt.verboseTime = False

gradMin = -0.5
gradMax = 0.5

def main(_):
    """ Train a sense2guass model. """
    global vocabulary

    if not FLAGS.train or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # Config to make tensorflow not take up all the GPU memory
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        # optimizer = tf.train.AdagradOptimizer(opt.alpha)
        optimizer = tf.train.AdamOptimizer(opt.alpha)
        # optimizer = tf.train.GradientDescentOptimizer(opt.alpha)
        # optimizer = tf.train.AdadeltaOptimizer(opt.alpha)

        # Build vocabulary or load vocabulary from file
        if opt.vocab != None:
            vocabulary = V()
            vocabulary.load(opt.vocab)
        else:
            vocabulary = V(opt.train)
            vocabulary.initAllSenses()

            if opt.saveVocab:
                if vocabulary.saveVocab(opt.saveVocab):
                    print('Vocab saved at %s.' % opt.saveVocab)
                else:
                    print('Vocab save FAILED!')

##----------------- Build Window Loss Graph ------------------
        print('Building Window Loss Graph...')
        from graph import windowLossGraph
        windowLossGraph, window = windowLossGraph(vocabulary)
        print('Finished.')
##----------------- Build Window Loss Graph ------------------

        if m_step:
            # writer = tf.summary.FileWriter('summary/', sess.graph)

        ##----------------- Build Sentence Loss Graph And Optimizer ------------------
            # print('Building Sentence Loss Graph...')
            # from graph import batchSentenceLossGraph as lossGraph
            #
            # with tf.name_scope('pos_loss'):
            #     batchSentenceLossGraph, senseIdxPlaceholder, l = lossGraph(vocabulary)
            #
            # # tf.summary.tensor_summary("pos_loss", batchSentenceLossGraph)
            # # merged = tf.summary.merge_all()
            # # minLossIdxGraph = tf.argmin(batchSentenceLossGraph, 0)
            # avgBatchStcLoss = batchSentenceLossGraph / (opt.sentenceLength * opt.windowSize * 2 - (opt.windowSize + 1) * opt.windowSize)
            # reduceAvgLoss = tf.reduce_sum(avgBatchStcLoss) / opt.batchSize
            # print('Finished Building Sentence Loss Graph.')
        ##----------------- Build Sentence Loss Graph And Optimizer ------------------

        ##----------------------- Build Negative Loss Graph --------------------------
            # print('Building Negative Loss Graph...')
            # from graph import negativeLossGraph
            # negativeLossGraph, (mid, negSamples) = negativeLossGraph(vocabulary)
            # avgNegLoss = negativeLossGraph / opt.negative / opt.sentenceLength
            # reduceAvgNegLoss = tf.reduce_sum(avgNegLoss) / opt.batchSize
            # print('Finished Building Negative Loss Graph.')
        ##----------------------- Build Negative Loss Graph --------------------------

        ##---------------------------- Build NCE Loss --------------------------------
            print('Building NCE Loss...')
            from graph import batchNCELossGraph

            posLoss, negLoss, senseIdxPlaceholder, negSamples= batchNCELossGraph(vocabulary)
            posNum = len(posLoss)
            negNum = len(negLoss)
            avgPosLoss = tf.reduce_sum(posLoss) / posNum / opt.batchSize
            avgNegLoss = tf.reduce_sum(negLoss) / negNum / opt.batchSize
            avgNCELoss = avgNegLoss - avgPosLoss
            loss = tf.nn.relu(opt.margin - avgNCELoss)

            # true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels = tf.ones_like(posLoss), logits = posLoss)
            # sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels = tf.zeros_like(negLoss), logits = negLoss)
            # avgPosLoss = tf.reduce_sum(true_xent) / posNum / opt.batchSize
            # avgNegLoss = tf.reduce_sum(sampled_xent) / negNum / opt.batchSize
            # loss = avgPosLoss + avgNegLoss

            with tf.name_scope("Summary"):
                tf.summary.scalar("Positive Loss", avgPosLoss)
                tf.summary.scalar("Negative Loss", avgNegLoss)
                tf.summary.scalar("NCE Loss", loss)
                tf.summary.scalar("Average Mean Norm", tf.reduce_sum(tf.norm(vocabulary.means, axis = 1)) / vocabulary.totalSenseCount)
                tf.summary.scalar("Average Sigma Norm", tf.reduce_sum(tf.norm(vocabulary.sigmas, axis = 1)) / vocabulary.totalSenseCount)
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter('logEL/' + time.strftime("%m%d", time.localtime()), graph = sess.graph)
            print('Finished.')
            # reduceNCELoss = tf.reduce_sum(nceLossGraph)
            # avgNCELoss = reduceNCELoss / opt.batchSize
            regular = 0 # -tf.norm(vocabulary.sigmas, ord = 'euclidean') if opt.covarShape != 'none' else 0

            print('Building Optimizer...')
            # grad = optimizer.compute_gradients(loss + regular)
            # clipedGrad = [(tf.clip_by_value(g, gradMin, gradMax), var) for g, var in grad]
            # op = optimizer.apply_gradients(clipedGrad)
            op = optimizer.minimize(loss + regular)
            # # op = optimizer(avgBatchStcLoss)
            print('Finished.')
        ##---------------------------- Build NCE Loss --------------------------------

            # grad = tf.gradients(batchSentenceLossGraph, vocabulary.outputMeans) #, vocabulary.outputMeans[vocabulary.getWord('is').senseStart], vocabulary.outputSigmas[vocabulary.getWord('is').senseStart]])

##------------------------- Build Validate Graph -----------------------------
        # print('Building Validate Graph...')
        # from utils.distance import diagKL
        # w1Plchdr = tf.placeholder(dtype=tf.int32)
        # w2Plchdr = tf.placeholder(dtype=tf.int32)
        #
        # dist = diagKL(tf.nn.embedding_lookup(vocabulary.means, w1Plchdr), tf.nn.embedding_lookup(vocabulary.sigmas, w1Plchdr), tf.nn.embedding_lookup(vocabulary.means, w2Plchdr), tf.nn.embedding_lookup(vocabulary.sigmas, w2Plchdr))
        # print('Finished Building Validate Graph.')
##------------------------- Build Validate Graph -----------------------------

##------------------------- Build Argmin Graph -----------------------------
        # lossPlaceholder = tf.placeholder(dtype = tf.float64, shape = [None, 1])
        # argmin = tf.argmin(lossPlaceholder, 0)
##------------------------- Build Argmin Graph -----------------------------

        tf.global_variables_initializer().run(session=sess)
        # Train iteration
        print('Start training...\n')

        # print(vocabulary.means[vocabulary.getWord('is').senseStart].eval())
        # print(vocabulary.sigmas[vocabulary.getWord('is').senseStart].eval())
        # orgMeans = vocabulary.means.eval()
        # orgSigmas = vocabulary.sigmas.eval()
        # fi = open('grads', 'w')

        from e_step.cinference import batchDPInference
        # from e_step.inference import batchDPInference as pyinference
        # from multiprocessing import Pool
        # pool = Pool()

        step = 0

        for i in range(opt.iter):
            if os.path.isfile(opt.train):
                with open(opt.train) as f:
                    # negativeSamplesList = []
                    batchStcW = []

                    try:
                        if opt.verboseTime:
                            wT = 0
                            iT = 0
                            cT = 0
                            tIT = 0
                            count = 0

                        for stcW in fetchSentencesAsWords(f, vocabulary, 20000, opt.sentenceLength, verbose=False):
    ##----------------------------- Train Batch ------------------------------
                            if len(stcW) > opt.windowSize and len(stcW) == opt.sentenceLength:
                                batchStcW.append(stcW)
                                # negativeSampleList = []
                                #
                                # for a in range(opt.sentenceLength):
                                #     sampleTmp = random.sample(range(vocabulary.totalSenseCount), opt.negative) # TODO 这里可以优化
                                #     # while a in sampleTmp:
                                #     #     sampleTmp = random.sample(range(vocabulary.totalSenseCount), opt.negative)
                                #
                                #     negativeSampleList.append(sampleTmp)
                                # negativeSamplesList.append(negativeSampleList)


                                if len(batchStcW) == opt.batchSize:
                                    batchLossSenseIdxList = []
                                    negativeSamplesList = np.random.randint(vocabulary.totalSenseCount, size=(opt.batchSize, opt.sentenceLength, opt.negative))
    ##--------------------------------- Inference By Batch ----------------------------------
                                    start = time.time()
                                    if opt.maxSensePerWord == 1:
                                        for p in batchStcW:
                                            tmpList = []

                                            for q in p:
                                                tmpList.append(q.senseStart)

                                            batchLossSenseIdxList.append(tmpList)
                                    else:
                                        batchLossSenseIdxList, twT, tcT, tiT = batchDPInference(batchStcW, sess, windowLossGraph, window)

                                        if opt.verboseTime:
                                            tIT += time.time() - start
                                            wT += twT
                                            cT += tcT
                                            iT += tiT
                                            count += 1
                                            print("\n\nTotal Inference Time:", tIT / count, '\n')
                                            print("Get Windows Time Takes: %.2f%%" % (wT * 100 / tIT))
                                            print("Calculate Time Takes: %.2f%%" % (cT * 100 / tIT))
                                            print("DP Time Takes: %.2f%%" % (iT * 100 / tIT))
                                            print("Data Transfer Time Takes: %.2f%%" % ((tIT - iT - cT - wT) * 100 / tIT))
    ##--------------------------------- Inference By Batch ----------------------------------

                                    if m_step:
                                        # summ, posloss = sess.run([merged, reduceAvgLoss], feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        # writer.add_summary(summ, i)
                                        start = time.time()
                                        posloss = sess.run(avgPosLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        negloss = sess.run(avgNegLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        nceloss = sess.run(loss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        # print(sess.run(grad, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList}))
                                        sys.stdout.write('\rIter: %d/%d, POSLoss: %.8f, NEGLoss: %.8f, neg - pos: %.8f, NCELoss: %.8f, Progress: %.2f%%.' % (i + 1, opt.iter, posloss, negloss, negloss - posloss, nceloss, (float(f.tell()) * 100 / os.path.getsize(opt.train))))
                                        sys.stdout.flush()
                                        summary_writer.add_summary(summary_op.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList}), step)
                                        step += 10
                                        # print("Cal Loss Time:", time.time() - start)

                                        # if posloss < 0:
                                        #     print(sess.run(l, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))

                                        # if posloss > 1000:
                                        #     print('')
                                        #     print("ASSIGN:", batchLossSenseIdxList)
                                        #     energys = sess.run(l, feed_dict = {senseIdxPlaceholder: batchLossSenseIdxList, mid: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        #     print("ENERGYS:", energys)
                                        #     print("VARLS:", sess.run(varl, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                        #
                                        #     for ind in range(len(energys)):
                                        #         if energys[ind] > 1000:
                                        #             pair = sess.run(varl[i], feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, mid: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        #
                                        #             mm = tf.nn.embedding_lookup(vocabulary.means, pair[0])
                                        #             sigm = tf.nn.embedding_lookup(vocabulary.sigmas, pair[0])
                                        #             moth = tf.nn.embedding_lookup(vocabulary.outputMeans, pair[1])
                                        #             sigoth = tf.nn.embedding_lookup(vocabulary.outputSigmas, pair[1])
                                        #
                                        #             m = mm - moth
                                        #             sig = sigm + sigoth
                                        #
                                        #             from utils.distance import diagEL
                                        #
                                        #             print("ENERGY:", energys[ind])
                                        #             print("ENERGY REAL:", sess.run(diagEL(mm, sigm, moth, sigoth)))
                                        #             print("TRACE VALUE:", sess.run(tf.log(tf.reduce_prod(sig, 1))))
                                        #             print("SQUARE VALUE:", sess.run(tf.reduce_sum(tf.square(m) * tf.reciprocal(sig), 1)))
                                        #             print("SQUARE SUM:", sess.run(tf.reduce_sum(tf.square(m), 1)))
                                        #             print("SIGMA:", sess.run(sig))
                                        #             print("MEAN:", sess.run(m))

                                        # start = time.time()
                                        for _ in range(1):
                                            sess.run(op, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        if opt.verboseTime:
                                            print('OP Time:', time.time() - start)

                                        # print(batchStcW)
                                        # print("Input Embedding", vocabulary.means[vocabulary.getWord('without').senseStart].eval())
                                        # print("Input Embedding", vocabulary.sigmas[vocabulary.getWord('without').senseStart].eval())
                                        # print("Output Embedding", vocabulary.outputMeans[vocabulary.getWord('without').senseStart].eval())
                                        # print("Output Embedding", vocabulary.outputSigmas[vocabulary.getWord('without').senseStart].eval())
                                        # print("Gradient:", vocabulary.getWord('without').senseStart in sess.run(tf.gradients(avgBatchStcLoss, vocabulary.means), feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})[0].indices)
                                        # gr = sess.run(grad, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        # print(gr)
                                        # gr[67320]
                                        # fi.write(str(batchStcW))
                                        # fi.write('\n')
                                        # fi.write(str(batchLossSenseIdxList))
                                        # fi.write('\n')
                                        # fi.write(str(list(gr[0][0].values)).replace('\n', ''))
                                        # fi.write('\n')
                                        # fi.write(str(list(gr[0][0].indices)))
                                        # fi.write('\n')
                                        # fi.write('\n')
                                        # print('OK')

                                    batchStcW = []
                                    # negativeSamplesList = []
    ##----------------------------- Train Batch ------------------------------
                    except KeyboardInterrupt:
                        print("Canceled by user, save data?(y/N)")
                        ans = input()
                        if ans == 'y':
                            vocabulary.saveVocabWithEmbeddings(opt.savePath, sess)
                        return

                # print('is', vocabulary.getWord('is').senseCount, vocabulary.getWord('is').senseNum)
                # print('english', vocabulary.getWord('english').senseCount, vocabulary.getWord('english').senseNum)
                # print('latin', vocabulary.getWord('latin').senseCount, vocabulary.getWord('latin').senseNum)
                # print('victoria', vocabulary.getWord('victoria').senseCount, vocabulary.getWord('victoria').senseNum)
                # print('a', vocabulary.getWord('a').senseCount, vocabulary.getWord('a').senseNum)
                #

                # aftMeans = vocabulary.means.eval()
                # aftSigmas = vocabulary.sigmas.eval()
                #
                # import pickle as pk
                #
                # with open('iter' + str(i) + '.pkl', 'w') as f:
                #     pk.dump({'orgMeans': orgMeans, 'orgSigmas': orgSigmas, 'aftMeans': aftMeans, 'aftSigmas': aftSigmas}, f)

                vocabulary.saveVocabWithEmbeddings(opt.savePath, sess)
            else:
                raise Exception(opt.train)


if __name__ == "__main__":
    tf.app.run()
