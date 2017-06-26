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

flags.DEFINE_string("output", '.pkl', "Directory to write the model and training summaries.")
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
opt.savePath = './data/' + opt.energy + '.' + time.strftime("%m%d%H%M", time.localtime()) + 'w' + str(opt.windowSize) +\
               'b' + str(opt.batchSize) + 'lr' + str(opt.alpha) + 'm' + str(opt.margin) + 'n' + str(opt.negative) + FLAGS.output + '.pkl'

vocabulary = None
e_step = True
m_step = True
opt.verboseTime = False

gradMin = -5
gradMax = 5

def main(_):
    """ Train a sense2guass model. """
    global vocabulary

    if not FLAGS.train: # or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train must be specified.")
        sys.exit(1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # Config to make tensorflow not take up all the GPU memory
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess, tf.device('CPU:0'), open('console_output.small.txt', 'w') as of:
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(opt.alpha, global_step,
                                                   3000, 0.96, staircase = True)
        # learning_rate = opt.alpha
        # Passing global_step to minimize() will increment it at each step.
        # optimizer = tf.train.AdagradOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # meanOpt = tf.train.AdamOptimizer()
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        # optimizer = tf.train.FtrlOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # sigmaOpt = tf.train.AdadeltaOptimizer(opt.alpha)

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
        # windowLossGraph, window = windowLossGraph(vocabulary)
        print('Finished.')
##----------------- Build Window Loss Graph ------------------

        if m_step:
        ##---------------------------- Build NCE Loss --------------------------------
            print('Building NCE Loss...')
            from graph import batchNCELossGraph

            lossList = batchNCELossGraph(vocabulary)
            senseIdxPlaceholder = tf.get_collection('POS_PHDR')[0]

            posLosses = tf.get_collection('POS_LOSS')
            negLosses = tf.get_collection('NEG_LOSS')

            # Margin Loss
            loss = tf.reduce_sum(lossList)
            avgLoss = loss / opt.batchSize / opt.negative / len(lossList)
            avgPosLoss = tf.reduce_sum(posLosses) / len(posLosses) / opt.batchSize
            avgNegLoss = tf.reduce_sum(negLosses) / opt.sentenceLength / opt.negative / opt.batchSize

            # Cross Entropy Loss
            # posLen = len(posLosses)
            # posLosses = tf.concat(posLosses, 0)
            #mse
            # negLosses = tf.concat(negLosses, 0)
            # avgPosLoss = tf.reduce_sum(-tf.log(1 - posLosses)) / posLen / opt.batchSize
            # avgNegLoss = tf.reduce_sum(-tf.log(negLosses)) / opt.sentenceLength / opt.negative / opt.batchSize

            # avgPosLoss = tf.reduce_sum(posLosses) / posLen / opt.batchSize
            # avgNegLoss = tf.reduce_sum(1 - negLosses) / opt.sentenceLength / opt.negative / opt.batchSize
            # avgPosLoss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(posLosses), logits = posLosses)) / posLen / opt.batchSize
            # avgNegLoss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(negLosses), logits = negLosses)) / opt.sentenceLength / opt.negative / opt.batchSize
            # loss = avgPosLoss + avgNegLoss
            # avgLoss = loss / 2

            with tf.name_scope("Summary"):
                tf.summary.scalar("Positive Loss", avgPosLoss)
                tf.summary.scalar("Negative Loss", avgNegLoss)
                # tf.summary.scalar("Num of Nonzero Loss", nonzeroNum)
                tf.summary.scalar("NCE Loss", avgLoss)
                avg_mean_norm, mean_var = tf.nn.moments(tf.norm(vocabulary.means, axis = 1), axes = [0])
                avg_cov_norm, cov_var = tf.nn.moments(tf.norm(vocabulary.sigmas, axis = 1), axes = [0])
                if opt.energy == 'EL':
                    tf.summary.scalar("Average Mean Norm", avg_mean_norm)
                    tf.summary.scalar("Mean Norm Variance", mean_var)
                    tf.summary.scalar("Average Sigma Norm", avg_cov_norm)
                    tf.summary.scalar("Sigma Norm Variance", cov_var)
                    posleng = len(tf.get_collection('POS_LOG_EL_FIRST'))
                    possecleng = len(tf.get_collection('POS_LOG_EL_SECOND'))
                    negleng = len(tf.get_collection('NEG_LOG_EL_FIRST'))
                    negsecleng = len(tf.get_collection('NEG_LOG_EL_SECOND'))
                    tf.summary.scalar("POS Log EL First", tf.reduce_sum(tf.add_n(tf.get_collection('POS_LOG_EL_FIRST'))) / 100 / posleng)
                    tf.summary.scalar("POS Log EL Second", tf.reduce_sum(tf.add_n(tf.get_collection('POS_LOG_EL_SECOND'))) / 100 / possecleng)
                    tf.summary.scalar("NEG Log EL First", tf.reduce_sum(tf.add_n(tf.get_collection('NEG_LOG_EL_FIRST'))) / 100 / negleng)
                    tf.summary.scalar("NEG Log EL Second", tf.reduce_sum(tf.add_n(tf.get_collection('NEG_LOG_EL_SECOND'))) / 100 / negsecleng)
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter('log' + opt.energy + '/' + time.strftime("%m%d", time.localtime()) + '/' + time.strftime("%H:%M", time.localtime()) +
                                                       '_w' + str(opt.windowSize) + 'b' + str(opt.batchSize) + 'lr' + str(opt.alpha) + '' + str(opt.margin) +\
                                                       'n' + str(opt.negative) + 'adam', graph = sess.graph)
            print('Finished.')
            regular = 0 # -tf.norm(vocabulary.sigmas, ord = 'euclidean') if opt.covarShape != 'none' else 0
            obj = loss

            print('Building Optimizer...')

            if opt.energy == 'IP':
                grad = optimizer.compute_gradients(obj, var_list = [vocabulary.means, vocabulary.outputMeans], gate_gradients = optimizer.GATE_NONE)
            elif opt.energy == 'EL':
                grad = optimizer.compute_gradients(obj, var_list = [vocabulary.trainableSigmas, vocabulary.trainableOutputSigmas], gate_gradients = optimizer.GATE_NONE)
            elif opt.energy == 'MSE':
                grad = optimizer.compute_gradients(obj, var_list = [vocabulary.trainableMeans, vocabulary.trainableOutputMeans], gate_gradients = optimizer.GATE_NONE)
            else:
                grad = optimizer.compute_gradients(obj, gate_gradients = optimizer.GATE_NONE)
            clipedGrad = grad
            # clipedGrad = [(tf.clip_by_value(g, gradMin, gradMax), var) for g, var in grad]
            # # clipedGrad = [
            # #     (tf.multiply(g, opt.gradConstraint, name = 'Scaled_Cov_Grad'), var) if 'igmas' in var.name else
            # #     (tf.clip_by_value(g, gradMin, gradMax, name = 'Cliped_Mean_Grad'), var) for g, var in grad] # Limit covariance gradients
            #
            op = optimizer.apply_gradients(clipedGrad, global_step = global_step)
            # tf.nn.sigmoid_cross_entropy_with_logits()
            # op = optimizer.minimize(loss + regular)
            # # op = optimizer(avgBatchStcLoss)
            print('Finished.')
        ##---------------------------- Build NCE Loss --------------------------------

        tf.global_variables_initializer().run(session=sess)
        # Train iteration
        print('Start training...\n')

        from e_step.cinference import batchDPInference
        # from e_step.inference import batchDPInference as pyinference
        # from multiprocessing import Pool
        # pool = Pool()

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

                                if len(batchStcW) == opt.batchSize:
                                    batchLossSenseIdxList = []
                                    # negativeSamplesList = np.random.randint(vocabulary.totalSenseCount, size=(opt.batchSize, opt.sentenceLength, opt.negative))
    ##--------------------------------- Inference By Batch ----------------------------------
                                    start = time.time()
                                    if opt.maxSensePerWord == 1:
                                        for p in batchStcW:
                                            tmpList = []

                                            for q in p:
                                                tmpList.append(q.senseStart)

                                            batchLossSenseIdxList.append(tmpList)
                                    else:
                                        # batchLossSenseIdxList, twT, tcT, tiT = batchDPInference(batchStcW, sess, windowLossGraph, window)

                                        # if opt.verboseTime:
                                            # tIT += time.time() - start
                                            # wT += twT
                                            # cT += tcT
                                            # iT += tiT
                                            # count += 1
                                            # print("\n\nTotal Inference Time:", tIT / count, '\n')
                                            # print("Get Windows Time Takes: %.2f%%" % (wT * 100 / tIT))
                                            # print("Calculate Time Takes: %.2f%%" % (cT * 100 / tIT))
                                            # print("DP Time Takes: %.2f%%" % (iT * 100 / tIT))
                                            # print("Data Transfer Time Takes: %.2f%%" % ((tIT - iT - cT - wT) * 100 / tIT))
                                        pass
    ##--------------------------------- Inference By Batch ----------------------------------

                                    if m_step:
                                        # summ, posloss = sess.run([merged, reduceAvgLoss], feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        # writer.add_summary(summ, i)
                                        start = time.time()
                                        posloss = sess.run(avgPosLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        negloss = sess.run(avgNegLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        nceloss = sess.run(avgLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                        # negloss = sess.run(avgNegLoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        # nceloss = sess.run(loss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                        # print(sess.run(grad, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList}))

                                        if isinstance(learning_rate, float):
                                            lr = learning_rate
                                            #print(global_step.eval(), learning_rate)
                                        else:
                                            lr = learning_rate.eval()
                                            #print(global_step.eval(), learning_rate.eval())

                                        # if global_step.eval() % 3000 == 1:
                                            # of.write('Iter:%d/%d, Step:%d, Lr:%.5f POSLoss:%.5f, NEGLoss:%.5f, NCELoss:%.5f, Progress:%.2f%%.\n' % (i + 1, opt.iter, global_step.eval(), lr, posloss, negloss, nceloss, (float(f.tell()) * 100 / os.path.getsize(opt.train))))
                                            #
                                            # for g, var in grad:
                                            #     if isinstance(g, tf.Tensor):
                                            #         l = []
                                            #         il = []
                                            #         covg = g.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                            #         for idx, tmpi in enumerate(covg):
                                            #             if np.sum(tmpi) != 0:
                                            #                 l.append(tmpi)
                                            #                 il.append(idx)
                                            #         of.write('Gradients ')
                                            #         of.write(repr(l))
                                            #         of.write('\n')
                                            #         of.write(var.name)
                                            #         of.write(' ')
                                            #         of.write(repr(tf.gather(var, il).eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})))
                                            #         of.write('\n')
                                            #         # print('Gradients', l)
                                            #         # print(var.name, tf.gather(var, il).eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                            #         # print('Gradients', tf.count_nonzero(g).eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                            #     else:
                                            #         of.write('Index: ' + repr(g.indices.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})) +
                                            #                  '\nGradients: ' + repr(g.values.eval(feed_dict = {senseIdxPlaceholder: batchLossSenseIdxList})) +
                                            #                  '\n' + var.name + ' ' + repr(tf.gather(var, g.indices).eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})) +
                                            #                  '\n'
                                            #         )
                                            #
                                            #         # print('\nIndex:', g.indices.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                            #         # print('Gradients:', g.values.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                            #         # # print('Unknown:', g.dense_shape.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))
                                            #         # print(var.name, tf.gather(var, g.indices).eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}))

                                        sys.stdout.write('\rIter:%d/%d, Step:%d, Lr:%.5f POSLoss:%.5f, NEGLoss:%.5f, NCELoss:%.5f, Progress:%.2f%%.' % (i + 1, opt.iter, global_step.eval(), lr, posloss, negloss, nceloss, (float(f.tell()) * 100 / os.path.getsize(opt.train))))
                                        sys.stdout.flush()

                                        # print(sess.run(grad, feed_dict ={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})[0])

                                        summary_writer.add_summary(summary_op.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList}), global_step.eval())
                                        # summary_writer.add_summary(summary_op.eval(feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList}), global_step.eval())
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
                                            sess.run(op, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                            # sess.run([mop, sop], feed_dict={senseIdxPlaceholder: batchLossSenseIdxList})
                                            # sess.run(op, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, negSamples: negativeSamplesList})
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
