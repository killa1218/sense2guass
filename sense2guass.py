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

random.seed(time.time())

flags = tf.app.flags

flags.DEFINE_string("output", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train", None, "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("vocab", None, "The vocabulary file path.")
flags.DEFINE_string("save_vocab", None, "If not None, save the vocabulary to this path.")
# flags.DEFINE_string("covariance", "diagnal", "Shape of covariance matrix, default is diagnal. Possible value is 'diagnal' or ")
flags.DEFINE_integer("size", 50, "The embedding dimension size. Default is 100.")
flags.DEFINE_integer("window", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("negative", 4, "Negative samples per sense. Default is 4.")
flags.DEFINE_integer("threads", 3, "How many threads are used to train. Default 12.")
flags.DEFINE_integer("iter", 15, "Number of iterations to train. Each iteration processes the training data once completely. Default is 15.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary. Default is 5.")
flags.DEFINE_integer("max_sentence_length", 20, "The maximum length of one sentence.")
flags.DEFINE_integer("min_sentence_length", 5, "The minimum length of one sentence.")
flags.DEFINE_integer("sentence_length", 15, "The length of one sentence.")
flags.DEFINE_integer("max_sense_per_word", 5, "The maximum number of one word.")
flags.DEFINE_integer("batch_size", 20, "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_float("alpha", 0.001, "Initial learning rate. Default is 0.001.")
flags.DEFINE_float("margin", 100, "Margin between positive and negative training pairs. Default is 100.")
flags.DEFINE_boolean("gpu", False, "If true, use GPU instead of CPU.")
flags.DEFINE_boolean("EL", False, "Use EL as energy function or KL, default is KL.")

FLAGS = flags.FLAGS

# Embedding dimension.
opt.embSize = FLAGS.size
# Training options. The training text file.
opt.train = FLAGS.train
# Number of negative samples per example.
opt.negative = FLAGS.negative
# The initial learning rate.
opt.alpha = FLAGS.alpha
# Margin between positive and negative pairs.
opt.margin = FLAGS.margin
# Number of epochs to train.
opt.iter = FLAGS.iter
# Concurrent training steps.
# opt.threads = FLAGS.threads
# Number of examples for one training step.
opt.batchSize = FLAGS.batch_size
# The number of words to predict to the left and right of the target word.
opt.windowSize = FLAGS.window
# The minimum number of word occurrences for it to be included in the vocabulary.
opt.minCount = FLAGS.min_count
# The maximum length of one sentence in training.
opt.maxSentenceLength = FLAGS.max_sentence_length
# The minimum length of one sentence in training.
opt.minSentenceLength = FLAGS.min_sentence_length
# The length of one sentence in training.
opt.sentenceLength = FLAGS.sentence_length
# The maximum sense number of one word in training.
opt.maxSensePerWord = FLAGS.max_sense_per_word
# Subsampling threshold for word occurrence.
# opt.sample = FLAGS.sample
# Load vocabulary from file.
opt.vocab = FLAGS.vocab
# Save the vocab to a file.
opt.saveVocab = FLAGS.save_vocab
# Use GPU or CPU. True for GPU, otherwise CPU
opt.gpu = FLAGS.gpu
# Use EL or KL, True for EL, otherwise KL
opt.EL = FLAGS.EL
# Where to write out summaries.
opt.savePath = FLAGS.output

vocabulary = None

def main(_):
    """ Train a sense2guass model. """
    global vocabulary

    if not FLAGS.train or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        optimizer = tf.train.AdamOptimizer(opt.alpha).minimize
        # merged_summary_op = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter('data/output', sess.graph)

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

##----------------- Build Sentence Loss Graph And Optimizer ------------------
        print('Building Sentence Loss Graph...')
        from graph import batchSentenceLossGraph as lossGraph
        batchSentenceLossGraph, (senseIdxPlaceholder) = lossGraph(vocabulary)
        minLossIdxGraph = tf.argmin(batchSentenceLossGraph, 0)
        avgBatchStcLoss = batchSentenceLossGraph / (opt.sentenceLength * opt.windowSize * 2 - (opt.windowSize + 1) * opt.windowSize)
        reduceAvgLoss = tf.reduce_sum(avgBatchStcLoss) / opt.batchSize
        print('Finished Building Sentence Loss Graph.')
##----------------- Build Sentence Loss Graph And Optimizer ------------------

##----------------- Build Window Loss Graph ------------------
        print('Building Window Loss Graph...')
        from graph import windowLossGraph
        windowLossGraph, window = windowLossGraph(vocabulary)
        print('Finished Building Window Loss Graph.')
##----------------- Build Window Loss Graph ------------------

##----------------- Build Batch Violent Inference Graph ------------------
        # print('Building Batch Violent Inference Graph...')
        # from graph import batchInferenceGraph
        # batchInferenceGraph, batchSenseIdxPlaceholder = batchInferenceGraph(vocabulary)
        # print('Finished Building Batch Violent Inference Graph.')
##----------------- Build Batch Violent Inference Graph ------------------

##----------------------- Build Negative Loss Graph --------------------------
        print('Building Negative Loss Graph...')
        from graph import negativeLossGraph
        negativeLossGraph, (mid, negSamples) = negativeLossGraph(vocabulary)
        avgNegLoss = negativeLossGraph / opt.negative / opt.sentenceLength
        reduceAvgNegLoss = tf.reduce_sum(avgNegLoss) / opt.batchSize
        print('Finished Building Negative Loss Graph.')
##----------------------- Build Negative Loss Graph --------------------------

##---------------------------- Build NCE Loss --------------------------------
        print('Building NCE Loss...')
        nceLossGraph = tf.nn.relu(opt.margin - avgBatchStcLoss + avgNegLoss)
        reduceNCELoss = tf.reduce_sum(nceLossGraph)
        avgNCELoss = reduceNCELoss / opt.batchSize
        op = optimizer(reduceNCELoss)
        # op = optimizer(avgBatchStcLoss)
        print('Finished Building NCE Loss.')
##---------------------------- Build NCE Loss --------------------------------

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

        from multiprocessing import Pool
        from e_step.cinference import batchDPInference

        for i in range(opt.iter):
            if os.path.isfile(opt.train):
                with open(opt.train) as f:
                    batchLossSenseIdxList = []
                    negativeSamplesList = []
                    batchStcW = []

                    # start = time.time()
                    for stcW in fetchSentencesAsWords(f, vocabulary, 20000, opt.sentenceLength, verbose=False):
##----------------------------- Train Batch ------------------------------
                        if len(stcW) > opt.windowSize and len(stcW) == opt.sentenceLength:
                            batchStcW.append(stcW)
            # E-Step: Do Inference
            #                 print('Inferencing sentence:', ' '.join(x.token for x in stcW))
            #                 start = time.time()

                            # end = time.time()
                            # sys.stdout.write('\rINFERENCE TIME: %.6f' % (end - start))
                            # sys.stdout.flush()
                            # print('INFERENCE TIME:', end - start)
                            # print('Inference of sentence:', assign)

                            # Build loss

                            negativeSampleList = []
                            # for a in assign:
                            for a in range(len(stcW)):
                                sampleTmp = random.sample(range(vocabulary.totalSenseCount), opt.negative)
                                while a in sampleTmp:
                                    sampleTmp = random.sample(range(vocabulary.totalSenseCount), opt.negative)

                                negativeSampleList.append(sampleTmp)
                            negativeSamplesList.append(negativeSampleList)

            # M-Step: Do Optimize
            #                 if len(batchLossSenseIdxList) == opt.batchSize:
                            if len(batchStcW) == opt.batchSize:
##--------------------------------- Inference By Batch ----------------------------------
                                # from e_step.inference import batchViolentInference
                                # start = time.time()
                                # batchLossSenseIdxList = batchViolentInference(batchStcW, sess, batchSentenceLossGraph, senseIdxPlaceholder, argmin, lossPlaceholder)
                                # pool = Pool()
                                batchLossSenseIdxList = batchDPInference(batchStcW, sess, windowLossGraph, window)
                                # from e_step.inference import batchDPInference
                                # batchLossSenseIdxList = batchDPInference(batchStcW, sess, windowLossGraph, window, pool)
                                # end = time.time()
                                # print('Inference time: %.5f' % (end - start))
##--------------------------------- Inference By Batch ----------------------------------

                                # end = time.time()
                                # print('Inferencing time: %.5f' % (end - start))
                                # start = time.time()

                                # loss = sess.run(avgNCELoss, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, mid: batchLossSenseIdxList, negSamples: negativeSamplesList})
                                loss = 0.
                                sys.stdout.write('\rIter: %d/%d, NCELoss: %.8f, Progress: %.2f%%.' % (i + 1, opt.iter, loss, (float(f.tell()) * 100 / os.path.getsize(opt.train))))
                                sess.run(op, feed_dict={senseIdxPlaceholder: batchLossSenseIdxList, mid: batchLossSenseIdxList, negSamples: negativeSamplesList})

                                # end = time.time()
                                # print('Optimization time: %.5f' % (end - start))

                                del(batchLossSenseIdxList)
                                del(negativeSamplesList)
                                del(batchStcW)
                                batchStcW = []
                                negativeSamplesList = []
                                batchLossSenseIdxList = []

                                # start = time.time()

##----------------------------- Train Batch ------------------------------

                # Save training result
                # from threadpool import *
                # tp = ThreadPool(1)
                # requests = makeRequests(vocabulary.saveEmbeddings, [opt.saveVocab])
                # for req in requests:
                #     tp.putRequest(req)

                vocabulary.saveVocabWithEmbeddings(opt.savePath, sess)
            else:
                raise Exception(opt.train)


if __name__ == "__main__":
    tf.app.run()
