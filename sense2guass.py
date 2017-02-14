#!/usr/local/bin/python3
# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random
import multiprocessing

# from six.moves import xrange    # pylint: disable=redefined-builtin


from tqdm import tqdm
from vocab import Vocab as V
from options import Options as opt
from loss import skipGramWindowLoss
from loss import skipGramNCELoss as loss
# from e_step.inference import dpInference as inference
# from e_step.inference import violentInference as inference
from threadpool import *
from utils.fileIO import fetchSentencesAsWords
import tensorflow as tf

random.seed(time.time())

flags = tf.app.flags

flags.DEFINE_string("output", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train", None, "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("vocab", None, "The vocabulary file path.")
flags.DEFINE_string("save_vocab", None, "If not None, save the vocabulary to this path.")
flags.DEFINE_integer("size", 50, "The embedding dimension size. Default is 100.")
flags.DEFINE_integer("window", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("negative", 100, "Negative samples per training example. Default is 100.")
flags.DEFINE_integer("threads", 3, "How many threads are used to train. Default 12.")
flags.DEFINE_integer("iter", 15, "Number of iterations to train. Each iteration processes the training data once completely. Default is 15.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary. Default is 5.")
flags.DEFINE_integer("max_sentence_length", 20, "The maximum length of one sentence.")
flags.DEFINE_integer("min_sentence_length", 5, "The minimum length of one sentence.")
flags.DEFINE_integer("max_sense_per_word", 5, "The maximum number of one word.")
flags.DEFINE_float("alpha", 0.2, "Initial learning rate. Default is 0.2.")
flags.DEFINE_boolean("gpu", False, "If true, use GPU instead of CPU.")
flags.DEFINE_integer("batch_size", 1, "Number of training examples processed per step (size of a minibatch).")

FLAGS = flags.FLAGS

# Embedding dimension.
opt.embSize = FLAGS.size
# Training options. The training text file.
opt.train = FLAGS.train
# Number of negative samples per example.
opt.negative = FLAGS.negative
# The initial learning rate.
opt.alpha = FLAGS.alpha
# Number of epochs to train. After these many epochs, the learning rate decays linearly to zero and the training stops.
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
# Where to write out summaries.
opt.save_path = FLAGS.output
# if not os.path.exists(opt.save_path):
#     os.makedirs(opt.save_path)


vocabulary = None
# pool = multiprocessing.Pool()
# inferenceGraph = skipGramWindowKLLossGraph()

def train(batch, sess, optimizer, res):
    global vocabulary
    # global inferenceGraph




def main(_):
    """ Train a sense2guass model. """
    global vocabulary

    if not FLAGS.train or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)

    with tf.Session() as sess:
        optimizer = tf.train.GradientDescentOptimizer(opt.alpha).minimize

        # Build vocabulary or load vocabulary from file
        if opt.vocab != None:
            vocabulary = V()
            vocabulary.load(opt.vocab)
        else:
            vocabulary = V(opt.train)
            vocabulary.initAllSenses()

            if opt.saveVocab:
                vocabulary.save(opt.saveVocab. sess)

##----------------- Build Window Loss Graph --------------------
        from utils.distance import diagKL
        mid = tf.placeholder(dtype=tf.int32, name='mid')
        others = tf.placeholder(dtype=tf.int32, name='others')

        midMean = tf.nn.embedding_lookup(vocabulary.means, mid)
        midSigma = tf.nn.embedding_lookup(vocabulary.sigmas, mid)
        l = []

        for i in range(opt.windowSize * 2):
            l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, others[i]), tf.nn.embedding_lookup(vocabulary.sigmas, others[i]), midMean, midSigma))
            l.append(diagKL(tf.nn.embedding_lookup(vocabulary.means, others[opt.windowSize * 2 - i - 1]), tf.nn.embedding_lookup(vocabulary.sigmas, others[opt.windowSize * 2 - i - 1]), midMean, midSigma))

        windowLossGraph = tf.clip_by_value(tf.add_n(l), tf.float64.min, tf.float64.max)
##----------------- Build Window Loss Graph --------------------

##----------------- Build Sentence Loss Graph ------------------
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

        sentenceLossGraph = tf.clip_by_value(tf.add_n(l), tf.float64.min, tf.float64.max)
##----------------- Build Sentence Loss Graph ------------------

        tf.global_variables_initializer().run(session=sess)

        # Train iteration
        print('Start training...')
        for i in tqdm(range(opt.iter)):
            print('\n')
            if os.path.isfile(opt.train):
                with open(opt.train) as f:
                    batch = []

                    for stcW in fetchSentencesAsWords(f, vocabulary, 20000, opt.sentenceLength):
                        batch.append(stcW)

##----------------------------- Train Batch ------------------------------
                        if len(batch) == opt.batchSize:
                            l = tf.constant(0., dtype=tf.float64, name='tmp')

                            for stcW in batch:
                                if len(stcW) > opt.windowSize and len(stcW) > opt.minSentenceLength:
    # E-Step: Do Inference
                                    print('Inferencing sentence:', ' '.join(str(stcW)))
                                    start = time.time()
                                    assign = None
                                    # sLabel = inference(stcW, sess)

                                    def dfs(stcW, mid, sess):
                                        l = len(stcW)
                                        fullWindowSize = 0
                                        global vocabulary

                                        if l < opt.windowSize * 2 + 1:
                                            fullWindowSize = l
                                        elif mid + opt.windowSize >= l:
                                            fullWindowSize = l - mid + opt.windowSize
                                        elif mid - opt.windowSize < 0:
                                            fullWindowSize = mid + opt.windowSize
                                        else:
                                            fullWindowSize = opt.windowSize * 2 + 1

                                        stack = [0] * fullWindowSize

                                        yield stack, skipGramWindowLoss(stcW, stack, mid).eval(), mid
                                        # yield stack, skipGramWindowLoss(stcW, stack, mid), mid

                                        while True:
                                            if (len(stack) == 0):
                                                break
                                            else:
                                                if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
                                                    stack.pop()
                                                else:
                                                    stack[-1] += 1
                                                    stack += [0] * (fullWindowSize - len(stack))
                                                    loss = skipGramWindowLoss(stcW, stack, mid).eval()
                                                    # loss = skipGramWindowLoss(stcW, stack, mid)

                                                    # print('\tASSIGN:', stack, 'LOSS:', loss)

                                                    yield stack, loss, mid

                                    v = {}  # Record Intermediate Probability
                                    tmpV = None
                                    assign = []  # Result of word senses in a sentence
                                    # minLoss = float('inf')  # Minimum loss

                                    assert len(stcW) > opt.windowSize
                                    print('Initializing first words...')
                                    for a, l, m in dfs(stcW, opt.windowSize, sess):
                                    # for a, l, m in tqdm(dfs(stcW, opt.windowSize, sess)):
                                        v[tuple(a)] = l

                                    tmpV = {}
                                    for j in v:
                                        li = list(j)
                                        jj = []

                                        for jjj in range(len(li)):
                                            jj.append(stcW[jjj].senseStart + li[jjj])

                                        tmpV[tuple(jj)] = v[j]

                                    del(v)
                                    v = tmpV

                                    print('Initialize first words finished.')

                                    print('Inferencing other words...')
                                    # for i in range(opt.windowSize + 1, len(stcW)):
                                    for i in tqdm(range(opt.windowSize + 1, len(stcW))):
                                        minLoss = float('inf')  # Minimum loss
                                        newWord = stcW[i + opt.windowSize] if i + opt.windowSize < len(stcW) else None
                                        del(tmpV)
                                        tmpV = {}
                                        assignList = []
                                        # lossTensorList = []

                                        midList = []
                                        otherList = []
                                        prevLossList = []

                                        for j in v:
                                            prevAssign = list(j)
                                            midSenseIdx = prevAssign[i]
                                            # prevLoss = v[j]
                                            # print('\tASSIGN:', prevAssign, 'LOSS:', prevLoss)

                                            prevSenseIdx = prevAssign[-opt.windowSize * 2:]
                                            prevSenseIdx.remove(midSenseIdx)
                                            # for k in range(1, opt.windowSize + 1):
                                            #     if i - k < 0:
                                            #         prevSenseIdx.append(stcW[i].senseStart + prevAssign[i])
                                            #     else:
                                            #         prevSenseIdx.append(stcW[i - k].senseStart + prevAssign[i - k])
                                            #     if i + k < len(stcW):
                                            #         if k < opt.windowSize:
                                            #             prevSenseIdx.append(stcW[i + k].senseStart + prevAssign[i + k])
                                            #     else:
                                            #         prevSenseIdx.append(stcW[i].senseStart + prevAssign[i])

                                            if newWord:
                                                for k in range(0, newWord.senseNum):
                                                    curAssign = prevAssign + [newWord.senseStart + k]
                                                    # start = time.time()
                                                    # curLoss = prevLoss + skipGramWindowLoss(stcW, curAssign, i)
                                                    # end = time.time()
                                                    # print('TIME SPENT:', end - start)

                                                    assignList.append(curAssign)
                                                    # lossTensorList.append(curLoss)
                                                    midList.append(midSenseIdx)
                                                    otherList.append(prevSenseIdx + [newWord.senseStart + k])
                                                    prevLossList.append(v[j])


                                                    # tmpV[tuple(curAssign)] = curLoss

                                            else:
                                                # curLoss = prevLoss + skipGramWindowLoss(stcW, prevAssign, i)

                                                assignList.append(prevAssign)
                                                # lossTensorList.append(curLoss)
                                                midList.append(midSenseIdx)
                                                otherList.append(midSenseIdx)
                                                prevLossList.append(v[j])

                                                # tmpV[tuple(prevAssign)] = curLoss

                                        # print('\tSearch state table finished.')

                                        del(v)
                                        v = {}

                                        # lossList = sess.run(lossTensorList)

                                        # for j in range(len(lossList)):
                                        #     if lossList[j] < minLoss:
                                        #         minLoss = lossList[j]
                                        #         assign = assignList[j][:]
                                        # assert len(midList) == len(otherList) and len(midList) == opt.sentenceLength
                                        lossList = sess.run(tf.constant(prevLossList, dtype=tf.float64) + windowLossGraph, feed_dict={mid: midList, others: otherList})

                                        for j in range(len(lossList)):
                                            tmpV[tuple(assignList[j])] = lossList[j]

                                            if lossList[j] < minLoss:
                                                minLoss = lossList[j]
                                                assign = assignList[j][:]

                                        del(assignList)
                                        # del(lossTensorList)

                                        for j in tmpV:
                                            if j[i - opt.windowSize - 1] == assign[i - opt.windowSize - 1]:
                                                v[j] = tmpV[j]





                                    end = time.time()
                                    print('INFERENCE TIME:', end - start)

                                    print('Inference of sentence:', assign)

                                    # Build loss
                                    l += loss(stcW, assign, vocabulary, sess)

                            if 'tmp' not in l.name:
    # M-Step: Do Optimize
                                sess.run(optimizer(l))
                                print('Loss:', sess.run(l))
                            del(batch)
                            batch = []
##----------------------------- Train Batch ------------------------------

                    # train(batch, sess, optimizer)

                # Save training result
                tp = ThreadPool(1)
                requests = makeRequests(vocabulary.saveEmbeddings, [opt.save_path])
                for req in requests:
                    tp.putRequest(req)
            else:
                raise Exception(file)


if __name__ == "__main__":
    tf.app.run()
