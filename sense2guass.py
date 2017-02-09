#!/usr/local/bin/python3
# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import random

# from six.moves import xrange    # pylint: disable=redefined-builtin


from tqdm import tqdm
from vocab import Vocab as V
from options import Options as opt
from loss import skipGramNCELoss as loss
from e_step.inference import dpInference as inference
from threadpool import *
from utils.fileIO import fetchSentences
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


def train(batch, sess, optimizer):
    global vocabulary

    l = tf.constant(0., dtype=tf.float64, name='tmp')

    for stc in batch:
        stcW = []

        for i in stc:
            w = vocabulary.getWord(i)

            if w:
                stcW.append(w)

        if len(stcW) > opt.windowSize and len(stcW) > opt.minSentenceLength:
            # E-Step: Do Inference
            print('Inferencing sentence:', ' '.join(stc))
            sLabel = inference(stcW, vocabulary, sess)

            print('Inference of sentence:', sLabel)

            # Build loss
            l += loss(stcW, sLabel, vocabulary, sess)

    if 'tmp' not in l.name:
        # M-Step: Do Optimize
        sess.run(optimizer(l))
        print('Loss:', sess.run(l))


def main(_):
    """ Train a sense2guass model. """
    global vocabulary

    if not FLAGS.train or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)

    with tf.Session() as sess:
        tp = ThreadPool(1)
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

        tf.global_variables_initializer().run(session=sess)

        # Train iteration
        print('Start training...')
        for i in tqdm(range(opt.iter)):
            print('\n')
            if os.path.isfile(opt.train):
                with open(opt.train) as f:
                    batch = []

                    for stc in fetchSentences(f, 20000, opt.maxSentenceLength):
                        batch.append(stc)

                        if len(batch) == opt.batchSize:
                            train(batch, sess, optimizer)
                            del(batch)
                            batch = []

                    train(batch, sess, optimizer)

                # Save training result
                requests = makeRequests(vocabulary.saveEmbeddings, [opt.save_path])
                for req in requests:
                    tp.putRequest(req)
            else:
                raise Exception(file)


if __name__ == "__main__":
    tf.app.run()
