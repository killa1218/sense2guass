#!/usr/bin/python
# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import random

# from six.moves import xrange    # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from vocab import Vocab as v
from options import Options as opt
from exceptions import NotAFileException
from loss import skipGramNCELoss as loss
import pickle as pk

random.seed(time.time())

flags = tf.app.flags

flags.DEFINE_string("output", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train", None, "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string("vocab", None, "The vocabulary file path.")
flags.DEFINE_string("save_vocab", None, "If not None, save the vocabulary to this path.")
flags.DEFINE_integer("size", 100, "The embedding dimension size. Default is 100.")
flags.DEFINE_integer("window", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("negative", 100, "Negative samples per training example. Default is 100.")
flags.DEFINE_integer("threads", 12, "How many threads are used to train. Default 12.")
flags.DEFINE_integer("iter", 15, "Number of iterations to train. Each iteration processes the training data once completely. Default is 15.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary. Default is 5.")
flags.DEFINE_float("alpha", 0.2, "Initial learning rate. Default is 0.2.")
flags.DEFINE_boolean("gpu", False, "If true, use GPU instead of CPU.")

flags.DEFINE_integer("batch_size", 128, "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_boolean("interactive", False,
                     "If true, enters an IPython interactive session to play with the trained model. E.g., try model.analogy(b'france', b'paris', b'russia') and model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval).")

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
opt.threads = FLAGS.threads
# Number of examples for one training step.
opt.batch_size = FLAGS.batch_size
# The number of words to predict to the left and right of the target word.
opt.windowSize = FLAGS.window
# The minimum number of word occurrences for it to be included in the vocabulary.
opt.min_count = FLAGS.min_count
# Subsampling threshold for word occurrence.
opt.sample = FLAGS.sample
# Load vocabulary from file.
opt.vacab = FLAGS.vocab
# Save the vocab to a file.
opt.save_vocab = FLAGS.save_vocab
# How often to print statistics.
opt.statistics_interval = FLAGS.statistics_interval
# How often to write to the summary file (rounds up to the nearest statistics_interval).
opt.summary_interval = FLAGS.summary_interval
# How often to write checkpoints (rounds up to the nearest statistics interval).
opt.checkpoint_interval = FLAGS.checkpoint_interval
# Use GPU or CPU. True for GPU, otherwise CPU
opt.gpu = FLAGS.gpu
# Where to write out summaries.
opt.save_path = FLAGS.output
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)


def train(s):
    global sess

    stc = []

    for i in s.split(' '):
        stc.append(vocab.getWord(i))

    # Do Inference
    sLabel = inference(stc)

    # Do Optimize
    l = loss(stc, sLabel)

    optimizer = tf.train.GradientDescentOptimizer(opt.alpha)
    sess.run(optimizer.minimize(l))


def negativeLoss():
    pass


def buildGraph(stc, sLabel):
    pass


vocab = None


def main(_):
    """Train a sense2guass model."""
    if not FLAGS.train or not FLAGS.output:  # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)
    # opt = Options()  # Instantiate option object

    stc = None
    device = "/cpu:0"
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build vocabulary or load vocabulary from file
        if opt.vacab != None:
            vocab = pk.load(opt.vocab)
        else:
            vocab = v(opt.train)

            if opt.save_vocab:
                pk.dump(vocab, opt.save_vocab)


        # Read batch sentences in
        if os.path.isfile(opt.train):
            with open(opt.train) as f:
                if os.path.getsize(opt.train) > 2000000000:
                    for line in f.readline():
                        train(line)
                else:
                    for line in f.readlines():
                        train(line)
        else:
            raise NotAFileException(file)









            # # if opt.gpu:                                # Judge whether use CPU(default) or GPU
            # #     device = "/gpu:0"
            # #
            # # with tf.device(device):
            # model = Word2Vec(opt, session)         # Instantiate model
            #     # model.read_analogies() # Read analogy questions
            # for _ in xrange(opt.iter):      # According to the parameter train the embeddings for iter iterations
            #     model.train()    # Process one epoch
            #     # model.eval()    # Eval analogies.
            # # Perform a final save.
            # model.saver.save(session, os.path.join(opt.save_path, "model.ckpt"), global_step=model.global_step)    # Save model
            # # if FLAGS.interactive:
            # #     # E.g.,
            # #     # [0]: model.analogy(b'france', b'paris', b'russia')
            # #     # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
            # #     _start_shell(locals())


if __name__ == "__main__":
    tf.app.run()
