# coding=utf8

import sys
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import batchSentenceLossGraph
from e_step.inference import violentInference
from vocab import Vocab
from utils.distance import diagKL as dist
# from utils.distance import diagEL as dist
from options import Options as opt

data = None
vocab = None

mero = []
hyper = []
random = []

with open('../data/BLESS/bless.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('../data/vec.txt')

with tf.Session() as sess:
    # lossGraph, placeholder = batchSentenceLossGraph(vocab)
    # minLossIdxGraph = tf.argmin(lossGraph, 0)

    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    # labelPlaceholder = tf.placeholder(dtype=tf.float64)
    # difference = 10 - labelPlaceholder - distance

    tf.global_variables_initializer().run()

    for i in data:
        word = data['w']
        if vocab.getWord(word):
            for m in data['mero']:
                if vocab.getWord(m):
                    mero.append([])

    print('Data size:', len(data))
