# coding=utf8

import sys
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import batchSentenceLossGraph
from e_step.inference import violentInference
from vocab import Vocab


data = None
vocab = None

with open('data/SWCS/testData.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('data/vec.txt')

with tf.Session() as sess:
    lossGraph, placeholder = batchSentenceLossGraph(vocab)
    minLossIdxGraph = tf.argmin(lossGraph, 0)

    tf.global_variables_initializer().run()
    inferenceList = []

    for i in data:
        stcW = []
        word1 = i['w1']

        for w in i['c1']:
            ww = vocab.getWord(w)

            if ww:
                stcW.append(ww)

        assign = violentInference(stcW, sess, minLossIdxGraph, placeholder)
