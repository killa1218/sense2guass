# coding=utf8

from __future__ import print_function
from __future__ import division

from options import Options as opt
import tensorflow as tf
# from tensorflow.nn import relu as act
# from tensorflow import sigmoid as act
# from tensorflow import tanh as act
from utils.distance import dist
import time

import random
import math

act = tf.nn.relu

def skipGramWindowLoss(stc, sLabel, mid):
    sum = tf.constant(0., dtype=tf.float64)

    for i in range(1, opt.windowSize + 1):
        if mid - i > -1:
            # print('mid:', mid, 'i:', i, 'Distance between', stc[mid].token, 'and', stc[mid - i].token, 'is:', sess.run(act(dist(
            #         (stc[mid], sLabel[mid]),
            #         (stc[mid - i], sLabel[mid - i])
            #     ))))

            sum += act(
                dist(
                    (stc[mid], sLabel[mid]),
                    (stc[mid - i], sLabel[mid - i])
                ),
                name="loss-ActivationFunction"
            )
        if mid + i < len(stc):
            # print('mid:', mid, 'i:', i, 'Distance between', stc[mid].token, 'and', stc[mid - i].token, 'is:', sess.run(act(dist(
            #         (stc[mid], sLabel[mid]),
            #         (stc[mid + i], sLabel[mid + i])
            #     ))))

            sum += act(
                dist(
                    (stc[mid],
                     sLabel[mid]),
                    (stc[mid + i],
                     sLabel[mid + i])
                ),
                name="loss-ActivationFunction"
            )

    return tf.clip_by_value(sum, tf.float64.min, tf.float64.max)


def skipGramSepWindowLoss(stc, sLabel, mid):
    res = []

    for i in range(1, opt.windowSize + 1):
        if mid - i > -1:
            res.insert(0, act(dist((stc[mid], sLabel[mid]), (stc[mid - i], sLabel[mid - i])), name="loss-ActivationFunction"))
        if mid + i < len(stc):
            res.append(act(dist((stc[mid], sLabel[mid]), (stc[mid + i], sLabel[mid + i])), name="loss-ActivationFunction"))

    return res


def skipGramLoss(stc, sLabel):
    '''Return the Tensorflow graph of skip-gram score of the sentence, the sentence is an array of Word()'''
    totalSum = tf.constant(0., dtype=tf.float64)

    for i in range(len(stc)):
        tmpSum = tf.constant(0., dtype=tf.float64)

        for offset in range(1, opt.windowSize + 1):
            if i - offset > -1:
                tmpSum += act(dist((stc[i], sLabel[i]), (stc[i - offset], sLabel[i - offset])), name="loss-ActivationFunction")
            if i + offset < len(stc):
                tmpSum += act(dist((stc[i], sLabel[i]), (stc[i + offset], sLabel[i + offset])), name="loss-ActivationFunction")

        totalSum += tmpSum / (((i + opt.windowSize) if i + opt.windowSize < len(stc) else (len(stc) - 1)) - ((i - opt.windowSize) if i - opt.windowSize > -1 else 0))   # Full window
        # totalSum += tmpSum / ((i if i == 0 else i) if i < opt.windowSize else opt.windowSize)   # Half window

    return totalSum


def avgSkipGramLoss(stc, sLabel):
    return skipGramLoss(stc, sLabel) / len(stc)


def skipGramNCELoss(stc, sLabel, vocab, sess):
    '''Return the Tensorflow graph of skip-gram score with negative sampling of the sentence, the sentence is an array of Word()'''
    assert len(sLabel) == len(stc)

    totalSum = tf.constant(0., dtype=tf.float64)
    margin = tf.constant(opt.margin, dtype=tf.float64)

    for i in range(len(stc)):
        for offset in range(1, opt.windowSize + 1):
            try:
                if i - offset > -1:
                    sampleWord = stc[i]
                    while sampleWord is stc[i]:
                        sampleWord = vocab.getWord(random.randint(0, vocab.size - 1))

                    totalSum += tf.nn.relu(margin - act(dist((stc[i], sLabel[i]), (stc[i - offset], sLabel[i - offset]))) + act(dist((stc[i], sLabel[i]), (sampleWord, 0))), name="loss-NCEMarginLoss")

                if i + offset < len(stc):
                    sampleWord = stc[i]
                    while sampleWord is stc[i]:
                        sampleWord = vocab.getWord(random.randint(0, vocab.size - 1))

                    totalSum += tf.nn.relu(margin - act(dist((stc[i], sLabel[i]), (stc[i + offset], sLabel[i + offset]))) + act(dist((stc[i], sLabel[i]), (sampleWord, 0))), name="loss-NCEMarginLoss")
            except IndexError:
                print('ERROR:', 'len-', len(stc), 'i-', i, 'offset-', offset, 'stack-', sLabel)


        # totalSum += tmpSum / (((i + opt.windowSize) if i + opt.windowSize < len(stc) else (len(stc) - 1)) - ((i - opt.windowSize) if i - opt.windowSize > -1 else 0))   # Full window
        # totalSum += tmpSum / ((i if i == 0 else i) if i < opt.windowSize else opt.windowSize)   # Half window

    return tf.clip_by_value(totalSum, tf.float64.min, tf.float64.max)
