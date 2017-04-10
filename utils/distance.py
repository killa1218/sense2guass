# coding=utf8

from __future__ import print_function

import tensorflow as tf
from options import Options as opt
from math import pi
import time

dataType = opt.dType

def EL(m1, sig1, m2, sig2, d=opt.embSize):                  # EL energy of two diagnal gaussian distributions
    m = m1 - m2
    sig = sig1 + sig2

    return tf.log(tf.div(
        tf.exp(tf.reduce_sum(tf.constant(-0.5, dtype=dataType) * tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1), name='diagEL-Exponential'),
        tf.sqrt(tf.reduce_prod(tf.constant(2., dtype=dataType) * pi * sig, 1), name='diagEL-SquareRoot'),
        name='diagEL'
    ))

def diagpowEL(m1, sig1, m2, sig2, d=opt.embSize):
    return -tf.pow(EL(m1, sig1, m2, sig2, d), 0.001)

def diagEL(m1, sig1, m2, sig2, d=opt.embSize):
    m = m1 - m2
    sig = sig1 + sig2

    # return tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1) # Only l2 norm
    return (tf.log(tf.reduce_prod(sig, 1)) + tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1) + d * 1.83787706641) / 2

def diagKL(m1, sig1, m2, sig2, d=opt.embSize):   # KL energy of two diagnal gaussian distributions
    m = m2 - m1
    sum = tf.log(tf.reduce_prod(sig2 / sig1, 1))
    sum += -d
    sum += tf.reduce_sum(sig1 / sig2 + tf.square(m) / sig2, 1)

    res = tf.div(sum, 2., name='diagKL')

    return res

def diagCE(m1, sig1, m2, sig2, d=opt.embSize):
    return diagKL(m1, sig1, m2, sig2, d) + (tf.log(tf.reduce_prod(sig1, 1)) + 2.83787706641 * d) / 2

def meanDist(m1, sig1, m2, sig2, d=opt.embSize):
    return tf.reduce_sum(m1 * m2)

def dist(w1, s1, w2, s2):
    start = time.time()

    res = diagKL(w1.getMean(s1), w1.getSigma(s1), w2.getMean(s2), w2.getSigma(s2), opt.embSize)

    end = time.time()
    # print('dist time:', end - start)
    return res


if __name__ == '__main__':
    with tf.Session() as sess:
        from vocab import Vocab
        v = Vocab()
        v.load('../data/gauss.EL.0328_w3_b50_m100.pkl3')

        tf.global_variables_initializer().run()

        sample1 = 1
        sample2 = 2

        print('diagEL: ', sess.run(diagEL(tf.nn.embedding_lookup(v.means, [sample1]), tf.nn.embedding_lookup(v.sigmas, [sample1]), tf.nn.embedding_lookup(v.means, [sample2]), tf.nn.embedding_lookup(v.sigmas, [sample2]))))
        print('diagKL: ', sess.run(diagKL(tf.nn.embedding_lookup(v.means, [sample1]), tf.nn.embedding_lookup(v.sigmas, [sample1]), tf.nn.embedding_lookup(v.means, [sample2]), tf.nn.embedding_lookup(v.sigmas, [sample2]))))
