# coding=utf8

from __future__ import print_function

import tensorflow as tf
from options import Options as opt
from math import pi

def diagEL(m1, sig1, m2, sig2, d):                  # EL energy of two diagnal gaussian distributions
    m = m1 - m2
    sig = sig1 + sig2

    return tf.div(
        tf.exp(tf.reduce_sum(tf.constant(-0.5, dtype=tf.float64) * tf.pow(m, 2) * tf.div(tf.constant(1., dtype=tf.float64), sig, name='diagEL-Exponential-Inverse')), name='diagEL-Exponential'),
        tf.sqrt(tf.pow(tf.constant(2., dtype=tf.float64) * pi, d) * tf.reduce_prod(sig), name='diagEL-SquareRoot'),
        name='diagEL'
    )

def diagKL(m1, sig1, m2, sig2, d):                  # KL energy of two diagnal gaussian distributions
    m = m2 - m1

    return tf.div(
        tf.log(tf.reduce_prod(sig2 / sig1)) -
        d +
        tf.reduce_sum(tf.div(tf.constant(1., dtype=tf.float64), sig2, name='diagKL-Inverse') * sig1, name='diagKL-Trace') +
        tf.reduce_sum(tf.pow(m, 2) * tf.div(tf.constant(1., dtype=tf.float64), sig2, name='diagKL-Inverse')),
        2.,
        name='diagKL'
    )

def crossEntropy():
    pass

def EL(m1, sig1, m2, sig2, d):                      # TODO
    pass

def KL(m1, sig1, m2, sig2, d):                      # TODO
    pass

def meanDist(m1, m2):
    return tf.reduce_sum(m1 * m2)

def dist(word1, word2):
    (w1,s1) = word1
    (w2, s2) = word2
    # return tf.reduce_sum(w1.means[s1])

    return diagKL(w1.means[s1], w1.sigmas[s1], w2.means[s2], w2.sigmas[s2], opt.embSize)


if __name__ == '__main__':
    sess = tf.Session()
    m1 = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float32)
    m2 = tf.constant([0,1,2,3,4,5,6,7,8], dtype=tf.float32)

    sig1 = tf.constant([1,1,2,4,5,4,7,8,9], dtype=tf.float32)
    sig2 = tf.constant([1,2,2,4,5,8,7,3,9], dtype=tf.float32)

    print('diagEL: ', sess.run(diagEL(m1, sig1, m2, sig2, 9)))
    print('diagKL: ', sess.run(diagKL(m1, sig1, m2, sig2, 9)))

