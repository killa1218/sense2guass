# coding=utf8

from __future__ import print_function

import tensorflow as tf
from math import pi

def diagEL(m1, sig1, m2, sig2, d):                  # EL energy of two diagnal gaussian distributions
    m = m1 - m2
    sig = sig1 + sig2

    return tf.div(
        tf.exp(tf.reduce_sum(-0.5 * tf.pow(m, 2) * tf.div(1., sig, name='diagEL-Exponential-Inverse')), name='diagEL-Exponential'),
        tf.sqrt(tf.pow(2. * pi, d) * tf.reduce_prod(sig), name='diagEL-SquareRoot'),
        name='diagEL'
    )

def diagKL(m1, sig1, m2, sig2, d):                  # KL energy of two diagnal gaussian distributions
    m = m2 - m1

    return tf.div(
        tf.log(tf.div(tf.reduce_prod(sig2), tf.reduce_prod(sig1))) -
        d +
        tf.reduce_sum(tf.div(1., sig2, name='diagKL-Inverse') * sig1, name='diagKL-Trace') +
        tf.reduce_sum(tf.pow(m, 2) * tf.div(1., sig2, name='diagKL-Inverse')),
        2.,
        name='diagKL'
    )

def EL(m1, sig1, m2, sig2, d):                      # TODO
    pass

def KL(m1, sig1, m2, sig2, d):                      # TODO
    pass

def meanDist(m1, m2):
    return tf.reduce_sum(m1 * m2)

if __name__ == '__main__':
    sess = tf.Session()
    m1 = tf.constant([1,2,3,4,5,6,7,8,9], dtype=tf.float32)
    m2 = tf.constant([0,1,2,3,4,5,6,7,8], dtype=tf.float32)

    sig1 = tf.constant([1,1,2,4,5,4,7,8,9], dtype=tf.float32)
    sig2 = tf.constant([1,2,2,4,5,8,7,3,9], dtype=tf.float32)

    print('diagEL: ', sess.run(diagEL(m1, sig1, m2, sig2, 9)))
    print('diagKL: ', sess.run(diagKL(m1, sig1, m2, sig2, 9)))
