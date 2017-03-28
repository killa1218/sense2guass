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

    return tf.div(
        tf.exp(tf.reduce_sum(tf.constant(-0.5, dtype=dataType) * tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1), name='diagEL-Exponential'),
        tf.sqrt(tf.reduce_prod(tf.constant(2., dtype=dataType) * pi * sig, 1), name='diagEL-SquareRoot'),
        name='diagEL'
    )

def diagpowEL(m1, sig1, m2, sig2, d=opt.embSize):
    return -tf.pow(EL(m1, sig1, m2, sig2, d), 0.001)

def diagEL(m1, sig1, m2, sig2, d=opt.embSize):
    m = m1 - m2
    sig = sig1 + sig2

    return tf.log(tf.reduce_prod(sig, 1)) + tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1)

def diagKL(m1, sig1, m2, sig2, d=opt.embSize):   # KL energy of two diagnal gaussian distributions
    m = m2 - m1
    sum = tf.log(tf.reduce_prod(sig2 / sig1, 1))
    sum += -d
    sum += tf.reduce_sum(sig1 / sig2 + tf.square(m) / sig2, 1)

    res = tf.div(sum, 2., name='diagKL')

    return res

def crossEntropy():
    pass

def meanDist(m1, sig1, m2, sig2, d=opt.embSize):
    return tf.reduce_sum(m1 * m2)

def dist(w1, s1, w2, s2):
    start = time.time()

    res = diagKL(w1.getMean(s1), w1.getSigma(s1), w2.getMean(s2), w2.getSigma(s2), opt.embSize)

    end = time.time()
    # print('dist time:', end - start)
    return res


if __name__ == '__main__':
    sess = tf.Session()
    m1 = tf.constant([1,2,3,4,5,6,7,8,9], dtype=dataType)
    m2 = tf.constant([1,2,3,4,5,6,7,8,9], dtype=dataType)
    # m2 = tf.constant([0,1,2,3,4,5,6,7,8], dtype=dataType)

    sig1 = tf.constant([1,1,2,4,5,4,7,8,9], dtype=dataType)
    sig2 = tf.constant([1,1,2,4,5,4,7,8,9], dtype=dataType)
    # sig2 = tf.constant([1,2,2,4,5,8,7,3,9], dtype=dataType)

    print('diagEL: ', sess.run(diagEL(m1, sig1, m2, sig2, 9)))
    print('diagKL: ', sess.run(diagKL(m1, sig1, m2, sig2, 9)))


# def diagELSingle(m1, sig1, m2, sig2, d=opt.embSize):                  # EL energy of two diagnal gaussian distributions
#     m = m1 - m2
#     sig = sig1 + sig2
#
#     return 1 - tf.div(
#         tf.exp(tf.reduce_sum(tf.constant(-0.5, dtype=dataType) * tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse')), name='diagEL-Exponential'),
#         tf.sqrt(tf.pow(tf.constant(2., dtype=dataType) * pi, d) * tf.reduce_prod(sig), name='diagEL-SquareRoot'),
#         name='diagEL'
#     )


# def diagKLSingle(m1, sig1, m2, sig2, d=opt.embSize):   # KL energy of two diagnal gaussian distributions
#     start = time.time()
#
#     m = m2 - m1
#     # sig2Reciprocal = tf.reciprocal(sig2, name='diagKL-Reciprocal')
#
#     # res = tf.div(
#     #     tf.add_n([tf.log(tf.reduce_prod(sig2 / sig1)), tf.constant(-d, dtype=dataType), tf.reduce_sum(sig2Reciprocal * sig1 + tf.square(m) * sig2Reciprocal)]),
#     #     2.,
#     #     name='diagKL'
#     # )
#
#     res = tf.div(
#         tf.add_n([tf.log(tf.reduce_prod(sig2 / sig1)), tf.constant(-d, dtype=dataType), tf.reduce_sum(sig1 / sig2 + tf.square(m) / sig2)]),
#         2.,
#         name='diagKL'
#     )
#
#     end = time.time()
#     # print('diagKL time:', end - start)
#     return res

