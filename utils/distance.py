# coding=utf8

from __future__ import print_function

import tensorflow as tf
import math
from options import Options as opt

dataType = opt.dType

def diagEL(m1, sig1, m2, sig2, d=opt.embSize):
    m = m1 - m2
    sig = sig1 + sig2
    fac = math.sqrt((2 * math.pi)**opt.embSize)

    firstTerm = tf.log(tf.reduce_prod(sig, 1))
    secondTerm = tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1)

    tf.add_to_collection('LOG_EL_FIRST', firstTerm)
    tf.add_to_collection('LOG_EL_SECOND', secondTerm)

    # return tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1) # Only l2 norm
    return (firstTerm + secondTerm + d * 1.83787706641) / 2 # log
    # return -tf.exp(-tf.reduce_sum(tf.square(m) * tf.reciprocal(sig, name='diagEL-Exponential-Inverse'), 1) / 2) / tf.sqrt(tf.reduce_prod(sig, 1)) # nolog

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
    with tf.name_scope("Inner_Product"):
        return tf.reduce_sum(m1 * m2, 1)

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
