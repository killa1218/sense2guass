#! usr/bin/python
# -*- coding:utf-8 -*-

from __future import print_function

import tensorflow as tf

embds = tf.Variable(
    tf.random_uniform([vocabSize, embdSize], -1.0, 1.0)
)
