#!/usr/bin/python
# coding:utf-8

from __future__ import print_function

from options import Options as opt
import tensorflow as tf

class Word(object):
    """Word with multiple senses"""
    def __init__(self, word, index, sNum = 1):
        self.token = word                   # String token of the word
        self.senseNum = sNum                # How many senses does this word have
        self.count = 1                      # Word count
        self.means = None                   # Means of senses
        self.sigmas = None                  # Covariance of senses
        self.index = index                  # The index in vocabulary


    def setSenseNum(self, num):
        self.senseNum = num

        return self


    def initSenses(self):
        self.senseCount = [0] * self.senseNum   # How many times does each sense appears in corpus
        self.senseP = [1. / self.senseNum] * self.senseNum  # The prior probability of each senses, calculated by senseCount / count

        self.means = tf.Variable(
            tf.random_uniform(
                [self.senseNum, opt.embSize],
                -opt.initWidth,
                opt.initWidth
            ),
            dtype=tf.float32,
            name="means"
        )

        if opt.covarShape == 'normal':
            self.sigmas = tf.Variable(
                tf.random_uniform(
                    [self.senseNum, opt.embSize, opt.embSize],
                    -opt.initWidth,
                    opt.initWidth
                ),
                dtype=tf.float32,
                name="sigmas"
            )
        elif opt.covarShape == 'diagnal':
            self.sigmas = tf.Variable(
                tf.random_uniform(
                    [self.senseNum, opt.embSize],
                    -opt.initWidth,
                    opt.initWidth
                ),
                dtype=tf.float32,
                name="sigmas"
            )
        else:
            self.sigmas = None

        return self


    def senseFind(self, sn):
        self.senseCount[sn] += 1


    def refreshPossible(self):
        for i in range(self.senseNum):
            self.senseP[i] = float(self.senseCount[i]) / self.count


    def getSensePossible(self, sN):
        return self.senseP[sN]
