#!/usr/bin/python
# coding:utf-8

from __future__ import print_function

from options import Options as opt
import tensorflow as tf

class Word():
    """Word with multiple senses"""
    def __init__(self, word, index=-1, sNum=1, c=1):
        self.token = word                   # String token of the word
        self.senseNum = sNum                # How many senses does this word have
        self.count = c                      # Word count
        # self.means = None                   # Means of senses
        # self.sigmas = None                  # Covariance of senses
        self.index = index                  # The index in vocabulary


    def setSenseNum(self, num):
        self.senseNum = num
        self.initSenses()

        return self


    def initSenses(self):
        # self.senseCount = [0] * self.senseNum   # How many times does each sense appears in corpus
        # self.senseP = [1. / self.senseNum] * self.senseNum  # The prior probability of each senses, calculated by senseCount / count

        sNum = self.senseNum
        eSize = opt.embSize
        iWidth = opt.initWidth

        self.means = tf.Variable(
            tf.random_uniform(
                [sNum, eSize],
                -iWidth,
                iWidth
            ),
            dtype=tf.float32,
            name="means-"+self.token,
            trainable=True
        )

        if opt.covarShape == 'normal':
            self.sigmas = tf.Variable(
                tf.random_uniform(
                    [sNum, eSize, eSize],
                    -iWidth,
                    iWidth
                ),
                dtype=tf.float32,
                name="sigmas-"+self.token,
                trainable=True
            )
        elif opt.covarShape == 'diagnal':
            self.sigmas = tf.Variable(
                tf.random_uniform(
                    [sNum, eSize],
                    -iWidth,
                    iWidth
                ),
                dtype=tf.float32,
                name="sigmas-"+self.token,
                trainable=True
            )
        else:
            self.sigmas = None


    def senseFind(self, sn):
        self.senseCount[sn] += 1


    def refreshPossible(self):
        for i in range(self.senseNum):
            self.senseP[i] = float(self.senseCount[i]) / self.count


    def getSensePossible(self, sN):
        return self.senseP[sN]
