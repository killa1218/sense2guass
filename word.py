#!/usr/bin/python
# coding:utf-8

from __future__ import print_function

from options import Options as opt
import tensorflow as tf

class Word(object):
    """Word with multiple senses"""
    def __init__(self, word, index, sNum = 1):
        self.token = word
        self.senseNum = sNum
        self.count = 1
        self.means = None
        self.sigmas = None
        self.index = index


    def setSenseNum(self, num):
        self.senseNum = num

        return self


    def initSenses(self):
        self.senseCount = [0] * self.senseNum
        self.senseP = [1. / self.senseNum] * self.senseNum

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


    def refreshPossiable(self):
        for i in range(self.senseNum):
            self.senseP[i] = float(self.senseCount[i]) / self.count
