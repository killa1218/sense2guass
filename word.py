#!/usr/bin/python
# coding:utf-8

from __future__ import print_function

# from options import Options as opt
import tensorflow as tf

class Word():
    """Word with multiple senses"""
    def __init__(self, word, index=-1, sNum=1, c=1, sStart=None):
        self.token = word                   # String token of the word
        self.senseNum = sNum                # How many senses does this word have
        self.count = c                      # Word count
        self.means = None                   # Means of senses
        self.sigmas = None                  # Covariance of senses
        self.index = index                  # The index in vocabulary
        self.senseStart = sStart            # Where does the senses starts


    def setSenseStart(self, sStart):
        self.senseStart = sStart

        return self


    def setMeans(self, means):
        self.means = means


    def setSigmas(self, sigmas):
        self.sigmas = sigmas


    def getMean(self, mIdx):
        return tf.nn.embedding_lookup(self.means, self.senseStart + mIdx - 1)


    def getSigma(self, sIdx):
        return tf.nn.embedding_lookup(self.sigmas, self.senseStart + sIdx - 1)


    def senseFind(self, sn):
        self.senseCount[sn] += 1


    def refreshPossible(self):
        for i in range(self.senseNum):
            self.senseP[i] = float(self.senseCount[i]) / self.count


    def getSensePossible(self, sN):
        return self.senseP[sN]
