# coding=utf8

import sys
import math
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from vocab import Vocab
from utils.distance import diagKL as dist
# from utils.distance import diagEL as dist

data = None
vocab = None

date = '0407'
condition = '_Adagrad'

similarthd = 60

mero = []
hyper = []
random = []

with open('../data/BLESS/bless.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('/mnt/dataset/sense2gauss/data/gauss.KL.' + date + '_w3_b50_m500' + condition + '.pkl3')

with tf.Session() as sess, open('../data/BLESS/result.mero_' + date + condition + '.txt', 'w') as mf, open('../data/BLESS/result.hyper_' + date + condition + '.txt', 'w') as hf, open('../data/BLESS/result.random_' + date + condition + '.txt', 'w') as rf:
    # lossGraph, placeholder = batchSentenceLossGraph(vocab)
    # minLossIdxGraph = tf.argmin(lossGraph, 0)

    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    reversedDistance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]))
    difference = distance - reversedDistance
    minLossIdx = tf.argmin(distance, 0)

    tf.global_variables_initializer().run()

    coveredWordNum = 0
    totalMeroNum = 0

    hyperNum = 0
    totalHyperNum = 0
    rightHyperNum = 0

    randNum = 0
    totalRandNum = 0
    rightRandNum = 0

    # data structure
    # [{
    #     'w': String, # Word
    #     'c': String, # Category
    #     'hyper': [], # Hypernym
    #     'mero': [], # Contains
    #     'r': [], # Random
    # }, ]

    for i in data:
        word = vocab.getWord(i['w'])
        if word:
            coveredWordNum += 1
            for m in i['mero']:
                w = vocab.getWord(m)
                if w:
                    tmpList = []
                    for k in range(word.senseStart, word.senseStart + word.senseNum):
                        for j in range(w.senseStart, w.senseStart + w.senseNum):
                            tmpList.append([k, j])

                    distances, reversedDistances, minIdx, differences = sess.run([distance, reversedDistance, minLossIdx, difference], feed_dict={sensePlaceholder: tmpList})

                    for k in range(len(distances)):
                        mf.write('%s\t%s\t%.4f\t%.4f\n' % (word.token, w.token, distances[k], reversedDistances[k]))

            for m in i['hyper']:
                hyperNum += 1
                w = vocab.getWord(m)
                if w:
                    totalHyperNum += 1
                    tmpList = []
                    for k in range(word.senseStart, word.senseStart + word.senseNum):
                        for j in range(w.senseStart, w.senseStart + w.senseNum):
                            tmpList.append([k, j])

                    distances, reversedDistances, minIdx, differences = sess.run([reversedDistance, distance, minLossIdx, difference], feed_dict={sensePlaceholder: tmpList})

                    for k in range(len(distances)):
                        hf.write('%s\t%s\t%.4f\t%.4f\n' % (w.token, word.token, distances[k], reversedDistances[k]))

                        if differences[k] > 0 and distances[k] < similarthd and reversedDistances[k] < similarthd:
                            # print(distances[k], reversedDistances[k])
                            rightHyperNum += 1
                            break

            for m in i['r']:
                randNum += 1
                w = vocab.getWord(m)
                if w:
                    totalRandNum += 1
                    tmpList = []
                    for k in range(word.senseStart, word.senseStart + word.senseNum):
                        for j in range(w.senseStart, w.senseStart + w.senseNum):
                            tmpList.append([k, j])

                    distances, reversedDistances, minIdx, differences = sess.run([reversedDistance, distance, minLossIdx, difference], feed_dict={sensePlaceholder: tmpList})

                    for k in range(len(distances)):
                        rf.write('%s\t%s\t%.4f\t%.4f\n' % (w.token, word.token, distances[k], reversedDistances[k]))

                        if distances[k] > similarthd and reversedDistances[k] > similarthd:
                            # print(distances[k], reversedDistances[k])
                            rightRandNum += 1
                            break

    print('Data size:', len(data))
    print('Encountered data size:', coveredWordNum)

    recall = float(totalHyperNum) / hyperNum
    precision = float(rightHyperNum) / totalHyperNum
    print('\nTotal Hyper Num:', totalHyperNum)
    print('Right Hyper Num:', rightHyperNum)
    print('Recall:', recall)
    print('Precision:', precision)
    print('F1:', 2 * recall * precision / (recall + precision))


    recall = float(totalRandNum) / randNum
    precision = float(rightRandNum) / totalRandNum
    print('\nTotal Rand Num:', totalRandNum)
    print('Right Rand Num:', rightRandNum)
    print('Recall:', float(totalRandNum) / randNum)
    print('Precision:', float(rightRandNum) / totalRandNum)
    print('F1:', 2 * recall * precision / (recall + precision))
