# coding=utf8

import sys
import pickle as pk
import tensorflow as tf
from os import path
from graph import windowLossGraph
from e_step.inference import batchDPInference
from vocab import Vocab
# from utils.distance import diagKL as dist
from utils.distance import meanDist as dist
# from utils.distance import mse as dist
# from utils.distance import diagEL as dist
from options import Options as opt
from multiprocessing import Pool

pool = Pool()
data = None
vocab = None

result = None
scoreList = None

# with open('../data/SCWS/testData.pk3', 'rb') as f:
# with open('/mnt/dataset/sense2gauss/data/SCWS/testData.pk3', 'rb') as f:
with open('/mnt/dataset/sense2gauss/data/wordsim353/wordsim353.pkl', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('/mnt/dataset/sense2gauss/data/MSE.06291043w3b20lr0.02m3.0n1adam.pkl')

with tf.Session() as sess:
    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    minValue = tf.argmin(distance, 0)
    maxValue = tf.argmax(distance, 0)
    labelPlaceholder = tf.placeholder(dtype=tf.float64)
    difference = 10 - labelPlaceholder - distance

    tf.global_variables_initializer().run()
    wordPairList = []
    scoreList = []

    for p, c in data.items():
        w1 = vocab.getWord(p[0])
        w2 = vocab.getWord(p[1])

        if w1 and w2:
            w1sIdx = w1.senseStart
            w2sIdx = w2.senseStart

            wordPairList.append([w1sIdx, w2sIdx])
            scoreList.append(float(c))


    result = sess.run(distance, feed_dict={sensePlaceholder: wordPairList})
    min = result[sess.run(minValue, feed_dict={sensePlaceholder: wordPairList})]
    max = result[sess.run(maxValue, feed_dict={sensePlaceholder: wordPairList})]
    idx = sess.run(tf.argmin(distance, 0), feed_dict={sensePlaceholder: wordPairList})

    print('Data size:', len(data), 'Data covered:', len(wordPairList), 'Recall:', float(len(wordPairList)) / len(data))
    print('minValue:', min)
    print('maxValue:', max)

dataSortList = []
resSortList = []
with open('../data/SCWS/ELResult_.txt', 'w') as f:
    for i in range(len(wordPairList)):
        f.write(vocab.getWordBySenseId(wordPairList[i][0]).token)
        f.write('  ')
        f.write(vocab.getWordBySenseId(wordPairList[i][1]).token)
        f.write('\t')
        f.write(str(result[i]))
        f.write('\t')
        f.write(str(scoreList[i]))
        f.write('\n')

        dataSortList.append((float(scoreList[i]), (vocab.getWordBySenseId(wordPairList[i][0]).token, vocab.getWordBySenseId(wordPairList[i][1]).token)))
        resSortList.append((float(result[i]), (vocab.getWordBySenseId(wordPairList[i][0]).token, vocab.getWordBySenseId(wordPairList[i][1]).token)))

dataSortList.sort(key = lambda x: x[0])
resSortList.sort(key = lambda x: x[0], reverse = True)

print(dataSortList, '\n\n\n\n', resSortList)

dic = {} # Spearman's rank correlation
ddic = {} # square error
for r, i in enumerate(dataSortList):
    if i[1] in dic:
        dic[i[1]][0] = r + 1
        ddic[i[1]][0] = i[0]
    else:
        dic[i[1]] = [r + 1, 0]
        ddic[i[1]] = [i[0], 0]

for r, i in enumerate(resSortList):
    if i[1] in dic:
        dic[i[1]][1] = r + 1
        ddic[i[1]][1] = i[0] / 5
    else:
        raise Exception(i[1], ' not in!')

# print(dic)

sum = 0
n = len(dataSortList)
for k, v in dic.items():
    sum += (v[0] - v[1])**2

print('Spearman\'s rank correlation:', 1 - (6 * sum / n / (n**2 - 1)))

sum = 0
for k,v in ddic.items():
    sum += (v[0] - v[1])**2

print('Mean square error:', sum / n)
