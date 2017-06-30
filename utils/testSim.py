# coding=utf8

from options import Options as opt
opt.energy = 'IP'
import sys
import pickle as pk
import tensorflow as tf
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import windowLossGraph
from e_step.inference import batchDPInference
from vocab import Vocab
# from utils.distance import diagKL as dist
# from utils.distance import diagEL as dist
from utils.distance import meanDist as dist
from multiprocessing import Pool

pool = Pool()
data = None
vocab = None

result = None
scoreList = None

date = '0515'
condition = '.adam.multisense'

# with open('../data/SCWS/testData.pk3', 'rb') as f:
with open('/mnt/dataset/sense2gauss/data/SCWS/testData.pk3', 'rb') as f:
    data = pk.load(f)
    vocab = Vocab()
    vocab.load('/mnt/dataset/sense2gauss/data/IP.06262002w3b20lr0.02m1.0n6adam.pkl')

with tf.Session() as sess:
    windowLossGraph, window = windowLossGraph(vocab)

    sensePlaceholder = tf.placeholder(dtype=tf.int32)
    distance = dist(tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 0]), tf.nn.embedding_lookup(vocab.means, sensePlaceholder[:, 1]), tf.nn.embedding_lookup(vocab.sigmas, sensePlaceholder[:, 1]))
    minValue = tf.argmin(distance, 0)
    maxValue = tf.argmax(distance, 0)
    labelPlaceholder = tf.placeholder(dtype=tf.float64)
    difference = 10 - labelPlaceholder - distance

    tf.global_variables_initializer().run()
    wordPairList = []
    scoreList = []

    wTime = 0
    cTime = 0
    iTime = 0

    for i in data:
        stcW = []
        word1 = i['w1']
        if vocab.getWord(word1):
            stcW.append(vocab.getWord(word1))
            try:
                w1sIdx = i['c1'].index(word1)
            except:
                print(i)
            j = 1

            while len(stcW) < opt.sentenceLength:
                if len(stcW) < opt.sentenceLength and w1sIdx - j >= 0 and vocab.getWord(i['c1'][w1sIdx - j]):
                    stcW.insert(0, vocab.getWord(i['c1'][w1sIdx - j]))
                if len(stcW) < opt.sentenceLength and w1sIdx + j < len(i['c1']) and vocab.getWord(i['c1'][w1sIdx + j]):
                    stcW.append(vocab.getWord(i['c1'][w1sIdx + j]))
                j += 1

            for k in range(len(stcW)):
                if stcW[k].token == word1:
                    w1sIdx = k
                    break

            assign1 = batchDPInference([stcW], sess, windowLossGraph, window, pool)

            stcW = []
            word2 = i['w2']
            if vocab.getWord(word2):
                stcW.append(vocab.getWord(word2))
                w2sIdx = i['c2'].index(word2)
                j = 1

                while len(stcW) < opt.sentenceLength:
                    if len(stcW) < opt.sentenceLength and w2sIdx - j >= 0 and vocab.getWord(i['c2'][w2sIdx - j]):
                        stcW.insert(0, vocab.getWord(i['c2'][w2sIdx - j]))
                    if len(stcW) < opt.sentenceLength and w2sIdx + j < len(i['c2']) and vocab.getWord(i['c2'][w2sIdx + j]):
                        stcW.append(vocab.getWord(i['c2'][w2sIdx + j]))
                    j += 1

                for k in range(len(stcW)):
                    if stcW[k].token == word2:
                        w2sIdx = k
                        break

                assign2 = batchDPInference([stcW], sess, windowLossGraph, window, pool)

                wTime += assign1[1] + assign2[1]
                cTime += assign1[2] + assign2[2]
                iTime += assign1[3] + assign2[3]

                wordPairList.append([assign1[0][0][w1sIdx], assign2[0][0][w2sIdx]])
                scoreList.append(i['r'])

    print(wTime / len(wordPairList) / 2, cTime / len(wordPairList) / 2, iTime / len(wordPairList) / 2)

    result = sess.run(distance, feed_dict={sensePlaceholder: wordPairList})
    min = result[sess.run(minValue, feed_dict={sensePlaceholder: wordPairList})]
    max = result[sess.run(maxValue, feed_dict={sensePlaceholder: wordPairList})]
    idx = sess.run(tf.argmin(distance, 0), feed_dict={sensePlaceholder: wordPairList})

    print('Data size:', len(data), 'Data covered:', len(wordPairList), 'Recall:', float(len(wordPairList)) / len(data))
    print('minValue:', min)
    print('maxValue:', max)

dataSortList = []
resSortList = []
with open('../data/SCWS/ELResult_' + date + condition + '.txt', 'w') as f:
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

# print(dataSortList, resSortList)

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

print(dic)

sum = 0
n = len(dataSortList)
for k, v in dic.items():
    sum += (v[0] - v[1])**2

print('Spearman\'s rank correlation:', 1 - (6 * sum / n / (n**2 - 1)))

sum = 0
for k,v in ddic.items():
    sum += (v[0] - v[1])**2

print('Mean square error:', sum / n)
