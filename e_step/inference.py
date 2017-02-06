# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from loss import *
from options import Options as opt
import tensorflow as tf
from loss import skipGramWindowLoss
# from tensorflow import sigmoid as act
# from tensorflow import tanh as act
# import tensorflow.nn.relu as act
# from utils.distance import dist
import pprint

pp = pprint.PrettyPrinter(indent = 4)

minLoss = float('inf')

def stc2stcW(stc, vocab):
    tmp = []

    for s in stc:
        s = s.lower()
        w = vocab.getWord(s)

        if w != None:
            tmp.append(w)

    del(stc)
    return tmp

def rdfs(stc, n, sLabel, assign, sess):
    global minLoss

    if n == len(stc):
        return

    # with tf.Session() as sess:
    for i in range(stc[n].senseNum):
        sLabel[n] = i
        cur = sess.run(avgSkipGramLoss(stc, sLabel))
        if cur < minLoss:
            minLoss = cur
            assign = sLabel
        rdfs(stc, n + 1, sLabel, assign, sess)


def violentInference(stc, sess):
    ''' Inference the senses using DFS '''
    senseLabel = [0] * len(stc)
    assign = []

    rdfs(stc, 0, senseLabel, assign, sess)

    return assign


def dfs(stcW, mid, sess):
    l = len(stcW)
    fullWindowSize = 0

    if l < opt.windowSize * 2 + 1:
        fullWindowSize = l
    elif mid + opt.windowSize >= l:
        fullWindowSize = l - mid + opt.windowSize
    elif mid - opt.windowSize < 0:
        fullWindowSize = mid + opt.windowSize
    else:
        fullWindowSize = opt.windowSize * 2 + 1

    # with tf.Session() as sess:
    stack = [0] * fullWindowSize
    #
    # print('\n', sess.run(stcW[0].means))
    # print(sess.run(skipGramWindowLoss(stcW, stack, mid)))

    yield stack,\
          sess.run(
              skipGramWindowLoss(stcW, stack, mid)
          ),\
          mid

    while True:
        if (len(stack) == 0):
            break
        else:
            if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
                stack.pop()
            else:
                stack[-1] += 1
                stack += [0] * (len(stcW) - len(stack))
                yield stack, sess.run(skipGramWindowLoss(stcW, stack, mid)), mid


def dpInference(stc, vocab, sess):
    v = {}  # Record Intermediate Probability
    assign = []  # Result of word senses in a sentence
    stcW = stc2stcW(stc, vocab)

    for a, l, m in dfs(stcW, opt.windowSize, sess):
        # v[tuple(a)] = dict(loss = l, mid = m)
        v[tuple(a)] = l

    # with tf.Session() as sess:
    for i in range(opt.windowSize + 1, len(stcW)):
        minLoss = float('inf')  # Minimum loss
        newWord = stcW[i + opt.windowSize if i + opt.windowSize < len(stcW) else len(stcW) - 1]
        tmpV = {}

        for j in v:
            prevAssign = list(j)
            # prevLoss = v[j]['loss']
            prevLoss = v[j]

            for j in range(0, newWord.senseNum):
                curAssign = prevAssign + [j]
                curLoss = prevLoss + sess.run(skipGramWindowLoss(stcW, curAssign, i))

                tmpV[tuple(curAssign)] = curLoss

                if curAssign < minLoss:
                    minLoss = curLoss
                    assign = curAssign

        print(assign)

        del(v)

        for j in tmpV:
            if j[i - opt.windowSize - 1] == assign[i - opt.windowSize - 1]:
                v[j] = tmpV[j]

    return assign
