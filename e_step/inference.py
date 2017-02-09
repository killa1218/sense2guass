# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from loss import *
from options import Options as opt
from tqdm import tqdm
import tensorflow as tf
from loss import skipGramWindowLoss
# from tensorflow import sigmoid as act
# from tensorflow import tanh as act
# import tensorflow.nn.relu as act
# from utils.distance import dist
import pprint

pp = pprint.PrettyPrinter(indent = 4)

def rdfs(stc, n, sLabel, assign, sess):
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

    stack = [0] * fullWindowSize

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
                stack += [0] * (fullWindowSize - len(stack))
                loss = sess.run(skipGramWindowLoss(stcW, stack, mid))

                print('\tASSIGN:', stack, 'LOSS:', loss)

                yield stack, loss, mid


def dpInference(stcW, vocab, sess):
    v = {}  # Record Intermediate Probability
    assign = []  # Result of word senses in a sentence
    minLoss = float('inf')  # Minimum loss

    assert len(stcW) > opt.windowSize
    for a, l, m in dfs(stcW, opt.windowSize, sess):
        v[tuple(a)] = l

        if l < minLoss:
            minLoss = l
            assign = a[:]

    for i in tqdm(range(opt.windowSize + 1, len(stcW))):
        minLoss = float('inf')  # Minimum loss
        newWord = stcW[i + opt.windowSize] if i + opt.windowSize < len(stcW) else None
        tmpV = {}

        for j in v:
            prevAssign = list(j)
            prevLoss = v[j]

            if newWord:
                for j in range(0, newWord.senseNum):
                    curAssign = prevAssign + [j]
                    curLoss = prevLoss + sess.run(skipGramWindowLoss(stcW, curAssign, i))

                    tmpV[tuple(curAssign)] = curLoss

                    if curLoss < minLoss:
                        minLoss = curLoss
                        assign = curAssign[:]
                        print('\tASSIGN:', assign, 'LOSS:', curLoss)
            else:
                curLoss = prevLoss + sess.run(skipGramWindowLoss(stcW, prevAssign, i))

                tmpV[tuple(prevAssign)] = curLoss

                if curLoss < minLoss:
                    minLoss = curLoss
                    assign = prevAssign[:]

        del(v)
        v = {}

        for j in tmpV:
            if j[i - opt.windowSize - 1] == assign[i - opt.windowSize - 1]:
                v[j] = tmpV[j]

    return assign
