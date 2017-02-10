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


def assignDFS(stcW):
    l = len(stcW)

    stack = [0] * l

    yield stack

    while True:
        if (len(stack) == 0):
            break
        else:
            if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
                stack.pop()
            else:
                stack[-1] += 1
                stack += [0] * (l - len(stack))

                yield stack


def violentInference(stcW, sess):
    ''' Inference the senses using DFS '''
    assert len(stcW) > opt.windowSize

    assignPoint = 0
    minLoss = float('inf')
    lossTensorList = []
    assignList = []

    for a in assignDFS(stcW):
        assignList.append(a)
        lossTensorList.append(skipGramLoss(stcW, a))

    lossList = sess.run(lossTensorList)

    for i in range(len(lossList)):
        if lossList[i] < minLoss:
            minLoss = lossList[i]
            assignPoint = i

    return assignList[assignPoint]


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

    # yield stack, sess.run(skipGramWindowLoss(stcW, stack, mid)), mid
    yield stack, skipGramWindowLoss(stcW, stack, mid), mid

    while True:
        if (len(stack) == 0):
            break
        else:
            if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
                stack.pop()
            else:
                stack[-1] += 1
                stack += [0] * (fullWindowSize - len(stack))
                # loss = sess.run(skipGramWindowLoss(stcW, stack, mid))
                loss = skipGramWindowLoss(stcW, stack, mid)

                # print('\tASSIGN:', stack, 'LOSS:', loss)

                yield stack, loss, mid


def dpInference(stcW, sess):
    v = {}  # Record Intermediate Probability
    tmpV = None
    assign = []  # Result of word senses in a sentence
    # minLoss = float('inf')  # Minimum loss

    assert len(stcW) > opt.windowSize
    print('Initializing first words...')
    for a, l, m in dfs(stcW, opt.windowSize, sess):
    # for a, l, m in tqdm(dfs(stcW, opt.windowSize, sess)):
        v[tuple(a)] = l
    print('Initialize first words finished.')

    print('Inferencing other words...')
    # for i in range(opt.windowSize + 1, len(stcW)):
    for i in tqdm(range(opt.windowSize + 1, len(stcW))):
        minLoss = float('inf')  # Minimum loss
        newWord = stcW[i + opt.windowSize] if i + opt.windowSize < len(stcW) else None
        del(tmpV)
        tmpV = {}
        assignList = []
        lossTensorList = []

        # print('\tSearching the state table...')
        for j in v:
            prevAssign = list(j)
            prevLoss = v[j]
            # print('\tASSIGN:', prevAssign, 'LOSS:', prevLoss)

            if newWord:
                for k in range(0, newWord.senseNum):
                    curAssign = prevAssign + [k]
                    # start = time.time()
                    curLoss = prevLoss + skipGramWindowLoss(stcW, curAssign, i)
                    # end = time.time()
                    # print('TIME SPENT:', end - start)

                    assignList.append(curAssign)
                    lossTensorList.append(curLoss)

                    tmpV[tuple(curAssign)] = curLoss

            else:
                curLoss = prevLoss + skipGramWindowLoss(stcW, prevAssign, i)

                assignList.append(prevAssign)
                lossTensorList.append(curLoss)

                tmpV[tuple(prevAssign)] = curLoss

        # print('\tSearch state table finished.')

        del(v)
        v = {}

        lossList = sess.run(lossTensorList)

        for j in range(len(lossList)):
            if lossList[j] < minLoss:
                minLoss = lossList[j]
                assign = assignList[j][:]

        del(assignList)
        del(lossTensorList)

        for j in tmpV:
            if j[i - opt.windowSize - 1] == assign[i - opt.windowSize - 1]:
                v[j] = tmpV[j]

    return assign
