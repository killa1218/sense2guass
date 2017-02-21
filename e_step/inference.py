# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from loss import *
from options import Options as opt
import tensorflow as tf


# def rdfs(stc, n, sLabel, assign, sess):
#     if n == len(stc):
#         return
#
#     # with tf.Session() as sess:
#     for i in range(stc[n].senseNum):
#         sLabel[n] = i
#         cur = sess.run(avgSkipGramLoss(stc, sLabel))
#         if cur < minLoss:
#             minLoss = cur
#             assign = sLabel
#         rdfs(stc, n + 1, sLabel, assign, sess)


def assign2SenseIdx(stcW, assign):
    res = []

    for i in range(len(assign)):
        res.append(stcW[i].senseStart + assign[i])

    return res


def senseIdxDFS(stcW):
    l = len(stcW)

    stack = [0] * l

    yield assign2SenseIdx(stcW, stack)

    while True:
        if (len(stack) == 0):
            break
        else:
            if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
                stack.pop()
            else:
                stack[-1] += 1
                stack += [0] * (l - len(stack))

                yield assign2SenseIdx(stcW, stack)


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


def violentInference(stcW, sess, minLossIdxGraph, senseIdxPlaceholder):
    ''' Inference the senses using DFS '''
    # assert len(stcW) == opt.sentenceLength
    senseIdxList = []

    for sIdx in senseIdxDFS(stcW):
        senseIdxList.append(sIdx)

    if len(senseIdxList) > 100000:
        reducedList = []

        for i in range(0, len(senseIdxList), 100000):
            subSenseIdxList = senseIdxList[i:i + 100000]
            subMinIdx = sess.run(minLossIdxGraph, feed_dict={senseIdxPlaceholder: subSenseIdxList})
            reducedList.append(senseIdxList[subMinIdx])

        senseIdxList = reducedList

    minLossSeqIdx = sess.run(minLossIdxGraph, feed_dict={senseIdxPlaceholder: senseIdxList})

    return senseIdxList[minLossSeqIdx]


# def dfs(stcW, mid, sess):
#     l = len(stcW)
#     fullWindowSize = 0
#
#     if l < opt.windowSize * 2 + 1:
#         fullWindowSize = l
#     elif mid + opt.windowSize >= l:
#         fullWindowSize = l - mid + opt.windowSize
#     elif mid - opt.windowSize < 0:
#         fullWindowSize = mid + opt.windowSize
#     else:
#         fullWindowSize = opt.windowSize * 2 + 1
#
#     stack = [0] * fullWindowSize
#
#     yield stack, sess.run(skipGramWindowLoss(stcW, stack, mid)), mid
#     # yield stack, skipGramWindowLoss(stcW, stack, mid), mid
#
#     while True:
#         if (len(stack) == 0):
#             break
#         else:
#             if stack[-1] == stcW[len(stack) - 1].senseNum - 1:
#                 stack.pop()
#             else:
#                 stack[-1] += 1
#                 stack += [0] * (fullWindowSize - len(stack))
#                 loss = sess.run(skipGramWindowLoss(stcW, stack, mid))
#                 # loss = skipGramWindowLoss(stcW, stack, mid)
#
#                 # print('\tASSIGN:', stack, 'LOSS:', loss)
#
#                 yield stack, loss, mid


# window = skipGramWindowKLLossGraph()


# def dpInference(stcW, sess):
#     global window
#     v = {}  # Record Intermediate Probability
#     tmpV = None
#     assign = []  # Result of word senses in a sentence
#     # minLoss = float('inf')  # Minimum loss
#
#     assert len(stcW) > opt.windowSize
#     print('Initializing first words...')
#     for a, l, m in dfs(stcW, opt.windowSize, sess):
#     # for a, l, m in tqdm(dfs(stcW, opt.windowSize, sess)):
#         v[tuple(a)] = l
#     print('Initialize first words finished.')
#
#     print('Inferencing other words...')
#     # for i in range(opt.windowSize + 1, len(stcW)):
#     for i in tqdm(range(opt.windowSize + 1, len(stcW))):
#         minLoss = float('inf')  # Minimum loss
#         newWord = stcW[i + opt.windowSize] if i + opt.windowSize < len(stcW) else None
#         del(tmpV)
#         tmpV = {}
#         assignList = []
#         lossTensorList = []
#
#         midList = []
#         otherList = []
#
#
#
#         # print('\tSearching the state table...')
#         for j in v:
#             prevAssign = list(j)
#             prevLoss = v[j]
#             # print('\tASSIGN:', prevAssign, 'LOSS:', prevLoss)
#
#             prevSenseIdx = []
#             for k in range(1, opt.windowSize + 1):
#                 if i - k < 0:
#                     prevSenseIdx.append(stcW[i].senseStart + prevAssign[i])
#                 else:
#                     prevSenseIdx.append(stcW[i - k].senseStart + prevAssign[i - k])
#                 if i + k < len(stcW):
#                     if k < opt.windowSize:
#                         prevSenseIdx.append(stcW[i + k].senseStart + prevAssign[i + k])
#                 else:
#                     prevSenseIdx.append(stcW[i].senseStart + prevAssign[i])
#
#             if newWord:
#                 for k in range(0, newWord.senseNum):
#                     curAssign = prevAssign + [k]
#                     # start = time.time()
#                     # curLoss = prevLoss + skipGramWindowLoss(stcW, curAssign, i)
#                     # end = time.time()
#                     # print('TIME SPENT:', end - start)
#
#                     assignList.append(curAssign)
#                     # lossTensorList.append(curLoss)
#                     midList.append(stcW[i].senseStart + curAssign[i])
#                     otherList.append(prevSenseIdx + [stcW[i + opt.windowSize].senseStart + k])
#
#
#                     # tmpV[tuple(curAssign)] = curLoss
#
#             else:
#                 # curLoss = prevLoss + skipGramWindowLoss(stcW, prevAssign, i)
#
#                 assignList.append(prevAssign)
#                 # lossTensorList.append(curLoss)
#                 midList.append(stcW[i].senseStart + curAssign[i])
#                 otherList.append(prevSenseIdx)
#
#                 # tmpV[tuple(prevAssign)] = curLoss
#
#         # print('\tSearch state table finished.')
#
#         del(v)
#         v = {}
#
#         # lossList = sess.run(lossTensorList)
#
#         # for j in range(len(lossList)):
#         #     if lossList[j] < minLoss:
#         #         minLoss = lossList[j]
#         #         assign = assignList[j][:]
#
#         lossList = sess.run(window, feed_dict={mid: midList, others: otherList})
#
#         for j in range(len(lossList)):
#             tmpV[tuple(assignList[j])] = lossList[j]
#
#             if lossList[j] < minLoss:
#                 minLoss = lossList[j]
#                 assign = assignList[j][:]
#
#         del(assignList)
#         del(lossTensorList)
#
#         for j in tmpV:
#             if j[i - opt.windowSize - 1] == assign[i - opt.windowSize - 1]:
#                 v[j] = tmpV[j]
#
#     return assign
