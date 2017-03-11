# coding=utf8

import sys
import time
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


def batchViolentInference(batchStcW, sess, batchSentenceLossGraph, senseIdxPlaceholder, argmin, lossPlaceholder):
    ''' Inference the senses using DFS '''
    senseIdxList = []
    sepList = []
    lossList = []
    result = []

    start = time.time()

    for stcW in batchStcW:
        for sIdx in senseIdxDFS(stcW):
            senseIdxList.append(sIdx)

        sepList.append(len(senseIdxList))

    for i in range(0, len(senseIdxList), 100000):
        subSenseIdxList = senseIdxList[i:i + 100000]
        lossList.extend(list(sess.run(batchSentenceLossGraph, feed_dict={senseIdxPlaceholder: subSenseIdxList})))

    minLoss = float('inf')
    minLossIdx = 0

    end = time.time()
    # print('CALCULATE TIME:', end - start)

# TODO
    start = time.time()
    for i in range(len(senseIdxList)):
        if i == sepList[0]:
            del(sepList[0])
            result.append(senseIdxList[minLossIdx])
            minLoss = float('inf')
            minLossIdx = i
        else:
            if lossList[i] < minLoss:
                minLoss = lossList[i]
                minLossIdx = i
    end = time.time()
    # print('LIST PERFORMENCE TIME:', end - start)

    return result


def violentInferenceWithBatchGrapg(stcW, sess, batchInferenceGraph, senseIdxPlaceholder):
    ''' Inference the senses using DFS '''
    senseIdxList = []

    for sIdx in senseIdxDFS(stcW):
        senseIdxList.append(sIdx)

    if len(senseIdxList) > 100000:
        reducedList = []

        for i in range(0, len(senseIdxList), 100000):
            subSenseIdxList = senseIdxList[i:i + 100000]
            subMinIdx = sess.run(batchInferenceGraph, feed_dict={senseIdxPlaceholder: subSenseIdxList})
            reducedList.append(senseIdxList[subMinIdx])

        senseIdxList = reducedList

    minLossSeqIdx = sess.run(batchInferenceGraph, feed_dict={senseIdxPlaceholder: senseIdxList})

    return senseIdxList[minLossSeqIdx]


def dpInference(stcW, sess, windowLossGraph, window):
    runtime = 0
    ssstart = time.time()
    # Use dictionary to record medium result
    assignList = [] # Sense lists to be calculated
    lossList = []   # Loss of sense lists
    prevAssignList = []
    assign = []     # Final result of inference
    map = {}

    # DFS of senses in the first window of sentence
    tmp = []
    for i in range(opt.windowSize * 2 + 1):
        tmp.append(stcW[i].senseStart)
    assignList.append(tmp[:])
    lossList.append(0)
    prevAssignList.append(None)
    while True:
        if (len(tmp) == 0):
            break
        else:
            if tmp[-1] == stcW[len(tmp) - 1].senseStart + stcW[len(tmp) - 1].senseNum - 1:
                tmp.pop()
            else:
                tmp[-1] += 1

                for i in range(len(tmp), opt.windowSize * 2 + 1):
                    tmp.append(stcW[i].senseStart)

                assignList.append(tmp[:])
                lossList.append(0)
                prevAssignList.append(None)

    for i in range(opt.windowSize, len(stcW)):
        start = time.time()
        tmpLossList = sess.run(windowLossGraph, feed_dict = {window: assignList})
        runtime += time.time() - start
        minLoss = float('inf')
        minLossIdx = 0
        map = {}

        for j in range(len(tmpLossList)):
            lossList[j] += tmpLossList[j]
            tmpAssign = assignList[j]
            tmpLoss = lossList[j]

            if tmpLoss < minLoss:
                minLoss = lossList[j]
                minLossIdx = j

            t = tuple(tmpAssign[1:])
            if t not in map.keys():
                map[t] = [tmpAssign[0], tmpLoss, prevAssignList[j]]
            else:
                if map[t][1] > tmpLoss:
                    map[t][0] = tmpAssign[0]
                    map[t][1] = tmpLoss
                    map[t][2] = prevAssignList[j]

        if i > opt.windowSize:
            assign.append(prevAssignList[minLossIdx]) # Record the inferenced sense

        newWordIdx = i + opt.windowSize + 1
        assignList = []
        lossList = []
        prevAssignList = []
        for j in map:
            if len(assign) == 0 or map[j][2] == assign[-1]:
                tmp = list(j)

                if newWordIdx < len(stcW):
                    for k in range(stcW[newWordIdx].senseNum):
                        assignList.append(tmp + [stcW[newWordIdx].senseStart + k])
                        lossList.append(map[j][1])
                        prevAssignList.append(map[j][0])
                else:
                    assignList.append(tmp + [tmp[opt.windowSize]])
                    lossList.append(map[j][1])
                    prevAssignList.append(map[j][0])

    minLoss = float('inf')
    minLossIdx = 0
    for i in map:
        if map[i][1] < minLoss:
            minLoss = map[i][1]
            minLossIdx = i

    assign.extend([map[minLossIdx][0]])
    assign.extend(list(minLossIdx)[:opt.windowSize])

    eeend = time.time()
    # print('RUN TIME:', runtime)
    # print('OTHER TIME:', eeend - ssstart - runtime)

    return assign


def olddpInference(stcW, sess, windowLossGraph, window):
    runtime = 0
    ssstart = time.time()
    # Use list to record medium result
    assignList = [] # Sense lists to be calculated
    lossList = []   # Loss of sense lists
    assign = []     # Final result of inference

    tmp = []
    for i in range(opt.windowSize * 2 + 1):
        tmp.append(stcW[i].senseStart)
    assignList.append(tmp[:])
    lossList.append(0)

    while True:
        if (len(tmp) == 0):
            break
        else:
            if tmp[-1] == stcW[len(tmp) - 1].senseStart + stcW[len(tmp) - 1].senseNum - 1:
                tmp.pop()
            else:
                tmp[-1] += 1

                for i in range(len(tmp), opt.windowSize * 2 + 1):
                    tmp.append(stcW[i].senseStart)

                assignList.append(tmp[:])
                lossList.append(0)

    for i in range(opt.windowSize, len(stcW)):
        start = time.time()
        tmpLossList = sess.run(windowLossGraph, feed_dict = {window: assignList})
        runtime += time.time() - start
        minLoss = float('inf')
        minLossIdx = 0
        l = []
        a = []

        for j in range(len(tmpLossList)):
            lossList[j] += tmpLossList[j]

            if lossList[j] < minLoss:
                minLoss = lossList[j]
                minLossIdx = j

        assign.append(assignList[minLossIdx][0]) # Record the inferenced sense

        for j in range(len(assignList)):
            if assignList[j][0] == assign[-1]:
                newWordIdx = i + opt.windowSize + 1

                if newWordIdx < len(stcW):
                    for k in range(stcW[newWordIdx].senseNum):
                        a.append((assignList[j] + [stcW[newWordIdx].senseStart + k])[1:])
                        l.append(lossList[j])
                else:
                    a.append((assignList[j] + [assignList[j][opt.windowSize + 1]])[1:])
                    l.append(lossList[j])

        assignList = a
        lossList = l


    eeend = time.time()
    print('RUN TIME:', runtime)
    print('OTHER TIME:', eeend - ssstart - runtime)


    return assign


def getAllWindows(stcW):
    windows = []
    firstWindowSize = 0
    stcWLen = len(stcW)

    for i in range(stcWLen - opt.windowSize):
        start = i
        mid = i + opt.windowSize
        end = mid + opt.windowSize + 1
        tmp = []

        for j in range(start, stcWLen if end > stcWLen else end):
            tmp.append(stcW[j].senseStart)

        windows.append(tmp[:])

        if i == 0:
            firstWindowSize = len(tmp)

        while True:
            if len(tmp) == 0:
                break
            else:
                windowLast = start + len(tmp) - 1 if start + len(tmp) - 1 < stcWLen else stcWLen - 1

                if tmp[-1] == stcW[windowLast].senseStart + stcW[windowLast].senseNum - 1:
                    tmp.pop()
                else:
                    tmp[-1] += 1

                    for k in range(len(tmp), opt.windowSize * 2 + 1):
                        if start + k < stcWLen:
                            tmp.append(stcW[start + k].senseStart)
                        else:
                            tmp.append(mid)

                    windows.append(tmp[:])

    return windows, firstWindowSize

def inferenceOneStc(stcW, lossTable, assignList):
    lossList = [0] * len(assignList)
    prevAssignList = [None] * len(assignList)
    assign = []
    map = {}

    for i in range(opt.windowSize, len(stcW)):
        minLoss = float('inf')
        minLossIdx = 0
        map = {}

        for j in range(len(assignList)):
            lossList[j] += lossTable[tuple(assignList[j])]
            tmpAssign = assignList[j]
            tmpLoss = lossList[j]

            if tmpLoss < minLoss:
                minLoss = lossList[j]
                minLossIdx = j

            t = tuple(tmpAssign[1:])
            if t not in map.keys():
                map[t] = [tmpAssign[0], tmpLoss, prevAssignList[j]]
            else:
                if map[t][1] > tmpLoss:
                    map[t][0] = tmpAssign[0]
                    map[t][1] = tmpLoss
                    map[t][2] = prevAssignList[j]

        if i > opt.windowSize:
            assign.append(prevAssignList[minLossIdx]) # Record the inferenced sense

        newWordIdx = i + opt.windowSize + 1
        assignList = []
        lossList = []
        prevAssignList = []
        for j in map:
            if len(assign) == 0 or map[j][2] == assign[-1]:
                tmp = list(j)

                if newWordIdx < len(stcW):
                    for k in range(stcW[newWordIdx].senseNum):
                        assignList.append(tmp + [stcW[newWordIdx].senseStart + k])
                        lossList.append(map[j][1])
                        prevAssignList.append(map[j][0])
                else:
                    assignList.append(tmp + [tmp[opt.windowSize]])
                    lossList.append(map[j][1])
                    prevAssignList.append(map[j][0])

    minLoss = float('inf')
    minLossIdx = 0
    for i in map:
        if map[i][1] < minLoss:
            minLoss = map[i][1]
            minLossIdx = i

    assign.extend([map[minLossIdx][0]])
    assign.extend(list(minLossIdx)[:opt.windowSize])


def inferenceHelper(arg):
    return inferenceOneStc(*arg)


def batchDPInference(batchStcW, sess, windowLossGraph, window, pool):
    assignList = []
    assign = []
    starts = []
    ends = []
    lossTable = {}

    for i, j in pool.imap_unordered(getAllWindows, batchStcW):
        assignList.extend(i)
        starts.append(len(assignList))
        ends.append(j)

    loss = sess.run(windowLossGraph, feed_dict = {window: assignList})

    for i in range(len(loss)):
        lossTable[tuple(assignList[i])] = loss[i]

    for i in pool.imap_unordered(inferenceHelper, [(batchStcW[j], lossTable, assignList[starts[j]: ends[j]]) for j in range(len(batchStcW))]):
        assign.append(i)

    return assign


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
