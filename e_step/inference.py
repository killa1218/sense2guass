# coding=utf8

import sys
import time
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

# from loss import *
from options import Options as opt


def dpgetAllWindows(stcW):
    stack = []
    tmpStack = []

    windowSize = 2 * opt.windowSize + 1
    windows = []
    firstWindowSize = 0

    for i in range(stcW[0].senseNum):
        stack.append([stcW[0].senseStart + i])

    for i in range(1, len(stcW) + opt.windowSize):
        s = set()

        for k in stack:
            if i < len(stcW):
                senseStart = stcW[i].senseStart

                for j in range(stcW[i].senseNum):
                    tmp = k + [senseStart + j]

                    if len(tmp) == windowSize:
                        windows.append(tmp)

                        if tuple(tmp[1:]) not in s:
                            tmpStack.append(tmp[1:])
                            s.add(tuple(tmp[1:]))


                        if i == 2 * opt.windowSize:
                            firstWindowSize += 1
                    else:
                        tmpStack.append(tmp)
            else:
                tmp = k + [k[opt.windowSize]] * (windowSize - len(k))
                windows.append(tmp)

                if tuple(k[1:]) not in s:
                    tmpStack.append(k[1:])
                    s.add(tuple(k[1:]))

        stack = tmpStack
        tmpStack = []

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
            # print(assignList[j])
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
                    assignList.append(tmp[:opt.windowSize - i - 1 + len(stcW)] + [tmp[opt.windowSize]] * (newWordIdx - len(stcW) + 1))
                    # assignList.append(tmp + [tmp[opt.windowSize]])
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

    start = time.time()
    # for i, j in pool.map(dpgetAllWindows, batchStcW): # 0.2+s
    for stcW in batchStcW: # 0.2s
        i, j = dpgetAllWindows(stcW)
        starts.append(len(assignList))
        ends.append(j + starts[-1])
        assignList.extend(i)
    # print("Build assignList time:", time.time() - start)

    start = time.time()
    loss = sess.run(windowLossGraph, feed_dict = {window: assignList})
    # print("Calculate time:", time.time() - start)

    start = time.time()
    for i in range(len(loss)):
        lossTable[tuple(assignList[i])] = loss[i]
    # print("Build lossTable time:", time.time() - start)

    start = time.time()
    for i in range(len(batchStcW)): # 0.2+s
        assign.append(inferenceOneStc(batchStcW[i], lossTable, assignList[starts[i]:ends[i]]))
    # for i in pool.map(inferenceHelper, [(batchStcW[j], lossTable, assignList[starts[j]:ends[j]]) for j in range(len(batchStcW))]): # 2+s
    #     assign.append(i)
    # print("Real inference time:", time.time() - start)

    return assign
