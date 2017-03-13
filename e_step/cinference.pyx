# coding=utf8

import sys
import time
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from loss import *
from options import Options as opt


def getAllWindows(stcW):
    windows = []
    firstWindowSize = 0
    stcWLen = len(stcW)

    for i in range(stcWLen - opt.windowSize):
        start = i
        mid = i + opt.windowSize
        end = mid + opt.windowSize + 1
        tmp = []
        midAppended = 0

        for j in range(start, end):
            if j < stcWLen:
                tmp.append(stcW[j].senseStart)
            else:
                tmp.append(tmp[opt.windowSize])
                midAppended += 1

        windows.append(tmp[:])

        while True:
            if len(tmp) == 0:
                break
            else:
                windowLast = start + len(tmp) - 1 if start + len(tmp) - 1 < stcWLen else stcWLen - 1

                if midAppended > 0 or tmp[-1] == stcW[windowLast].senseStart + stcW[windowLast].senseNum - 1:
                    tmp.pop()

                    if midAppended > 0:
                        midAppended -= 1
                else:
                    tmp[-1] += 1

                    for k in range(len(tmp), opt.windowSize * 2 + 1):
                        if start + k < stcWLen:
                            tmp.append(stcW[start + k].senseStart)
                        else:
                            tmp.append(tmp[opt.windowSize])
                            midAppended += 1

                    windows.append(tmp[:])

        if i == 0:
            firstWindowSize = len(windows)

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

    for i, j in pool.map(getAllWindows, batchStcW):
        starts.append(len(assignList))
        ends.append(j + starts[-1])
        assignList.extend(i)
        # print("Length of current assignList:", len(i))
        # print("Current assignList:", i)
        # print(starts[-1], ends[-1])

    # print("Length of total assignList:", len(assignList))
    # print("Total assignList:", assignList)

    end = time.time()
    print("Build assignList time:", end - start)
    start = time.time()

    loss = sess.run(windowLossGraph, feed_dict = {window: assignList})

    end = time.time()
    print("Calculate time:", end - start)
    start = time.time()

    for i in range(len(loss)):
        lossTable[tuple(assignList[i])] = loss[i]

    end = time.time()
    print("Build lossTable time:", end - start)
    start = time.time()

    for i in pool.imap_unordered(inferenceHelper, [(batchStcW[j], lossTable, assignList[starts[j]:ends[j]]) for j in range(len(batchStcW))]):
        assign.append(i)

    end = time.time()
    print("Real inference time:", end - start)

    return assign
