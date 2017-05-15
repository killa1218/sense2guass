# coding=utf8

import sys
import time
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

# from loss import *
from options import Options as opt
from multiprocessing import Pool, Manager

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
    assignRec = [{}] * (len(stcW) - 2 * opt.windowSize) # 用于记录每个window后几位为key最大的值对应的第一位是啥
    map = None

    for i in range(opt.windowSize, len(stcW) - opt.windowSize): # Each iteration check a window
        tmpMap = {}

        if i == opt.windowSize: # 记录并筛选第一个window
            for j in range(len(assignList)):
                curWloss = lossTable[tuple(assignList[j])]
                tmpAssign = assignList[j]

                t = tuple(tmpAssign[1:])
                if t not in tmpMap.keys() or tmpMap[t] > curWloss:
                    tmpMap[t] = curWloss
                    assignRec[i - opt.windowSize][t] = tmpAssign[0]
        else: # 分析除第一个window之外的window
            newWordIdx = i + opt.windowSize
            newWord = stcW[newWordIdx]

            for k in range(newWord.senseNum):
                newSense = newWord.senseStart + k

                for key, val in map.items():
                    tmp = list(key) + [newSense] # 将当前新进入单词的所有sense与map中所有key拼接成window
                    curWloss = lossTable[tuple(tmp)] + val
                    t = tuple(tmp[1:]) # 截取窗口除第一个之外的所有sense作为key

                    if t not in tmpMap.keys() or tmpMap[t] > curWloss:
                        tmpMap[t] = curWloss
                        assignRec[i - opt.windowSize][t] = tmp[0]

        map = tmpMap

    minLoss = float('inf')
    tmpMinLossIdx = None
    for key, val in map.items():
        if val < minLoss:
            minLoss = val
            tmpMinLossIdx = key

    assign = list(tmpMinLossIdx)
    for i in range(len(assignRec) - 1, -1, -1):
        se = assignRec[i][tuple(assign[:2 * opt.windowSize])]
        assign.insert(0, se)

    return assign


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
    getWTime = time.time() - start

    start = time.time()
    loss = []
    step = 100000
    for i in range(0, len(assignList), step):
        subAssignList = assignList[i:i + step]
        loss.extend(list(sess.run(windowLossGraph, feed_dict = {window: subAssignList})))
    # print("Calculate time:", time.time() - start)
    calTime = time.time() - start

    # with Manager() as manager:
    #     lossTable = manager.dict()
    for i in range(len(loss)):
        lossTable[tuple(assignList[i])] = loss[i]
    # print("Build lossTable time:", time.time() - start)

    start = time.time()
    for i in range(len(batchStcW)): # 0.2+s
        assign.append(inferenceOneStc(batchStcW[i], lossTable, assignList[starts[i]:ends[i]]))
    # for i in pool.map(inferenceHelper, [(batchStcW[j], lossTable, assignList[starts[j]:ends[j]]) for j in range(len(batchStcW))]): # 2+s
    #     assign.append(i)
    # print("Real inference time:", time.time() - start)
    infTime = time.time() - start

    return assign, getWTime, calTime, infTime
