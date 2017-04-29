#coding=utf8

import time
import itertools
cimport numpy as np
from options import Options as opt
from cython.parallel import parallel, prange
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from multiprocessing.dummy import Pool

cdef dpgetAllWindows(int[:] senseStarts, int[:] senseNums, int windowSize) nogil:
    cdef:
        int firstWindowSize
        int i
        int j

    stack = []
    tmpStack = []
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

    # print(windows)

    return windows, firstWindowSize


cdef inferenceOneStc(stcW, lossTable, assignList):
    cdef:
        int i
        int j
        int k
        int newWordIdx
        int minLossIdx
        int newSense
        double minLoss

    assignRec = [{}] * (len(stcW) - 2 * opt.windowSize - 1) # 用于记录每个window后几位为key最大的值对应的第一位是啥
    map = None

    for i in range(opt.windowSize, len(stcW) - opt.windowSize - 1): # Each iteration check a window
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

                for key, val in map.iteritems():
                    tmp = list(key) + [newSense] # 将当前新进入单词的所有sense与map中所有key拼接成window
                    curWloss = lossTable[tuple(tmp)] + val
                    t = tuple(tmp[1:]) # 截取窗口除第一个之外的所有sense作为key

                    if t not in tmpMap.keys() or tmpMap[t] > curWloss:
                        tmpMap[t] = curWloss
                        assignRec[i - opt.windowSize][t] = tmp[0]

        if i != len(stcW) - opt.windowSize - 2:
            map = tmpMap

    minLoss = float('inf')
    tmpMinLossIdx = None
    for key, val in map.iteritems():
        if val < minLoss:
            minLoss = val
            tmpMinLossIdx = key

    assign = list(tmpMinLossIdx)
    for i in range(-len(assignRec), -1, -1):
        se = assignRec[i][tuple(assign[:])]
        assign.insert(0, se)

    return assign


cpdef batchDPInference(int[:, :] batchSStarts, int[:, :] batchSNums, sess, windowLossGraph, window):
    cdef int i
    cdef int length
    assignList = []
    assign = []
    starts = []
    ends = []
    lossTable = {}
    # pool = Pool()

    # cdef int** arr = new int[100][100]

    # for i in prange(length, nogil=True):
    #     print(i)

    start = time.time()
    # for a, b in pool.map(dpgetAllWindows, batchStcW):
    for stcW in batchStcW:
        a, b = dpgetAllWindows(stcW)
        starts.append(len(assignList))
        ends.append(b + starts[-1])
        assignList.extend(a)
    # print("Get Windows Time:", time.time() - start)
    getWTime = time.time() - start

    loss = []
    step = 100000
    start = time.time()
    for i in range(0, len(assignList), step):
        subAssignList = assignList[i:i + step]
        loss.extend(list(sess.run(windowLossGraph, feed_dict = {window: subAssignList})))
    # print("Calculate Time:", time.time() - start)
    calTime = time.time() - start

    for i in range(len(loss)):
        lossTable[tuple(assignList[i])] = loss[i]

    start = time.time()
    for i in range(len(batchStcW)):
        assign.append(inferenceOneStc(batchStcW[i], lossTable, assignList[starts[i]:ends[i]]))
    # print("Inference Time:", time.time() - start)
    infTime = time.time() - start

    # delete[] arr

    return assign, getWTime, calTime, infTime
