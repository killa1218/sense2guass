# coding=utf8

import time
from options import Options as opt
from multiprocessing.dummy import Pool

cdef dpgetAllWindows(stcW):
    cdef:
        int windowSize
        int firstWindowSize
        int i
        int j

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


# cdef inferenceOneStc(stcW, lossTable, assignList):
#     cdef int i
#     cdef int j
#     cdef int k
#     cdef int newWordIdx
#     cdef int minLossIdx
#     cdef double minLoss
#     cdef double tmpLoss
#
#     assign = []
#     lossList = [0] * len(assignList)
#     prevAssignList = [None] * len(assignList)
#
#     map = {}
#
#     mapTime = 0
#     parseMapTime = 0
#
#     for i in range(opt.windowSize, len(stcW)): # Each iteration check a window
#         minLoss = float('inf')
#         minLossIdx = 0
#         map = {}
#
#         start = time.time()
#
#         for j in range(len(assignList)): #
#             lossList[j] = lossTable[tuple(assignList[j])] + lossList[j]
#             tmpAssign = assignList[j]
#             tmpLoss = lossList[j]
#
#             if tmpLoss < minLoss:
#                 minLoss = tmpLoss
#                 minLossIdx = j
#
#             t = tuple(tmpAssign[1:])
#             if t not in map.keys():
#                 map[t] = [tmpAssign[0], tmpLoss, prevAssignList[j]]
#             else:
#                 if map[t][1] > tmpLoss:
#                     map[t][0] = tmpAssign[0]
#                     map[t][1] = tmpLoss
#                     map[t][2] = prevAssignList[j]
#
#         end = time.time()
#         mapTime += (end - start)
#         start = time.time()
#
#         if i > opt.windowSize:
#             w = stcW[len(assign)]
#             w.senseCount[prevAssignList[minLossIdx] - w.senseStart] += 1
#             assign.append(prevAssignList[minLossIdx]) # Record the inferenced sense
#
#         newWordIdx = i + opt.windowSize + 1
#         assignList = []
#         lossList = []
#         prevAssignList = []
#
#         for key in map:
#             if len(assign) == 0 or map[key][2] == assign[-1]:
#                 tmp = list(key)
#
#                 if newWordIdx < len(stcW):
#                     for k in range(stcW[newWordIdx].senseNum):
#                         assignList.append(tmp + [stcW[newWordIdx].senseStart + k])
#                         lossList.append(map[key][1])
#                         prevAssignList.append(map[key][0])
#                 else:
#                     assignList.append(tmp[:opt.windowSize - i - 1 + len(stcW)] + [tmp[opt.windowSize]] * (newWordIdx - len(stcW) + 1))
#                     lossList.append(map[key][1])
#                     prevAssignList.append(map[key][0])
#
#         end = time.time()
#         parseMapTime += (end - start)
#
#     minLoss = float('inf')
#     tmpMinLossIdx = 0
#     for key in map:
#         if map[key][1] < minLoss:
#             minLoss = map[key][1]
#             tmpMinLossIdx = key
#
#     assign.append(map[tmpMinLossIdx][0])
#     assign.extend(list(tmpMinLossIdx)[:opt.windowSize])
#
#     return assign


cdef inferenceOneStc(stcW, lossTable, assignList):
    cdef int i
    cdef int j
    cdef int k
    cdef int newWordIdx
    cdef int minLossIdx
    cdef double minLoss
    cdef double tmpLoss

    assign = []
    lossList = [0] * len(assignList)
    prevAssignList = [None] * len(assignList)

    map = {}

    mapTime = 0
    parseMapTime = 0

    for i in range(opt.windowSize, len(stcW)): # Each iteration check a window
        minLoss = float('inf')
        minLossIdx = 0
        map = {}

        start = time.time()

        for j in range(len(assignList)): #
            lossList[j] = lossTable[tuple(assignList[j])] + lossList[j]
            tmpAssign = assignList[j]
            tmpLoss = lossList[j]

            if tmpLoss < minLoss:
                minLoss = tmpLoss
                minLossIdx = j

            t = tuple(tmpAssign[1:])
            if t not in map.keys():
                map[t] = [tmpAssign[0], tmpLoss, prevAssignList[j]]
            else:
                if map[t][1] > tmpLoss:
                    map[t][0] = tmpAssign[0]
                    map[t][1] = tmpLoss
                    map[t][2] = prevAssignList[j]

        end = time.time()
        mapTime += (end - start)
        start = time.time()

        if i > opt.windowSize:
            w = stcW[len(assign)]
            w.senseCount[prevAssignList[minLossIdx] - w.senseStart] += 1
            assign.append(prevAssignList[minLossIdx]) # Record the inferenced sense

        newWordIdx = i + opt.windowSize + 1
        assignList = []
        lossList = []
        prevAssignList = []

        for key in map:
            if len(assign) == 0 or map[key][2] == assign[-1]:
                tmp = list(key)

                if newWordIdx < len(stcW):
                    for k in range(stcW[newWordIdx].senseNum):
                        assignList.append(tmp + [stcW[newWordIdx].senseStart + k])
                        lossList.append(map[key][1])
                        prevAssignList.append(map[key][0])
                else:
                    assignList.append(tmp[:opt.windowSize - i - 1 + len(stcW)] + [tmp[opt.windowSize]] * (newWordIdx - len(stcW) + 1))
                    lossList.append(map[key][1])
                    prevAssignList.append(map[key][0])

        end = time.time()
        parseMapTime += (end - start)

    minLoss = float('inf')
    tmpMinLossIdx = 0
    for key in map:
        if map[key][1] < minLoss:
            minLoss = map[key][1]
            tmpMinLossIdx = key

    assign.append(map[tmpMinLossIdx][0])
    assign.extend(list(tmpMinLossIdx)[:opt.windowSize])

    return assign


def batchDPInference(batchStcW, sess, windowLossGraph, window):
    cdef int i
    assignList = []
    assign = []
    starts = []
    ends = []
    lossTable = {}
    # pool = Pool()

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

    return assign, getWTime, calTime, infTime
