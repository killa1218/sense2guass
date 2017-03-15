# coding=utf8

import time
from options import Options as opt
from libcpp.vector cimport vector


cpdef getAllWindows(stcW):
    cdef:
        int i
        int j
        int k
        int stcWLen
        int firstWindowSize
        int start
        int end
        int midAppended
        int windowLast
    cdef vector[vector[int]] windows
    cdef vector[int] tmp

    firstWindowSize = 0
    stcWLen = len(stcW)

    for i in range(stcWLen - opt.windowSize):
        start = i
        end = 2 * opt.windowSize + 1 + i
        midAppended = 0

        for j in range(start, end):
            if j < stcWLen:
                tmp.push_back(stcW[j].senseStart)
            else:
                tmp.push_back(tmp[opt.windowSize])
                midAppended += 1

        windows.push_back(vector[int](tmp))

        while True:
            if tmp.size() == 0:
                break
            else:
                windowLast = start + tmp.size() - 1 if start + tmp.size() - 1 < stcWLen else stcWLen - 1

                if midAppended > 0 or tmp.back() == stcW[windowLast].senseStart + stcW[windowLast].senseNum - 1:
                    tmp.pop_back()

                    if midAppended > 0:
                        midAppended -= 1
                else:
                    windowLast = tmp.back() + 1 # Use windowLast for temp purpose
                    tmp.pop_back()
                    tmp.push_back(windowLast)

                    for k in range(tmp.size(), opt.windowSize * 2 + 1):
                        if start + k < stcWLen:
                            tmp.push_back(stcW[start + k].senseStart)
                        else:
                            tmp.push_back(tmp[opt.windowSize])
                            midAppended += 1

                    windows.push_back(vector[int](tmp))

        if i == 0:
            firstWindowSize = windows.size()

    return windows, firstWindowSize


cpdef vector[int] inferenceOneStc(stcW, lossTable, assignList):
    cdef int i
    cdef int j
    cdef int k
    cdef int newWordIdx
    cdef int minLossIdx
    cdef double minLoss
    cdef double tmpLoss
    # cdef vector[int] assign
    # cdef vector[double] lossList
    # cdef vector[int] prevAssignList

    assign = []
    lossList = [0] * len(assignList)
    prevAssignList = [None] * len(assignList)

    # lossList = vector[double](len(assignList))
    # prevAssignList = vector[int](len(assignList))
    map = {}

    for i in range(opt.windowSize, len(stcW)):
        minLoss = float('inf')
        minLossIdx = 0
        map = {}

        for j in range(len(assignList)):
            lossList[j] = lossTable[tuple(assignList[j])] + lossList[j]
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
            # assign.push_back(prevAssignList[minLossIdx]) # Record the inferenced sense

        newWordIdx = i + opt.windowSize + 1
        assignList = []
        lossList = []
        prevAssignList = []
        # lossList.clear()
        # prevAssignList.clear()
        for key in map:
            if len(assign) == 0 or map[key][2] == assign[-1]:
            # if assign.size() == 0 or map[key][2] == assign.back():
                tmp = list(key)

                if newWordIdx < len(stcW):
                    for k in range(stcW[newWordIdx].senseNum):
                        assignList.append(tmp + [stcW[newWordIdx].senseStart + k])
                        lossList.append(map[key][1])
                        prevAssignList.append(map[key][0])
                        # lossList.push_back(map[key][1])
                        # prevAssignList.push_back(map[key][0])
                else:
                    assignList.append(tmp[:opt.windowSize - i - 1 + len(stcW)] + [tmp[opt.windowSize]] * (newWordIdx - len(stcW) + 1))
                    lossList.append(map[key][1])
                    prevAssignList.append(map[key][0])
                    # lossList.push_back(map[key][1])
                    # prevAssignList.push_back(map[key][0])

    minLoss = float('inf')
    tmpMinLossIdx = 0
    for key in map:
        if map[key][1] < minLoss:
            minLoss = map[key][1]
            tmpMinLossIdx = key

    assign.append(map[tmpMinLossIdx][0])
    assign.extend(list(tmpMinLossIdx)[:opt.windowSize])

    # assign.push_back(map[tmpMinLossIdx][0])
    # for i in list(tmpMinLossIdx)[:opt.windowSize]:
    #     assign.push_back(i)


cpdef vector[vector[int]] batchDPInference(batchStcW, sess, windowLossGraph, window):
    cdef:
        vector[vector[int]] assignList
        vector[vector[int]] assign
        vector[vector[int]] a
        vector[int] starts
        vector[int] ends
        int b

    # assignList = []
    # assign = []
    # starts = []
    # ends = []
    lossTable = {}

    start = time.time()

    for i in range(len(batchStcW)):
        a, b = getAllWindows(batchStcW[i])
        starts.push_back(assignList.size())
        ends.push_back(b + starts.back())
        assignList.insert(assignList.end(), a.begin(), a.end())
        # starts.append(len(assignList))
        # ends.append(b + starts[-1])
        # assignList.extend(a)

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

    for i in range(len(batchStcW)):
        assign.push_back(inferenceOneStc(batchStcW[i], lossTable, assignList[starts[i]:ends[i]]))
        # assign.append(inferenceOneStc(batchStcW[i], lossTable, assignList[starts[i]:ends[i]]))

    end = time.time()
    print("Real inference time:", end - start)

    return assign
