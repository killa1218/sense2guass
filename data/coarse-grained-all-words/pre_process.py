#!/usr/bin/python
# coding:utf-8

from __future__ import print_function
import pickle as pk

with open('senseNumberDict.pk', 'wb') as save, open('sense_clusters-21.senses', 'r') as f:
    dic = {}

    for line in f.readlines():
        word = line.split('%')[0]

        if word in dic.keys():
            dic[word] += 1
        else:
            dic[word] = 1

    print(dic)

    pk.dump(dic, save)
