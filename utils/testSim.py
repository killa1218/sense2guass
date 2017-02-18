# coding=utf8

import sys
import pickle as pk
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from graph import batchSentenceLossGraph


data = None
vocab = None

with open('data/SWCS/testData.pk3', 'rb') as f, open('data/'):
    data = pk.load(f)

