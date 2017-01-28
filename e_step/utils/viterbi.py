# coding=utf8

import sys

sys.path.append('../..')

from utils.distance import dist as dist
from loss import *
from options import Options as opt
import tensorflow as tf
from tf.nn import sigmoid as act
from tf.nn import tanh as act
from tf.nn import relu as act
import pprint

pp = pprint.PrettyPrinter(indent=4)



def dfs(stc, n, sLabel):
    global minLoss
    global assign

    if n == len(stc):
        return

    with tf.Session() as sess:
        for i in range(stc[n].senseNum):
            sLabel[n] = i
            cur = sess.run(avgSkipGramLoss(stc, sLabel))
            if cur < minLoss:
                minLoss = cur
                assign = sLabel
            dfs(stc, n + 1, sLabel)



def violentInference(stc):
    senseLabel = [0] * len(stc)
    assign = []

    dfs(stc, 0, senseLabel)

    return assign



def dpInference(stc, vocab):
    V = [{}]                            # Record Intermediate Probability
    path = []                           # Output
    assign = []                         # Result of word senses in a sentence

    # Initialize Probability Table
    for i in range(opt.windowSize - 1):
        for j in range(vocab.getWord(stc[i]).senseCount):
            V[i][j] = {"prob": vocab.getWord(stc[i]).senseP[j], "prev": None}

    # for st in states:
    #     V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    #     pp.pprint(V)
    # Run Viterbi when t > 0
    for t in range(1, len(stc)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    pp.pprint(V)
                    break

    return V

    # for line in dptable(V):
    #     print line
    # opt = []
    # # The highest probability
    # max_prob = max(value["prob"] for value in V[-1].values())
    # previous = None
    # # Get most probable state and its backtrack
    # for st, data in V[-1].items():
    #     if data["prob"] == max_prob:
    #         opt.append(st)
    #         previous = st
    #         break
    # # Follow the backtrack till the first observation
    # for t in range(len(V) - 2, -1, -1):
    #     opt.insert(0, V[t + 1][previous]["prev"])
    #     previous = V[t + 1][previous]["prev"]
    # print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        pp.pprint(V)
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    pp.pprint(V)
                    break

    for line in dptable(V):
        print line
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }

viterbi(observations,
    states,
    start_probability,
    transition_probability,
    emission_probability)
