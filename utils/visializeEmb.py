# coding=utf8

import sys
import os.path as path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

import random
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from vocab import Vocab
from mpl_toolkits.mplot3d import Axes3D
from utils.tsne import tsne
import tensorflow as tf


def main():
    vocabPath = sys.argv[1]
    vocab = Vocab('../data/text8')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        means = sess.run(vocab.means)

    # vocab.load(vocabPath)
    colorMap = ['c','g','r','m','k']

    with open(vocabPath, 'rb') as f:
        # means = pk.load(f)['means']
        # sigmas = np.array(vocab.sigmas)

        data = []
        markers = []
        colors = []
        for i in random.sample(range(vocab.size), 10000):
            word = vocab.getWord(i)

            if word.count > 1000 and len(word.token) > 3:
                senseStart = word.senseStart
                senseNum = word.senseNum
                for j in range(senseNum):
                    markers.append(r'${}$'.format(word.token))
                    data.append(means[senseStart + j])
                    colors.append(colorMap[j])

    # ax = plt.subplot(111, projection='3d')
    # P = tsne(np.array(data), 3, len(data[0]))

    ax = plt.subplot(111)
    P = tsne(np.array(data), 2, len(data[0]))

    for i in range(len(P)):
        # ax.scatter(P[i,0], P[i,1], P[i,2], c = colors[i], marker = markers[i], edgecolors = 'none', s = len(markers[i]) * 100)
        ax.scatter(P[i,0], P[i,1], c = colors[i], marker = markers[i], edgecolors = 'none', s = (len(markers[i]) * 100))

    plt.show()

if __name__ == '__main__':
    main()
