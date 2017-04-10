# coding:utf-8

from __future__ import print_function

import re
import tensorflow as tf

class Options(object):
    """ Global options of training. """
    embSize = 50                        # Size of embeddings
    initWidth = 1                       # Range of initialization for embeddings
    covarShape = 'diagnal'              # Shape of covariance matrix, possible values: 'diagnal', 'spherical', 'normal'
    windowSize = 3                      # Window size of the energy function
    fixWindowSize = True                # Whether fix the window size or choose a size randomly
    margin = 100                        # The margin between positive and negative pair energy
    sensePrior = True                   # Whether use sense statistical information as prior probability when doing reference
    sentenceLength = 20                 # Length of a sentence
    maxSensePerWord = 5                 # Maximum number of senses of one word
    batchSize = 50                      # How many sentences trained in a batch
    threads = 3                         # How many threads are used to train
    wordSeparator = re.compile('\s*,\s*|\s*\.\s*|\s+')  # Separator pattern of words in corpus
    minCount = 10                       # Words who exist under minCount times will be omitted
    negative = 1                        # Negative samples for each sense
    EL = False                          # Whether use EL or KL
    dType = tf.float64                  # Data type
