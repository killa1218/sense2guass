# coding:utf-8

from __future__ import print_function

import re

class Options(object):
    """ Global options of training. """
    embSize = 50                        # Size of embeddings
    initWidth = 2                       # Range of initialization for embeddings
    covarShape = 'diagnal'              # Shape of covariance matrix, possible values: 'diagnal', 'spherical', 'normal'
    windowSize = 5                      # Window size of the energy function
    fixWindowSize = True                # Whether fix the window size or choose a size randomly
    margin = 1                          # The margin between positive and negative pair energy
    sensePrior = True                   # Whether use sense statistical information as prior probability when doing reference
    maxSentenceLength = 1000            # Maximum length of a sentence
    batchSize = 1                       # How many sentences trained in a batch
    threads = 12                        # How many threads are used to train
    wordSeparator = re.compile('\s*,\s*|\s*\.\s*|\s+')  # Separator pattern of words in corpus
    minCount = 5                        # Words who exist under minCount times will be omitted

