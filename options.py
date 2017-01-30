# coding:utf-8

from __future__ import print_function


class Options(object):
    """ Global options of training. """
    embSize = 50                        # Size of embeddings
    initWidth = 2                       # Range of initialization for embeddings
    covarShape = 'diagnal'              # Shape of covariance matrix, possble values: 'diagnal', 'spherical', 'normal'
    windowSize = 5                      # Window size of the energy function
    fixWindowSize = True                # Whether fix the window size or choose a size randomly
    margin = 1                          # The margin between positive and negative pair energy
    sensePrior = True                   # Whether use sense statistical information as prior probability when doing reference
