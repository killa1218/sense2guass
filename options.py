# coding:utf-8

from __future__ import print_function


class Options(object):
    """Golobal options of training"""
    # def __init__(self):
    embSize = 50                        # Size of embeddings
    initWidth = 2                       # Range of initialization for embeddings
    covarShape = 'diagnal'              # Shaoe of covariance matrix, possable values: 'diagnal', 'spherical', 'normal'
    windowSize = 5                      # Window size of the energy function
    fixWindowSize = True                # Whether fix the window size or choose a size randomly
