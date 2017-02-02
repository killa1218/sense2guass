#!/usr/bin/python
# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from vocab import Vocab as V

def main():
    pass
    corpus = sys.argv[1]
    target = sys.argv[2]

    v = V(corpus)

    v.save(target)




if __name__ == '__main__':
    main()
