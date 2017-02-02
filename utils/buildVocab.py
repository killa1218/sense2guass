#!/usr/bin/python
# coding=utf8

import sys

sys.path.append('../')

from vocab import Vocab as V

def main():
    corpus = sys.argv[1]
    target = sys.argv[2]

    v = V(corpus)

    v.save(target)




if __name__ == '__main__':
    main()
