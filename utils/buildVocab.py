#!/usr/bin/python
# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from vocab import Vocab as V

def main():
    corpus = sys.argv[1]
    target = sys.argv[2]
    v = None

    print(sys.path)

    try:
        v = V(corpus)
    except KeyboardInterrupt:
        print('Met key interrupt.')
    except Exception as e:
        print(e)
        print('Some error.')
    finally:
        if v:
            v.save(target)
            print('Vocab saved.')

if __name__ == '__main__':
    main()
