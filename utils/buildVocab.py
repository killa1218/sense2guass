#!/usr/bin/python
# coding=utf8

import sys
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

from vocab import Vocab as V
from options import Options as opt
import tensorflow as tf

def main():
    corpus = sys.argv[1]
    target = sys.argv[2]
    min = sys.argv[3] if sys.argv[3] else 1
    v = None

    opt.minCount = int(min)

    try:
        v = V()
        v.parse(corpus)
    except KeyboardInterrupt:
        print('Met key interrupt.')
    except Exception as e:
        print(e)
        print('Some error.')
    finally:
        if v:
            with tf.Session() as sess:
                v.saveVocabWithEmbeddings(target, sess)
            print('Vocab saved.')

if __name__ == '__main__':
    main()
