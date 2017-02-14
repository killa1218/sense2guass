# coding=utf8

import time
import sys
import os

from options import Options as opt
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

def fetchSentences(f, buffer=20000, sentenceLength=1000, verbose=True):
    tmp = []
    # tank = StringIO()
    EOF = False
    readtime = 0

    while not EOF:
        start = time.time()
        chunk = f.read(buffer)
        # tank.write(chunk)
        if not chunk:
            EOF = True

        c = f.read(1)
        while c and opt.wordSeparator.match(c) is None:
            chunk += c
            # tank.write(c)
            c = f.read(1)

        if not EOF:
            # tmp.extend(opt.wordSeparator.split(tank.getvalue()))
            # del(tank)
            # tank = StringIO()

            tmp.extend(opt.wordSeparator.split(chunk))

            while len(tmp) >= sentenceLength:
                yield tmp[:sentenceLength]

                tmp = tmp[sentenceLength:]
        else:
            yield tmp

        if verbose:
            end = time.time()
            readtime += end - start
            sys.stdout.write('\rSpeed: %.2fKB/s. %.2f%% Loaded' % (float(f.tell()/1000)/readtime, float(f.tell())*100/os.path.getsize(f.name)))
            sys.stdout.flush()


def fetchSentencesAsWords(f, vocab, buffer=20000, sentenceLength=1000, verbose=True):
    tmp = []
    # tank = StringIO()
    EOF = False
    readtime = 0

    while not EOF:
        start = time.time()
        chunk = f.read(buffer)
        # tank.write(chunk)
        if not chunk:
            EOF = True

        c = f.read(1)
        while c and opt.wordSeparator.match(c) is None:
            chunk += c
            # tank.write(c)
            c = f.read(1)

        if not EOF:
            # tmp.extend(opt.wordSeparator.split(tank.getvalue()))
            # del(tank)
            # tank = StringIO()

            for i in opt.wordSeparator.split(chunk):
                w = vocab.getWord(i)

                if w:
                    tmp.append(w)

            while len(tmp) >= sentenceLength:
                yield tmp[:sentenceLength]

                tmp = tmp[sentenceLength:]
        else:
            yield tmp

        if verbose:
            end = time.time()
            readtime += end - start
            sys.stdout.write('\rSpeed: %.2fKB/s. %.2f%% Loaded' % (float(f.tell()/1000)/readtime, float(f.tell())*100/os.path.getsize(f.name)))
            sys.stdout.flush()
