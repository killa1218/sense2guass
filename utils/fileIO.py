# coding=utf8

import time
import sys

from options import Options as opt
from os import path

sys.path.append(path.abspath(path.join(path.dirname(path.realpath(__file__)), path.pardir)))

def fetchSentences(f, buffer=20000, sentenceLength=1000):
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

        end = time.time()
        readtime += end - start
        sys.stdout.write('\rSpeed: %.2fKB/s.' % (float(f.tell()/1000)/readtime))
        sys.stdout.flush()
