from vocab import Vocab
from options import Options as opt
import tensorflow as tf
from struct import pack
from tqdm import tqdm
# from multiprocessing import Pool

file = "gauss.IP.0425_w3_b50_m1.word2vec_text8.pkl3"

a = []
v = Vocab()

v.load("../data/" + file)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    M = v.means.eval()

    with open("../data/vecPy.bin", "wb") as f:
        s = "%d %d\n" % (len(v._idx2word), opt.embSize)
        f.write(bytes(s.encode('ascii')))

        for i in tqdm(v._idx2word):
            word = i.token
            mean = M[i.senseStart]
            f.write(bytes(("%s " % word).encode('ascii')))

            for x in mean:
                f.write(pack('f', x))

            f.write(bytes('\n'.encode('ascii')))

            # s = "%s %s\n" % (word, ' '.join([pack('f', x) for x in mean]))
            # a.append(s)

        # data = ''.join(a)

        # f.write(bytes(data.encode('ascii')))
