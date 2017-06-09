from vocab import Vocab
from options import Options as opt
import tensorflow as tf
from struct import pack
from tqdm import tqdm

# for i in [50,80,100,120,130,140,150,200,300]:
#     file = "gauss.EL.0520_w3_b50_m" + str(i) + ".adam.pkl3"
file = "IP.06072121w3b20lr0.02m1.0n6adam.pkl"

a = []
v = Vocab()

v.load("../data/" + file)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    M = v.means.eval()

    with open("../data/" + file + ".bin", "wb") as f:
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
