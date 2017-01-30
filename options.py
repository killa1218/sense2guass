# coding:utf-8

from __future__ import print_function


class Options(object):
    """Golobal options of training"""
    # def __init__(self):
    embSize = 50                        # Size of embeddings
    initWidth = 2                       # Range of initialization for embeddings
    covarShape = 'diagnal'              # Shape of covariance matrix, possble values: 'diagnal', 'spherical', 'normal'
    windowSize = 5                      # Window size of the energy function
    fixWindowSize = True                # Whether fix the window size or choose a size randomly
    margin = 1                          # The margin between positive and negative pair energy


# import tensorflow as tf
#
# v1 = tf.Variable(tf.random_uniform([1, 50], -1, 1), dtype=tf.float32, name='v1')
# v2 = tf.Variable(tf.random_uniform([1, 50], -1, 1), dtype=tf.float32, name='v2')
# opt = tf.train.GradientDescentOptimizer(0.5)
# sess = tf.Session()
# loss = tf.reduce_sum(v1 * v2)
# sess.run(tf.global_variables_initializer())
#
# print sess.run(v1)
#
# for i in range(100):
#     sess.run(opt.minimize(loss))
#     print sess.run(loss)
#
# print sess.run(v1)


def dfs(arr):
    stack = [0] * len(arr)
    yield stack

    while True:
        if(len(stack) == 0):
            break
        else:
            if stack[-1] == arr[len(stack) - 1] - 1:
                stack.pop()
            else:
                stack[-1] += 1
                stack += [0] * (len(arr) - len(stack))
                yield stack


def main():
    arr = [1,2,3,4]

    # dfs(arr)

    for i in dfs(arr):
        print(i)

main()
