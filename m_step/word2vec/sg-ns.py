#!/usr/bin/python
# coding:utf8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

# from six.moves import xrange    # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("output", None, "Directory to write the model and training summaries.")
flags.DEFINE_string("train", None, "Training text file. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_integer("size", 100, "The embedding dimension size. Default is 100.")
flags.DEFINE_integer("window", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_float("sample", 1e-3, "Subsample threshold for word occurrence. Words that appear with higher frequency will be randomly down-sampled. Set to 0 to disable. Default is 1e-3")
flags.DEFINE_integer("negative", 100, "Negative samples per training example. Default is 100.")
flags.DEFINE_integer("threads", 12, "How many threads are used to train. Default 12.")
flags.DEFINE_integer("iter", 15, "Number of iterations to train. Each iteration processes the training data once completely. Default is 15.")
flags.DEFINE_integer("min_count", 5, "The minimum number of word occurrences for it to be included in the vocabulary. Default is 5.")
flags.DEFINE_float("alpha", 0.2, "Initial learning rate. Default is 0.2.")
flags.DEFINE_boolean("gpu", False, "If true, use GPU instead of CPU.")

flags.DEFINE_integer("batch_size", 128, "Number of training examples processed per step (size of a minibatch).")
flags.DEFINE_boolean("interactive", False, "If true, enters an IPython interactive session to play with the trained model. E.g., try model.analogy(b'france', b'paris', b'russia') and model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5, "Save training summary to file every n seconds (rounded up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600, "Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS


class Options(object):
    """Options used by our word2vec model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.size = FLAGS.size
        # Training options. The training text file.
        self.train = FLAGS.train
        # Number of negative samples per example.
        self.negative = FLAGS.negative
        # The initial learning rate.
        self.alpha = FLAGS.alpha
        # Number of epochs to train. After these many epochs, the learning rate decays linearly to zero and the training stops.
        self.iter = FLAGS.iter
        # Concurrent training steps.
        self.threads = FLAGS.threads
        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size
        # The number of words to predict to the left and right of the target word.
        self.window = FLAGS.window
        # The minimum number of word occurrences for it to be included in the vocabulary.
        self.min_count = FLAGS.min_count
        # Subsampling threshold for word occurrence.
        self.sample = FLAGS.sample
        # How often to print statistics.
        self.statistics_interval = FLAGS.statistics_interval
        # How often to write to the summary file (rounds up to the nearest statistics_interval).
        self.summary_interval = FLAGS.summary_interval
        # How often to write checkpoints (rounds up to the nearest statistics interval).
        self.checkpoint_interval = FLAGS.checkpoint_interval
        # Use GPU or CPU. True for GPU, otherwise CPU
        self.gpu = FLAGS.gpu
        # Where to write out summaries.
        self.save_path = FLAGS.output
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)



class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}                          # The hash map from word to index
        self._id2word = []                          # The vocab array
        self.build_graph()
        # self.build_eval_graph()
        self.save_vocab()


    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.size
        emb = tf.Variable(
                tf.random_uniform(
                        [opts.vocab_size, opts.size], -init_width, init_width),
                name="emb")
        self._emb = emb

        # Softmax weight: [vocab_size, emb_dim]. Transposed.
        sm_w_t = tf.Variable(
                tf.zeros([opts.vocab_size, opts.size]),
                name="sm_w_t")

        # Softmax bias: [emb_dim].
        sm_b = tf.Variable(tf.zeros([opts.vocab_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
                tf.cast(labels,
                                dtype=tf.int64),
                [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=opts.negative,
                unique=True,
                range_max=opts.vocab_size,
                distortion=0.75,
                unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.mul(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [opts.negative])
        sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits


    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                true_logits, tf.ones_like(true_logits))
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                sampled_logits, tf.zeros_like(sampled_logits))

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                                             tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor


    def optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.iter)
        lr = opts.alpha * tf.maximum(
                0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss,
             global_step=self.global_step,
             gate_gradients=optimizer.GATE_NONE)
        self._train = train


    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples, labels) = word2vec.skipgram(
            filename=opts.train,
            batch_size=opts.batch_size,
            window_size=opts.window,
            min_count=opts.min_count,
            subsample=opts.sample
        )                                           # Build the graph that processes corpus file

        (
            opts.vocab_words,                       # List of all the words in the corpus
            opts.vocab_counts,                      # List of the count of words ordered by desc
            opts.words_per_epoch                    # Number of tokens in the corpus file
        ) = self._session.run([words, counts, words_per_epoch])    # Process corpus file

        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.scalar_summary("NCE loss", loss)         # Used for visuallization

        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        tf.initialize_all_variables().run()         # r0.10
        # tf.global_variables_initializer().run()   # r0.12

        self.saver = tf.train.Saver()


    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (vocab_word, opts.vocab_counts[i]))


    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break


    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(opts.save_path, self._session.graph)
        workers = []
        for _ in xrange(opts.threads):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_checkpoint_time = 0
        while True:
            time.sleep(opts.statistics_interval)    # Reports our progress once a while.
            (epoch, step, loss, words, lr) = self._session.run(
                    [self._epoch, self.global_step, self._loss, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (
                    now - last_time)
            print("Iter %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
                        (epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session,
                                                os.path.join(opts.save_path, "model.ckpt"),
                                                global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()

        return epoch


    def save_emb(self, file):
        with open(file, 'wb') as f:
            pass




def main(_):
    """Train a word2vec model."""
    if not FLAGS.train or not FLAGS.output:         # Whether the train corpus and output path is set
        print("--train and --output must be specified.")
        sys.exit(1)
    opts = Options()                                # Instantiate option object
    device = "/cpu:0"
    with tf.Graph().as_default(), tf.Session() as session:
        # if opts.gpu:                                # Judge whether use CPU(default) or GPU
        #     device = "/gpu:0"
        #
        # with tf.device(device):
        model = Word2Vec(opts, session)         # Instantiate model
            # model.read_analogies() # Read analogy questions
        for _ in xrange(opts.iter):      # According to the parameter train the embeddings for iter iterations
            model.train()    # Process one epoch
            # model.eval()    # Eval analogies.
        # Perform a final save.
        model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"), global_step=model.global_step)    # Save model
        # if FLAGS.interactive:
        #     # E.g.,
        #     # [0]: model.analogy(b'france', b'paris', b'russia')
        #     # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
        #     _start_shell(locals())


if __name__ == "__main__":
    tf.app.run()
