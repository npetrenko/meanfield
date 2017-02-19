import tensorflow as tf
import numpy as np
from math import pi
import gc

sample_size = 1

class Dense():
    def __init__(self, dim, input_layer, act=tf.nn.relu, params=None, samp_size=sample_size, prior=3):
        self.dim = dim
        self.inp_dim = input_layer.dim

        shape = [self.inp_dim, self.dim]
        self.mean = tf.Variable(np.random.normal(size=shape, scale=0.05), dtype=tf.float32)
        self.sigma = tf.Variable(np.random.normal(size=shape, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma = tf.log(tf.exp(self.sigma + 1) + 0.1)

        self.mean_const = tf.Variable(np.random.normal(size=dim, scale=0.05), dtype=tf.float32)
        self.sigma_const = tf.Variable(np.random.normal(size=dim, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma_const = tf.log(tf.exp(self.sigma_const + 0.1) + 1)

        activation_matrix = tf.random_normal(shape=[samp_size] + shape, dtype=tf.float32,
                                             stddev=1) * self.sigma + self.mean
        bias = tf.random_normal(shape=[samp_size, dim], dtype=tf.float32, stddev=1) * self.sigma_const + self.mean_const

        l = []
        for i in range(samp_size):
            temp = tf.matmul(input_layer.output[i], activation_matrix[i])
            l.append(temp)
        temp = tf.pack(l, axis=0)
        self.output = tf.transpose(act(tf.transpose(temp, [1, 0, 2]) + bias), [1, 0, 2])
        print(self.output)

        l1 = tf.reduce_sum(-tf.log(np.sqrt(2 * pi) * self.sigma)) + tf.reduce_sum(
            -tf.log(np.sqrt(2 * pi) * self.sigma_const))
        l2 = (tf.reduce_sum(-activation_matrix ** 2) + tf.reduce_sum(- bias ** 2)) / (2 * prior)
        self.loss = l1 - l2
        self.loss += input_layer.loss

class Input():
    def __init__(self, dim, sample_size=sample_size):
        self.input = tf.placeholder(shape=tuple([sample_size, None] + [dim]), dtype=tf.float32)
        self.output = self.input
        self.dim = dim
        self.loss = 0

class Model():
    def terminate(self):
        self.sess.close()
    def __init__(self, input, output, optimizer=tf.train.AdamOptimizer(0.001)):
        self.sess = tf.Session()
        self.loss = output.loss
        self.y_ph = tf.placeholder(shape=(sample_size, None, 1), dtype=tf.float32)

        self.loss_final = False

        self.input = input
        self.output = output

        self.optimizer = optimizer

        self.sess.run(tf.initialize_all_variables())

    def fit(self, X, y, nepoch, batchsize, log_freq=100, valid_set = None, shuffle_freq = 1):
        in_tens = np.repeat([X], sample_size, axis=0)#.reshape((sample_size, -1, 1))
        in_tens_y = np.repeat([y], sample_size, axis=0)#.reshape((sample_size, -1, 1))
        nbatch = int(len(X)/batchsize)

        if not self.loss_final:
            loss = tf.reduce_sum(((self.output.output - self.y_ph) ** 2)) / (2 * 5 ** 2) + self.loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss += loss
            self.optimizer = self.optimizer.minimize(self.loss)
            self.sess.run(tf.initialize_all_variables())
            self.loss_final = True

        for epoch in range(nepoch):
            if epoch % log_freq == 0:
                preds = self.predict(X, samplesize=20)
                train_mse = np.sqrt(np.mean((preds-y) ** 2))
                if valid_set is not None:
                    preds = self.predict(valid_set[0], samplesize=20)
                    valid_mse = np.sqrt(np.mean((preds - valid_set[1]) ** 2))
                    print('epoch: {} \n train error: {} \n valid_error: {} \n\n\n'.format(epoch, train_mse, valid_mse))
                else:
                    print('epoch: {} \n train error: {} \n\n\n'.format(epoch, train_mse))
            for i in range(nbatch):
                self.sess.run(self.optimizer,
                              feed_dict={
                                  self.input.input: in_tens[:, batchsize * i:batchsize * (i + 1), :],
                                  self.y_ph: in_tens_y[:, batchsize * i:batchsize * (i + 1), :]
                              })

            if epoch % shuffle_freq == 0:
                shuffle = np.random.permutation(in_tens.shape[1])
                gc.collect()
                in_tens = in_tens[:, shuffle, :]
                in_tens_y = in_tens_y[:, shuffle, :]


    def predict(self, X, samplesize=250, return_distrib=False):
        n = int(samplesize / sample_size) + 1
        X = np.repeat([X], sample_size, axis=0)
        preds = None
        if return_distrib:
            for i in range(n):
                part_pred = self.sess.run(self.output.output,
                                          feed_dict={self.input.input: X})
                if not preds is None:
                    preds = np.append(part_pred, preds, axis=0)
                else:
                    preds = part_pred
        else:
            for i in range(n):
                if not preds is None:
                    preds += self.sess.run(tf.reduce_mean(self.output.output, reduction_indices=0),
                                           feed_dict={self.input.input : X})
                else:
                    preds = self.sess.run(tf.reduce_mean(self.output.output, reduction_indices=0),
                                          feed_dict={self.input.input : X})
            preds = preds/n
        return preds


