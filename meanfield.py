import numpy as np
from math import pi
import gc
import tensorflow as tf
import time


class Network():
    '''
    initialize global NN class
    sample_size: number of samples on monte carlo step
    target_std_deviation: standard deviation of P(y |X, tetha)
    '''
    sample_size = 1
    target_std_deviation = 0.2


class Layer(Network):
    '''
    empty class to keep everything neat
    '''
    pass


class Dense(Layer):
    initial_sigma = -6
    def __init__(self, dim, input_layer, act=tf.nn.relu, prior=3, name=''):
        '''
        :param dim: number of neurons in the layer
        :param input_layer: input layer calss object
        :param act: tensorflow activation function
        :param prior: standard deviation of prior
        '''
        prior = tf.Variable(prior, dtype=tf.float32, trainable=False, name='prior_' + name)
        
        self.weights = input_layer.weights

        sample_size = self.sample_size
        self.dim = dim
        self.inp_dim = input_layer.dim

        shape = [self.inp_dim, self.dim]
        self.mean = tf.Variable(np.random.normal(size=shape, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma = tf.Variable(np.random.normal(size=shape, scale=0.01, loc=0), dtype=tf.float32)
        self.weights += [self.mean, self.sigma]
        
        #self.sigma = tf.log(tf.exp(self.sigma + self.initial_sigma) + 1)
        self.sigma = tf.exp(self.sigma + self.initial_sigma)

        self.mean_const = tf.Variable(np.random.normal(size=dim, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma_const = tf.Variable(np.random.normal(size=dim, scale=0.01, loc=0), dtype=tf.float32)
        self.weights += [self.mean_const, self.sigma_const]
        
        #self.sigma_const = tf.log(tf.exp(self.sigma_const + self.initial_sigma) + 1)
        self.sigma_const = tf.exp(self.sigma_const + self.initial_sigma)

        # sample of activation matrixes and biases
        activation_matrix = tf.random_normal(shape=[sample_size] + shape, dtype=tf.float32,
                                             stddev=1) * self.sigma + self.mean
        bias = tf.random_normal(shape=[sample_size, 1, dim], dtype=tf.float32, stddev=1) * self.sigma_const + self.mean_const

        # calculate matrix multiplication for each sample
        l = []
        for i in range(sample_size):
            temp = tf.matmul(input_layer.output[i], activation_matrix[i])
            l.append(temp)
        temp = tf.stack(l, axis=0)

        self.logits = temp+bias
        self.output = act(self.logits)
        print(self.output)

        # ...
        l1 = tf.reduce_sum(-tf.log(np.sqrt(2 * pi) * self.sigma)) + tf.reduce_sum( -tf.log(np.sqrt(2 * pi) * self.sigma_const) )
        # prior loss
        l2 = (-tf.reduce_sum((activation_matrix ** 2)) - tf.reduce_sum((bias ** 2))) / (2 * prior**2) #probably i need a square here

        # feeding loss to the next layer
        self.loss = l1 - l2
        
        self.loss += input_layer.loss


class Input(Layer):
    def __init__(self, dim):
        '''
        :param dim: number of sensors in the first layer
        '''
        sample_size = self.sample_size
        self.input = tf.placeholder(shape=tuple([sample_size, None] + [dim]), dtype=tf.float32)
        self.output = self.input
        self.dim = dim
        self.loss = 0
        self.weights = []


class Model(Network):

    def terminate(self):
        '''
        close tensorflow session
        '''
        self.sess.close()

    def __init__(self, input, output, optimizer=tf.train.AdamOptimizer(0.001), loss = 'mse'):
        '''
        :param input: input node of the neural network
        :param output: output node of the network
        :param optimizer: tensorflow optimizer
        '''

        self.weights = output.weights
        sample_size = self.sample_size
        self.sess = tf.Session()
        
        # store part of loss dependent on variables:
        self.var_loss = output.loss

        # create placeholder for target values
        self.y_ph = tf.placeholder(shape=(sample_size, None, output.dim), dtype=tf.float32)

        # parameter which helps to build the final loss
        self.loss_final = False

        self.input = input
        self.output = output

        self.optimizer = optimizer



        if loss == 'mse':
            def loss_func(preds, y):
                return np.sqrt(np.mean((preds-y) ** 2))
            self.match_loss = tf.reduce_sum(((self.output.output - self.y_ph) ** 2)) / (2 * self.target_std_deviation ** 2)
        elif loss == 'crossentropy':
            def loss_func(preds, y):
                return np.mean(np.argmax(preds, axis=1) - np.argmax(y, axis=1) != 0)
            self.match_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output.logits, labels=self.y_ph)
        else:
            Exception('No correct loss specified. Use either "mse" of "crossentropy"')

        self.loss_func = loss_func

        self.objective = self.match_loss + self.var_loss

    def fit(self, X, y, nepoch, batchsize, log_freq=100, valid_set = None, shuffle_freq = 1, running_backup_dir=None):
        
        sample_size = self.sample_size
        
        # create input suitable for feeding into the input node
        in_tens = np.repeat([X], sample_size, axis=0)
        in_tens_y = np.repeat([y], sample_size, axis=0)
        
        nbatch = int(len(X)/batchsize)

        if not self.loss_final:

            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.optimizer = self.optimizer.minimize(self.loss)
            self.sess.run(tf.initialize_all_variables())
            self.loss_final = True

            # remember batchsize in case of change
            self.batchsize = batchsize

        # reconfigure loss in case of batch size change
        if self.loss_final and self.batchsize != batchsize:
            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.batchsize = batchsize
            self.optimizer = self.optimizer.minimize(self.loss)

        for epoch in range(nepoch):
            
            # print logs every log_freq epochs:
            if epoch % log_freq == 0:
                preds = self.predict(X, prediction_sample_size=100)
                train_mse = self.loss_func(preds, y)
                obj = self.sess.run(tf.reduce_mean(self.objective, reduction_indices=0), feed_dict={self.input.input: in_tens,self.y_ph: in_tens_y})
                
                if valid_set is not None:
                    preds = self.predict(valid_set[0], prediction_sample_size=100)
                    valid_mse = self.loss_func(preds,y)
                    print('epoch: {} \n train error: {} \n valid_error: {} \n objective: {}\n\n\n'.format(epoch, train_mse, valid_mse, obj))
                else:
                    print('epoch: {} \n train error: {} \n objective: {}\n\n\n'.format(epoch, train_mse, obj))
                
                # record NN weights if the backup dir is set:
                if running_backup_dir is not None:
                        if valid_set is not None:
                            self.save(running_backup_dir+'runnung_tr{}_test{}.npy'.format(train_mse, valid_mse))
                        else:
                            self.save(running_backup_dir+'runnung_tr{}.npy'.format(train_mse))

            for i in range(nbatch):
                self.sess.run(self.optimizer,
                              feed_dict={
                                  self.input.input: in_tens[:, batchsize * i:batchsize * (i + 1), :],
                                  self.y_ph: in_tens_y[:, batchsize * i:batchsize * (i + 1), :]
                              })
            
            # shuffle data every shuffle_freq epochs
            if shuffle_freq is not None:
                if epoch % shuffle_freq == 0:
                    shuffle = np.random.permutation(in_tens.shape[1])
                    # not running gc right after shuffle causes memory leak
                    gc.collect()
                    in_tens = in_tens[:, shuffle, :]
                    in_tens_y = in_tens_y[:, shuffle, :]

    def save(self, path):
        '''
        save neural network weights
        :param path: path to weight file
        :return: None
        '''
        print('Saving weights...')
        
        t0 = time.time()
        arrs = []
        for mat in self.weights:
            arrs.append(self.sess.run(mat))
        np.save(path, arrs)
        t1 = time.time()
        
        print('Done in {} seconds'.format(t1-t0))
            
    def load(self, path):
        '''
        load weights of a neural network
        :param path: path to weights file
        :return:
        '''
        
        print('Loading weights...')
        
        t0 = time.time()
        arrs = np.load(path, encoding='utf-8')
        for mat, arr in zip(self.weights, arrs):
            self.sess.run(mat.assign(arr))
        t1 = time.time()
        print('Done in {} seconds'.format(t1-t0))
        
    def predict(self, X, prediction_sample_size=250, return_distrib=False):
        '''
        :param prediction_sample_size: size of prediction sample of variables
        :param return_distrib: whether to return a whole set of samples of only the mean value
        '''
        sample_size = self.sample_size

        n = int(prediction_sample_size / sample_size) + 1
        
        # prepare data for feeding into the NN
        X = np.repeat([X], sample_size, axis=0)
        
        preds = None
        
        
        if return_distrib:
            for i in range(n):
                part_pred = self.sess.run(self.output.output,
                                          feed_dict={self.input.input: X})
                if preds is not None:
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


