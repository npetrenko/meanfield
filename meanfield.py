import numpy as np
from math import pi
import gc
import theano as th
from theano import tensor as tt
import time
from tqdm import tqdm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

rng = RandomStreams()
dtype = th.config.floatX

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
    def __init__(self, dim, input_layer, act=tt.nnet.relu, prior=3, name=''):
        '''
        :param dim: number of neurons in the layer
        :param input_layer: input layer calss object
        :param act: tensorflow activation function
        :param prior: standard deviation of prior
        '''
        #prior = tf.Variable(prior, dtype=tf.float32, trainable=False, name='prior_' + name)
        prior = th.shared(prior, name='prior_' + name)
        
        self.weights = input_layer.weights

        #sample_size = tf.shape(input_layer.output)[0]
        sample_size = input_layer.output.shape[0]
        self.dim = dim
        self.inp_dim = input_layer.dim

        shape = [self.inp_dim, self.dim]
        #self.mean = tf.Variable(np.random.normal(size=shape, scale=0.01, loc=0), dtype=tf.float32)
        self.mean = th.shared(np.random.normal(size=shape, scale=0.01, loc=0).astype(dtype))
        #self.sigma = tf.Variable(np.random.normal(size=shape, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma = th.shared(np.random.normal(size=shape, scale=0.01, loc=0).astype(dtype))
        self.weights += [self.mean, self.sigma]
        
        #self.sigma = tf.log(tf.exp(self.sigma + self.initial_sigma) + 1)
        self.sigma = tt.log1p(tt.exp(self.sigma + self.initial_sigma))

        #self.mean_const = tf.Variable(np.random.normal(size=dim, scale=0.01, loc=0), dtype=tf.float32)
        self.mean_const = th.shared(np.random.normal(size=dim, scale=0.01, loc=0).astype(dtype))
        #self.sigma_const = tf.Variable(np.random.normal(size=dim, scale=0.01, loc=0), dtype=tf.float32)
        self.sigma_const = th.shared(np.random.normal(size=dim, scale=0.01, loc=0).astype(dtype))
        self.weights += [self.mean_const, self.sigma_const]
        
        #self.sigma_const = tf.log(tf.exp(self.sigma_const + self.initial_sigma) + 1)
        self.sigma_const = tt.log1p(tt.exp(self.sigma_const + self.initial_sigma))

        # sample of activation matrixes and biases
        #activation_matrix = tf.random_normal(shape=[sample_size] + shape, dtype=tf.float32,
        #                                     stddev=1) * self.sigma + self.mean
        activation_matrix = rng.normal(size=[sample_size] + shape,dtype=dtype) * self.sigma + self.mean
        #bias = tf.random_normal(shape=[sample_size, 1, dim], dtype=tf.float32, stddev=1) * self.sigma_const + self.mean_const
        bias = rng.normal(size=[sample_size, 1, dim], dtype=dtype) * self.sigma_const + self.mean_const

        # calculate matrix multiplication for each sample and add bias

        #temp = tf.transpose(tf.tensordot(input_layer.output, activation_matrix, axes=[[2], [1]]), perm=[0,2,1,3])
        #temp = tt.transpose(tt.tensordot(input_layer.output, activation_matrix, axes=[[2],[1]]), axes=[0,2,1,3])

        #ind = tf.stack([tf.range(tf.shape(temp)[0])]*2, axis=1)
        #matrixes = tf.gather_nd(temp, indices=ind)
        #matrixes = tt.diagonal(temp, axis1=0, axis2=1)
        matrixes = tt.batched_dot(input_layer.output, activation_matrix)

        # index = np.array([[i,i] for i in range(sample_size)], dtype='int')
        #self.logits = tf.gather_nd(tf.tensordot(input_layer.output, activation_matrix, axes=[[2], [1]]), indices=index) + bias
        self.logits = matrixes + bias
        shape = self.logits.shape
        self.output = act(self.logits.reshape((shape[0]*shape[1], shape[2]))).reshape(shape)
        print(self.output)

        # ...
        #l1 = tf.reduce_sum(-tf.log(np.sqrt(2 * pi) * self.sigma)) + tf.reduce_sum( -tf.log(np.sqrt(2 * pi) * self.sigma_const) )
        l1 = tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma)) + tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma_const))
        # prior loss
        #l2 = (-tf.reduce_sum((activation_matrix ** 2)) - tf.reduce_sum((bias ** 2))) / (2 * prior**2) #probably i need a square here
        l2 = (-tt.sum((activation_matrix ** 2)) - tt.sum((bias ** 2))) / (2 * prior ** 2)
        # feeding loss to the next layer
        self.loss = l1 - l2
        
        self.loss += input_layer.loss


class Input(Layer):
    def __init__(self, dim):
        '''
        :param dim: number of sensors in the first layer
        '''
        #self.input = tf.placeholder(shape=(None, None,dim), dtype=tf.float32)
        self.input = tt.matrix(name='input', dtype=dtype) #shape=(None, None, dim)
        self.sample_size = th.shared(self.sample_size)
        self.output = tt.tile(self.input, (self.sample_size,1,1))
        self.dim = dim
        self.loss = 0
        self.weights = []


class Model(Network):

    def terminate(self):
        '''
        close tensorflow session
        '''
        self.sess.close()

    def __init__(self, input, output, updates = lasagne.updates.adam, loss = 'mse'):
        '''
        :param input: input node of the neural network
        :param output: output node of the network
        :param optimizer: tensorflow optimizer
        '''

        self.weights = output.weights
        sample_size = self.sample_size
        
        # store part of loss dependent on variables:
        self.var_loss = output.loss

        # create placeholder for target values
        #self.y_ph = tf.placeholder(shape=(None, None, output.dim), dtype=tf.float32)
        self.y = tt.matrix('y',dtype)
        self.y_ph = tt.tile(self.y, (self.sample_size,1,1))
        #tt.tensor3(name='y_ph', dtype=dtype) #shape=(None, None, output.dim)

        # parameter which helps to build the final loss
        self.loss_final = False

        self.input = input
        self.output = output

        self.updates = updates



        if loss == 'mse':
            def loss_func(preds, y):
                return np.sqrt(np.mean((preds-y) ** 2))
            self.match_loss = tt.sum(((self.output.output - self.y_ph) ** 2)) / (2 * self.target_std_deviation ** 2)
        elif loss == 'crossentropy':
            def loss_func(preds, y):
                return np.mean(np.argmax(preds, axis=1) - np.argmax(y, axis=1) != 0)
            sh = output.output.shape
            out_resh = output.output.reshape((sh[0]*sh[1],sh[2]))
            self.match_loss = tt.nnet.categorical_crossentropy(out_resh, self.y_ph.reshape((sh[0]*sh[1],sh[2]))).sum()
            #self.match_loss *= tt.cast(sh[0]*sh[1], dtype)
        else:
            Exception('No correct loss specified. Use either "mse" of "crossentropy"')

        self.loss_func = loss_func

        self.objective = self.match_loss + self.var_loss

    def fit(self, X, y, nepoch, batchsize, log_freq=100, valid_set = None, shuffle_freq = 1, running_backup_dir=None):
        
        sample_size = self.sample_size
        
        # create input suitable for feeding into the input node
        #in_tens = np.repeat([X], sample_size, axis=0).astype(dtype)
        #in_tens_y = np.repeat([y], sample_size, axis=0).astype(dtype)
        in_tens = X.astype(dtype)
        in_tens_y = y.astype(dtype)
        
        nbatch = int(len(X)/batchsize)

        if not self.loss_final:

            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            #self.optimizer = self.optimizer.minimize(self.loss)
            self.loss_final = True

            # remember batchsize in case of change
            self.batchsize = batchsize

        # reconfigure loss in case of batch size change
        if self.loss_final and self.batchsize != batchsize:
            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.batchsize = batchsize
            #self.optimizer = self.optimizer.minimize(self.loss)
            #
            #
            #do we need a var init here?

        obj_fun = th.function([self.input.input, self.y], self.objective.mean())

        train = th.function([self.input.input, self.y], updates=self.updates(self.loss, self.weights))

        for epoch in range(nepoch):
            
            # print logs every log_freq epochs:
            if epoch % log_freq == 0:
                preds = self.predict(in_tens, prediction_sample_size=100, train_mode=True)
                train_mse = self.loss_func(preds, y)
                #obj = self.sess.run(tf.reduce_mean(self.objective), feed_dict={self.input.input: in_tens,self.y_ph: in_tens_y})
                obj = obj_fun(in_tens, in_tens_y)
                
                if valid_set is not None:
                    preds = self.predict(valid_set[0].astype(dtype), prediction_sample_size=100, train_mode=True)
                    valid_mse = self.loss_func(preds,valid_set[1])
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
                train(in_tens[batchsize * i:batchsize * (i + 1), :],
                      in_tens_y[batchsize * i:batchsize * (i + 1), :])
            
            # shuffle data every shuffle_freq epochs
            if shuffle_freq is not None:
                if epoch % shuffle_freq == 0:
                    shuffle = np.random.permutation(in_tens.shape[1])
                    # not running gc right after shuffle causes memory leak
                    gc.collect()
                    in_tens = in_tens[shuffle, :]
                    in_tens_y = in_tens_y[shuffle, :]

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
            arrs.append(mat.get_value())
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
        arrs = np.load(path, encoding='bytes')
        for mat, arr in zip(self.weights, arrs):
            mat.set_value(arr)
        t1 = time.time()
        print('Done in {} seconds'.format(t1-t0))
        
    def predict(self, X, prediction_sample_size=250, batchsize = 360, return_distrib=False, train_mode=False, return_std=False):
        '''
        :param prediction_sample_size: size of prediction sample of variables
        :param return_distrib: whether to return a whole set of samples of only the mean value
        '''
        if not train_mode:
            bar = tqdm(total=100)
        # exception handling required for tqdm to work correctly
        try:
            pred_op = tt.mean(self.output.output, axis=0)
            std_op =  tt.sum(tt.sqrt(tt.mean((self.output.output - pred_op)**2, axis=0)), axis=-1)

            pred_distrib = th.function([self.input.input], self.output.output)
            pred = th.function([self.input.input], pred_op)
            predstd = th.function([self.input.input], [pred_op, std_op])
            # prepare data for feeding into the NN
            nbatch = int(len(X)/batchsize) + 1

            temp = []
            stds = []
            for i in range(nbatch):
                if not train_mode:
                    bar.update(100./nbatch)
                if (i+1)*batchsize > len(X)-1:
                    batch = X[i*batchsize:].astype(dtype)
                else:
                    batch = X[i * batchsize : (i + 1) * batchsize].astype(dtype)
                if return_distrib:
                    #preds = self.sess.run(self.output.output, feed_dict={self.input.input: batch})
                    preds = pred_distrib(batch)
                else:
                    if return_std:
                        #preds, std = self.sess.run([pred_op, std_op], feed_dict = {self.input.input : batch})
                        preds, std = predstd(batch)
                        stds.append(std)
                    else:
                        preds = pred(batch)
                temp.append(preds)
                if (i + 1) * batchsize > len(X) - 1:
                    break
        finally:
            if not train_mode:
                bar.close()
        if return_distrib:
            return np.concatenate(temp, axis=1)
        else:
            if return_std:
                return np.concatenate(temp, axis=0), np.concatenate(stds, axis=0)
            else:
                return np.concatenate(temp, axis=0)



