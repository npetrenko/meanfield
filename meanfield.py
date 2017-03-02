import numpy as np
from math import pi
import gc
import theano as th
from theano import tensor as tt
import time
from tqdm import tqdm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from theano import In

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

        prior = th.shared(prior, name='prior_' + name)
        
        self.weights = input_layer.weights


        sample_size = input_layer.output.shape[0]
        self.dim = dim
        self.inp_dim = input_layer.dim

        shape = [self.inp_dim, self.dim]

        self.mean = th.shared(np.random.normal(size=shape, scale=0.01, loc=0).astype(dtype))
        self.sigma = th.shared(np.random.normal(size=shape, scale=0.01, loc=0).astype(dtype))
        self.weights += [self.mean, self.sigma]
        self.sigma = tt.log1p(tt.exp(self.sigma + self.initial_sigma))


        self.mean_const = th.shared(np.random.normal(size=dim, scale=0.01, loc=0).astype(dtype))
        self.sigma_const = th.shared(np.random.normal(size=dim, scale=0.01, loc=0).astype(dtype))
        self.weights += [self.mean_const, self.sigma_const]
        self.sigma_const = tt.log1p(tt.exp(self.sigma_const + self.initial_sigma))

        # sample of activation matrixes and biases
        activation_matrix = rng.normal(size=[sample_size] + shape,dtype=dtype) * self.sigma + self.mean

        bias = rng.normal(size=[sample_size, 1, dim], dtype=dtype) * self.sigma_const + self.mean_const

        # calculate matrix multiplication for each sample and add bias

        matrixes = tt.batched_dot(input_layer.output, activation_matrix)

        # index = np.array([[i,i] for i in range(sample_size)], dtype='int')

        self.logits = matrixes + bias
        shape = self.logits.shape
        self.output = act(self.logits.reshape((shape[0]*shape[1], shape[2]))).reshape(shape)
        print(self.output)

        # ...
        l1 = tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma)) + tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma_const))
        # prior loss
        l2 = (-tt.sum((activation_matrix ** 2)) - tt.sum((bias ** 2))) / (2 * prior ** 2)
        # feeding loss to the next layer
        self.loss = l1 - l2
        
        self.loss += input_layer.loss


class Input(Layer):
    def __init__(self, dim):
        '''
        :param dim: number of sensors in the first layer
        '''
        self.input = tt.matrix(name='input', dtype=dtype) #shape=(None, None, dim)
        self.sample_size = tt.iscalar('sample size controller')
        self.output = tt.tile(self.input, (self.sample_size,1,1))
        self.dim = dim
        self.loss = 0
        self.weights = []


class Model(Network):

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
        self.y = tt.matrix('y',dtype)
        self.y_ph = tt.tile(self.y, (input.sample_size,1,1))

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
        else:
            Exception('No correct loss specified. Use either "mse" of "crossentropy"')

        self.loss_func = loss_func

        self.objective = self.match_loss + self.var_loss

    def fit(self, X, y, nepoch, batchsize, log_freq=100, valid_set = None, shuffle_freq = 1, running_backup_dir=None):
        
        sample_size = self.sample_size
        
        # create input suitable for feeding into the input node
        in_tens = X.astype(dtype)
        in_tens_y = y.astype(dtype)
        
        nbatch = int(len(X)/batchsize)

        if not self.loss_final:

            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.loss_final = True

            # remember batchsize in case of change
            self.batchsize = batchsize

        # reconfigure loss in case of batch size change
        if self.loss_final and self.batchsize != batchsize:
            loss = self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.batchsize = batchsize
            #
            #
            #do we need a var init here?

        obj_fun = th.function([ self.input.input, self.y,
                                In(self.input.sample_size, value=sample_size)], self.objective.mean())

        train = th.function([ self.input.input, self.y,
                              In(self.input.sample_size, value=sample_size)], updates=self.updates(self.loss, self.weights))

        for epoch in range(nepoch):
            
            # print logs every log_freq epochs:
            if epoch % log_freq == 0:
                #preds = self.predict(in_tens, prediction_sample_size=100, train_mode=True)
                #train_mse = self.loss_func(preds, in_tens_y)
                obj = obj_fun(in_tens, in_tens_y)
                
                #if valid_set is not None:
                    #preds = self.predict(valid_set[0].astype(dtype), prediction_sample_size=100, train_mode=True)
                    #valid_mse = self.loss_func(preds, valid_set[1])
                    #print('epoch: {} \n train error: {} \n valid_error: {} \n objective: {}\n\n\n'.format(epoch, train_mse, valid_mse, obj))
                #else:
                    #print('epoch: {} \n train error: {} \n objective: {}\n\n\n'.format(epoch, train_mse, obj))
                print('epoch: {} \n objective: {}\n\n\n'.format(epoch, obj))
                
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

            pred_distrib = th.function([self.input.input, In(self.input.sample_size, value=prediction_sample_size)], self.output.output)
            pred = th.function([self.input.input, In(self.input.sample_size, value=prediction_sample_size)], pred_op)
            predstd = th.function([self.input.input, In(self.input.sample_size, value=prediction_sample_size)], [pred_op, std_op])
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
                    preds = pred_distrib(batch)
                else:
                    if return_std:
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



