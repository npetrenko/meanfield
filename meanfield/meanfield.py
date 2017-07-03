import numpy as np
from math import pi
import gc
import theano as th
from theano import tensor as tt
import time
from tqdm import tqdm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne import updates
from theano import In
import traceback

rng = RandomStreams()
dtype = th.config.floatX

class Model(Network):

    def __init__(self, input, output, nnupdates=updates.adam, loss = 'mse', init_value_loss_repar = -3, loss_rapar_speed = 1):
        '''
        initialize nn model
        :param input: input node of the neural network
        :param output: output node of the network
        :param nnupdates: lasagne updates optimizer
        :param init_value_loss_repar: starting value of match/var loss scaling
        :return: model class instance
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

        self.updates = nnupdates
        self.batch_iterated = init_value_loss_repar
        self.repar_speed = loss_rapar_speed



        if loss == 'mse':
            def loss_func(preds, y):
                return np.sqrt(np.mean((preds-y) ** 2))
            def loss_func_nf(preds, y):
                return ((preds-y) ** 2)
            self.match_loss = tt.sum(((self.output.output - self.y_ph) ** 2)) / (2 * self.target_std_deviation ** 2)
        elif loss == 'crossentropy':
            def loss_func(preds, y):
                return np.mean(np.argmax(preds, axis=1) - np.argmax(y, axis=1) != 0)
            def loss_func_nf(preds, y):
                return np.argmax(preds, axis=1) - np.argmax(y, axis=1) != 0
            sh = output.output.shape
            out_resh = output.output.reshape((sh[0]*sh[1], sh[2]))
            self.match_loss = tt.nnet.categorical_crossentropy(out_resh, self.y_ph.reshape((sh[0]*sh[1], sh[2]))).sum()
        else:
            Exception('No correct loss specified. Use either "mse" of "crossentropy"')

        self.loss_func = loss_func
        self.loss_func_nf = loss_func_nf

        self.objective = self.match_loss + self.var_loss

    def fit(self, X, y, nepoch, batchsize, log_freq=100, valid_set = None, shuffle_freq = 1, running_backup_dir=None, scale_var_grad=1, logfile=None):

        if logfile:
            logs = open(logfile,'w')
        
        sample_size = self.sample_size
        
        # create input suitable for feeding into the input node
        in_tens = X.astype(dtype)
        in_tens_y = y.astype(dtype)
        
        nbatch = int(len(X)/batchsize)

        init_val = self.batch_iterated

        batch_iterated_ph = th.shared(np.array(self.batch_iterated, dtype=dtype), 'batch number placeholder')
        repar_speed = th.shared(np.array(self.repar_speed, dtype=dtype), 'repar speed constant')
        loss_scaler = 1/(1 + tt.exp(-(batch_iterated_ph-np.array(init_val, dtype))*repar_speed - np.array(init_val, dtype)))

        if not self.loss_final:

            loss = loss_scaler*self.match_loss + self.var_loss / (nbatch * 1.)
            loss = loss / sample_size
            self.loss = loss
            self.loss_final = True

            # remember batchsize in case of change
            self.batchsize = batchsize

        # reconfigure loss in case of batch size change
        if self.loss_final and self.batchsize != batchsize:
            loss = loss_scaler*self.match_loss + self.var_loss / (nbatch * 1.)
            loss /= sample_size
            self.loss = loss
            self.batchsize = batchsize

        obj_fun = th.function([ self.input.input, self.y,
                                In(self.input.sample_size, value=sample_size)], self.objective/self.input.sample_size)

        grad = th.grad(self.loss, self.weights)

        # grad_scaler = np.ones(shape=(len(self.weights),), dtype=dtype)
        for i in range(len(self.weights)):
            if i%2 == 1:
                grad[i] *= scale_var_grad

        # grad_scaler_th = tt.constant(grad_scaler, name='gradient scaler')

        # grad *= grad_scaler_th
        train = th.function([self.input.input, self.y,
                             In(self.input.sample_size, value=sample_size)], updates=self.updates(grad, self.weights))
        to_write = None
        try:
            for epoch in range(nepoch):
                # update the number of passed epochs
                self.batch_iterated += 1
                if loss_scaler.eval() < 1-0.0001:
                    batch_iterated_ph.set_value(np.array(self.batch_iterated, dtype=dtype))

                # print logs every log_freq epochs:
                if epoch % log_freq == 0:
                    preds = self.predict(in_tens, prediction_sample_size=100, train_mode=True)
                    train_mse = self.loss_func(preds, in_tens_y)
                    obj = obj_fun(in_tens, in_tens_y)

                    if valid_set is not None:
                        preds, std = self.predict(valid_set[0].astype(dtype), prediction_sample_size=100, train_mode=True, return_std=True)
                        valid_mse = self.loss_func(preds, valid_set[1])
                        losses = self.loss_func_nf(preds, valid_set[1])
                        corr = np.sum((losses - np.mean(losses))*(std-np.mean(std))/(np.std(std)*np.std(losses)))
                        logstr = 'epoch: {} \n  train error: {} \n  valid_error: {} \n  objective: {}\n  loss_scale: {}\n  loss-std corr: {}\n\n'.format(epoch,
                                                                                                                         train_mse, valid_mse, obj, loss_scaler.eval(), corr)
                    else:
                        logstr = 'epoch: {} \n  train error: {} \n  objective: {}\n  loss_scale: {}\n\n'.format(epoch, train_mse, obj, loss_scaler.eval())
                    #print('epoch: {} \n objective: {}\n\n\n'.format(epoch, obj))
                    print(logstr)
                    if logfile:
                        logs.write(logstr)

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
                        shuffle = np.random.permutation(in_tens.shape[0])
                        # not running gc right after shuffle causes memory leak
                        gc.collect()
                        in_tens = in_tens[shuffle, :]
                        in_tens_y = in_tens_y[shuffle, :]
        except (Exception, BaseException) as exc:
            to_write = traceback.format_exc(exc)
            raise exc
        finally:
            if logfile:
                if to_write:
                    logs.write(to_write)
                logs.close()

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

            pred_distrib = th.function([self.input.input,
                                        In(self.input.sample_size, value=prediction_sample_size)], self.output.output)
            pred = th.function([self.input.input,
                                In(self.input.sample_size, value=prediction_sample_size)], pred_op)
            predstd = th.function([self.input.input,
                                   In(self.input.sample_size, value=prediction_sample_size)], [pred_op, std_op])
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



