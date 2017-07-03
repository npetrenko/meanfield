from meanfield.layers.base import *

class Input(Layer):
    def __init__(self, dim):
        '''
        initialize an input layer
        :param dim: number of sensors in the first layer
        :return: instance of input layer calss
        '''
        self.input = tt.matrix(name='input', dtype=dtype)
        self.sample_size = tt.iscalar('sample size controller')
        self.output = tt.tile(self.input, (self.sample_size, 1, 1))
        self.dim = dim
        self.loss = 0
        self.weights = []
