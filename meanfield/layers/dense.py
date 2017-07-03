from meanfield.layers.base import *

class Dense(Layer):
    initial_sigma = -6

    def __init__(self, dim, input_layer, act=tt.nnet.relu, prior=3, name=''):
        '''
        initialize dense layer
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
        activation_matrix = rng.normal(size=[sample_size] + shape, dtype=dtype) * self.sigma + self.mean

        bias = rng.normal(size=[sample_size, 1, dim], dtype=dtype) * self.sigma_const + self.mean_const

        # calculate matrix multiplication for each sample and add bias

        #matrixes = tt.batched_dot(input_layer.output, activation_matrix)

        mb, _ = th.scan(lambda i, m1, m2: tt.dot(m1[i], m2[i]), sequences=[tt.arange(sample_size)],
                        non_sequences=[input_layer.output, activation_matrix])

        # index = np.array([[i,i] for i in range(sample_size)], dtype='int')

        self.logits = mb + bias
        shape = self.logits.shape
        self.output = act(self.logits.reshape((-1,shape[-1]))).reshape(shape)
        print(self.output)

        # ...
        l1 = tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma)) + tt.sum(-tt.log(np.sqrt(2 * pi) * self.sigma_const))
        # prior loss
        l2 = (-tt.sum((activation_matrix ** 2)) - tt.sum((bias ** 2))) / (2 * prior ** 2)
        # feeding loss to the next layer
        self.loss = l1 - l2
        
        self.loss += input_layer.loss
