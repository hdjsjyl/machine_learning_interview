import numpy as np

class Layer:
    def __init__(self, names):
        self.names = names
        self.paras = None
        self.grads = None

    def forward(self, inputs):
        raise NotImplementedError

    def backword(self, grads):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, num_in, num_out, w_init=XavierUniformInit(), b_init=ZeroInit()):
        super(Dense, self).__init__('Linear')
        self.inputs = None
        self.paras = {}
        self.paras['w'] = w_init([num_in, num_out])
        self.paras['b'] = b_init([1, num_out])

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs,  self.paras['w']) + self.paras['b']

    def backward(self, grads):
        self.grads['w'] = grads * self.inputs
        self.grads['b'] = np.sum(grads, axis=0)
        return np.dot(grads, self.paras['w'].T)

class Activation:
    def __init__(self, name):
        self.name  = name
        self.input = None

    def forward(self, input):
        return self.func(input)

    def func(self, input):
        raise NotImplementedError

    def backward(self, grads):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__('ReLU')

    def func(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self, grads):
        return (self.input > 0) * grads
