import numpy as np


class BaseOptimizer:
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd

    def compute_step(self, grads, params):
        step = list()
        # flatten all gradients
        flatten_grads = [np.ravel(v) for grad in grads for v in grad.values()]
        # compute step
        flatten_step = self._compute_step(flatten_grads)
        # reshape gradients
        p = 0
        for param in params:
            layer = {}
            for k, v in param.items():
                block = np.prod(v.shape)
                _step = flatten_step[p:p+block].reshape(v.shape)
                ## ??
                _step -= self.wd*v
                layer[k] = _step
                ## ??
                p += block
            step.append(layer)
        return step

    def _compute_step(self, input):
        raise NotImplementedError


