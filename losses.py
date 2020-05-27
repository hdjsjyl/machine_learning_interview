import numpy as np

class BaseLoss:
    def loss(self, y_hat, y):
        raise NotImplementedError

    def grad(self, y_hat, y):
        raise NotImplementedError


class SoftmaxCrossEntropyLoss(BaseLoss):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def loss(self, y_hat, y):
        m = y_hat.shape[0]
        exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        ps =  exps / np.sum(exps, axis=1, keepdims=True)
        # cs = -np.log(np.sum(ps*y, axis=1))
        cs = -np.sum(y*np.log(ps), axis=1)
        return np.sum(cs)/m

    def grad(self, y_hat, y):
        m = y_hat.shape[0]
        y_copy = np.copy(y_hat)
        y_copy -= y
        return y_copy/m

