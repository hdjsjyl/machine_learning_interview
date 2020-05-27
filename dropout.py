import numpy as np


def dropout(x, parameters):
    p = parameters['p']
    mode = parameters['mode']
    seed = parameters['seed']

    np.random.seed(seed)

    mask, out = None, None

    if mode == 'train':
        ps = np.random.rand(*x.shape)
        ps = ps.reshape(x.shape)
        masks = (ps>=p)/(1-p)
        out = masks*out
    elif mode == 'test':
        out = p*out

    return out

def dropoutbackward(dout, mask, mode):
    out = None
    if mode == 'train':
        out = dout*mask
    elif mode == 'test':
        out = dout

    return out