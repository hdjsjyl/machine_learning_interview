import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io

def get_case_data(number=200):
    np.random.seed(6)
    x, y = sklearn.datasets.make_moons(number, noise=0.3)
    return x, y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def model(x, w1, w2, b):
    # print(x.shape)
    # print(w1, w2, b)
    z = (x[:, 0])[:, None]*w1 + (x[:, 1])[:, None]*w2 + b
    # print(z.shape)
    # z = np.sum(z, axis=1)
    return sigmoid(z)

def sigmoidCrossEntropy(y_hat, y):
    l1 = y*np.log(y_hat) + (1-y)*np.log(1-y_hat)
    l  = np.sum(-1*l1, axis=0)/len(y)
    return l

def optimizer(x, y_hat, y):
    dw1 = (y_hat-y)*(x[:, 0])[:, None]
    dw1 = np.sum(dw1, axis=0)/len(x)
    dw1 = (dw1)[:, None]
    dw2 = (y_hat-y)*(x[:, 1])[:, None]
    dw2 = np.sum(dw2, axis=1)/len(x)
    dw2 = (dw2)[:, None]
    db   = (y_hat - y)
    db   = np.sum(db)/len(x)
    # db   = db
    return dw1, dw2, db

def accuracy(x, y, w1, w2, b):
    # print('accuracy: ')
    y_hat = model(x, w1, w2, b)
    masks = (y_hat > 0.5)
    masks2 = (masks == y)
    return np.sum(masks2)/len(x)

def dataLoader(x, y, batchsize):
    indices = [i for i in range(len(x))]
    np.random.shuffle(indices)
    for i in range(len(x)):
        x1 = x[indices[i:min(i+batchsize, len(x))]]
        y1 = y[indices[i:min(i+batchsize, len(x))]]
        yield x1, y1

if __name__ == '__main__':
    x, y = get_case_data()
    y = (y)[:, None]
    iters = 3000
    lr    = 0.05
    batchsize = 32
    w1 = 1
    w2 = 1
    b = 0
    lambd = 0.01
    for iter in range(iters):
        tmp = 0
        acc2 = 0
        for bx, by in dataLoader(x, y, batchsize):
            output = model(bx, w1, w2, b)
            loss = sigmoidCrossEntropy(output, by)
            l2_regularization_cost = loss + lambd * (np.square(w1) + np.square(w2))/batchsize
            dw1, dw2, db = optimizer(bx, output, by)
            ## l2 regularization gradient
            w1 -= lr*lambd * w1
            w1 -= lr*dw1[0, 0]
            w2 -= lr*lambd * w2
            w2 -= lr*dw2[0, 0]
            b  -= lr*db
            acc = accuracy(bx, by, w1, w2, b)
            tmp += l2_regularization_cost
            acc2 += acc
        print('iter: {}'.format(iter), 'loss: {}'.format(tmp/(len(x)/batchsize)), 'acc: {}'.format(acc2/(len(x)/batchsize)))