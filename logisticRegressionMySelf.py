# step1 load dataset
# step2 make the dataset iterable
# step3 create model class
# step4 instantiate model class
# step5 instantiate loss function
# step6 instantiate optimizer
# step7 train model
# step8 test mode


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# step2 make the dataset iterable
def dataLoader(x, y, iters):
    for iter in range(iters):
        lens = len(x)
        indices = [i for i in range(lens)]
        np.random.shuffle(indices)
        x1 = x[indices]
        y1 = y[indices]
        yield x1, y1

# step3 creat model x and step4 instantiate model
def model(w, x, b):
    res = np.sum(x*w.T+b, axis=1)
    return res

# step5 instantiate loss function
def sigmoid(x):
     res = 1/(1+np.exp(-x))
     res = res.reshape(res.shape[0], 1)
     return res

def sigmoidCross(y_hat, y):
    n = len(y)
    l1 = y*np.log(y_hat)+(1-y)*np.log((1-y_hat))
    l1  = -(1/n)*sum(l1)
    return l1

# step6 instantiate optimizer
def optimizer(x, y_hat, y, lr, w, b):
    n = len(x)
    dw    = (1.0/n) *((y_hat-y)*x)
    dw    = np.sum(dw, axis=0)
    dw    = dw.reshape((30, 1))
    # w     = w - lr*dw
    db    = (y_hat-y)
    db    = np.sum(db, axis=0)/len(x)
    # db    = db.reshape((30, 1))
    # b     = b - lr*db
    return dw, db

def accuracy(bx, by, w, b):
    out = model(w, bx, b)
    sout = sigmoid(out)
    # preCls = np.argmax(sout, axis=1)
    masks = (sout > 0.5)
    ss = sum(masks == by)
    return ss/len(bx)

def add_ones(X):
    ones=np.ones((X.shape[0],1))
    X_with_ones=np.hstack((ones, X))
    return X_with_ones

if __name__ == '__main__':
    # step1
    dataset = load_breast_cancer()
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    data['cancer'] = [dataset.target_names[t] for t in dataset.target]

    x = dataset.data
    y = dataset.target
    n_features = x.shape[1]

    ## data normalization
    mean = x.mean(axis=0)
    # mean = (mean)[:, None]
    std = x.std(axis=0)
    # std = (std)[:, None]
    xnorm = (x - mean) / std

    # X_with_ones = add_ones(xnorm)

    X_train, X_test, y_train, y_test = train_test_split(xnorm, y, test_size=0.3, random_state=12345)

    y_train=y_train.reshape((y_train.shape[0], 1))
    y_test=y_test.reshape((y_test.shape[0], 1))

    # step7 train model
    lr = 1e-1
    iters = 1500
    # for bx, by in dataLoader(xnorm, y, iters):
    w = np.random.randn(n_features, 1)
    b = 0
    for iter in range(iters):
        indices = [i for i in range(len(X_train))]
        np.random.shuffle(indices)
        # bx = xnorm[indices]
        # by = y_train[indices]
        bx = X_train[indices]
        by = y_train[indices]
        # print(w[:10], b)
        y_hat1 = model(w, bx, b)
        # print(y_hat.shape)
        y_hat2 = sigmoid(y_hat1)
        loss = sigmoidCross(y_hat2, by)
        dw, db = optimizer(bx, y_hat2, by, lr, w, b)
        w -= lr*dw
        b -= lr*db
        acc = accuracy(bx, by, w, b)
        print(loss, acc)

    testacc = accuracy(X_test, y_test, w, b)
    print(testacc)