"""
PyTorch Linear Regression
"""
import torch as tc
import numpy as np

samples      = 1000
num_features = 2

# randn produce data with mean of 0 and variance of 1
x = tc.randn(samples, num_features, dtype=tc.float32)
# x = x / 100

wt = tc.tensor([2, -3.4])
bt = tc.tensor([4.2], dtype=tc.float32)

y = wt*x + bt
y = tc.sum(y, dim=1)

def dataIter(x, y, batchsize):
    samples = len(x)
    indices = list(range(samples))
    np.random.shuffle(indices)
    for i in range(0, samples, batchsize):
        x1 = x[i:min(i+batchsize, samples)]
        y1 = y[i:min(i+batchsize, samples)]
        yield x1, y1

w = tc.tensor(np.random.normal(0, 0.01, (num_features, )), dtype=tc.float32)
b = tc.zeros(1, dtype=tc.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def out(x):
    global w, b
    return tc.sum(x*w + b, dim=1)

def loss(y, y_hat):
    return (1/2)*(y-y_hat)**2

def sgd(params, lr, batchsize):
    for param in params:
        param.data -= lr*param.grad / batchsize

iters = 3
batchsize = 10
lr = 0.03
for iter in range(iters):
    for bx, by in dataIter(x, y, batchsize):
        y_hat = out(bx)
        ls    = loss(by, y_hat).sum()
        ls.backward()
        sgd([w, b], lr, batchsize)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(y, out(x))
    print("epoch {}".format(iter), " loss: ", train_l.mean().item())

print(wt, w)
print(bt, b)