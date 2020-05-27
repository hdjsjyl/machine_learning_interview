import numpy as np

# step1: load dataset
# step2: make it iterable
# step3: create the model class
# step4: instantiate the model class
# step5: instantiate the loss function
# step6: instantiate the optimizer
# step7: train the model
# step8: test the model

# y = w1*x1 + w2*x2 + b
num_samples = 1000
x = np.random.randn(1000, 2)
w1 = -2.0
w2 = 4.2
b = 1
y = w1*x[:, 0] + w2*x[:, 1] + b
y = (y)[:, None]
noise = np.random.normal(0, 0.001, (1000, 1))
y += noise

# parameter initialization
ws = np.random.normal(0, 0.001, (2, 1))
bs = 0

# data loader
def dataLoader(x, y, batch, samples):
    indexs = [i for i in range(samples)]
    np.random.shuffle(indexs)
    for i in range(0, samples, batch):
        x1 = x[i:min(i+batch, samples), :]
        y1 = y[i:min(i+batch, samples), :]
        yield x1, y1

# output
def output(x, w, b):
    y_hat = w[0, 0]*x[:, 0] + w[1, 0]*x[:, 1] + b
    return y_hat

# loss function:
def loss(y_hat, y):
    l = (1/2)*(y_hat - y)**2
    return l

## optimizer
def mnsgd(y, y_hat, x):
    # y_hat = w*x + b
    # l = (1/2)(y_hat-y)**2
    # print(y_hat.shape, y.shape, x.shape)
    dw = sum((y_hat - y)*x)/len(y)
    dw = (dw)[:, None]
    db = sum((y_hat - y))/len(y)
    db = (db)[:, None]
    return dw, db


if __name__ == '__main__':
    # global x, y, ws, bs, w1, w2, b
    batch = 10
    lr = 0.001
    iters = 100
    for i in range(iters):
        ls = 0
        num = 0
        # acc = 0
        for bx, by in dataLoader(x, y, batch, samples=num_samples):
            y_hat = output(bx, ws, bs)
            y_hat = (y_hat)[:, None]
            l = loss(y_hat, by)
            ls += np.sum(l)
            num += len(bx)
            dw, db = mnsgd(by, y_hat, bx)
            ws -= lr*dw
            bs -= lr*db[0]
        print('iter: {}'.format(i), 'loss: {}'.format(ls/num))

print(ws, bs)
print([w1, w2], b)