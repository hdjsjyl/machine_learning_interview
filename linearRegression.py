import numpy as np
import matplotlib.pyplot as plt

## y = x1 * w1 + x2 * w2 + b

samples = 1000
x = np.random.normal(0, 1, (samples, 2))
w1 = -2.0
w2 = 4.0
b1 = 1.0
y = x[:, 0] * w1 + x[:, 1] * w2 + b1
tmpy = (y)[:, None]

# plt.scatter(x, tmpy, alpha=0.7, s=60) # 绘制散点图
# plt.title('Random Scatter')
# plt.show()
## intialization
w11 = 0
w22 = 0
b2 = 0

# square loss
def loss(y, y_hat):
    return (y-y_hat)**2/2

## batch gradient descent
def bgd(y, y_hat, x1, x2, lr):
    ## batch gradient descent
    ## (y-y_hat)**2/2
    ## y_hat = w1 * x1 + w2 * x2 + b
    global w11, w22, b2
    w11 = w11 - lr*(sum(-1*(y-y_hat)*x1)/len(y))
    w22 = w22 - lr*(sum(-1*(y-y_hat)*x2)/len(y))
    b2  = b2 - lr*(sum(-1*(y-y_hat)*1)/len(y))

def loader(x, y, batch):
    global samples
    indices = [i for i in range(samples)]
    indices = np.array(indices)
    np.random.shuffle(indices)
    for i in range(0, samples, batch):
        x1 = x[indices[i:min(i+batch, samples)]]
        y1 = y[indices[i:min(i+batch, samples)]]
        yield x1, y1

def output(bx, w1, w2, b):
    return bx[:, 0] * w1 + bx[:, 1] * w2 + b

if __name__ == '__main__':
    batch = 10
    iters = 100
    for i in range(iters):
        tmp = 0
        for bx, by in loader(x, y, batch):
            outputs = output(bx, w11, w22, b2)
            losses = loss(by, outputs)
            tmp += losses/len(bx)
            bgd(by, outputs, bx[:, 0], bx[:, 1], 0.01)

        print('iter: ', i, sum(tmp)/(samples//batch))

print(w1, w11)
print(w2, w22)
print(b1, b2)