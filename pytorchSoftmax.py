import torch as tc
import torchvision as tv
import torchvision.transforms as trans
import torch.utils.data as Data
import numpy as np
import sys

## sigmoid with crossEntropy and softmax with crossEntropy
## y1 = w*x + b
## y2 = sigmoid or softmax (y1)
## y3 = crossEntropy(y2)
## dy3/dy1 = y1 - y

mnist_train = tv.datasets.FashionMNIST(root='./data', train=True, download=False, transform=trans.ToTensor())
mnist_test  = tv.datasets.FashionMNIST(root='./data', train=False, download=False, transform=trans.ToTensor())

trainIter = Data.DataLoader(mnist_train, batch_size=256, shuffle=True)
testIter  = Data.DataLoader(mnist_test, batch_size=256, shuffle=False)

num_input  = 784
num_output = 10

w = tc.tensor(np.random.normal(0, 0.01, (num_input, num_output)), dtype=tc.float32)
b = tc.zeros(num_output, dtype=tc.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(x):
    x = x - tc.max(x)
    expx = x.exp()
    return expx / expx.sum(dim=1, keepdim=True)

def net(x):
    global w, b
    return softmax(tc.mm(x.view((-1, num_input)), w) + b)

def crossEntropy(y_hat, y):
    # print(y_hat.shape, y.shape)
    return -tc.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y.view(-1, 1)).float().mean().item()

def evaluate_accruacy(data_iter, net):
    acc_sum = 0
    n = 0
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n

def sgd(parameters, lr, batchsize):
    for param in parameters:
        param.data -= lr*param.grad/batchsize

def train_ch3(net, train_iter, test_iter, loss, params, num_epochs=5, batch_size=256, lr=0.1, optimizer=None):
    # optimizer = tc.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            # if optimizer is not None:
            #     optimizer.zero_grad()
            # elif params is not None and params[0].grad is not None:
            l.backward()
            sgd(params, lr, batch_size)
            for param in params:
                param.grad.data.zero_()
            # if optimizer is None:
            #     d2l.sgd(params, lr, batch_size)
            # else:
            # optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accruacy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# print(w, b)
train_ch3(net, trainIter, testIter, crossEntropy, [w, b])