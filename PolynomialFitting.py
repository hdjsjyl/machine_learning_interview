import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
# import d2lzh_pytorch as d2l

## y = 1.2x - 3.4square(x) + 5.6cube(x) + 5
draw = False
num_samples = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5
_x = np.random.randn(num_samples*2, 1)
_x = tc.tensor(_x, dtype=tc.float32)
_xs = tc.cat((_x, tc.pow(_x, 2), tc.pow(_x, 3)), 1)
_y = true_w[0] * _x + true_w[1] * _x**2 + true_w[2] * _x**3 + true_b

def draw(x, y):
    print(x.shape, y.shape)
    plt.scatter(x, y)
    plt.show()

iters = 20
loss = tc.nn.MSELoss()
# normal fitting
# net = tc.nn.Linear(3, 1)

# under fitting
# net = tc.nn.Linear(1, 1)

# overfitting
net = tc.nn.Linear(3, 1)

tc.nn.init.normal_(net.weight, mean=0, std=0.01)
tc.nn.init.constant_(net.bias, val=0)

optimizer = tc.optim.SGD(net.parameters(), lr=0.01)

def dataiter(train_features, train_labels, test_features, test_labels):
    traindataset = Data.TensorDataset(train_features, train_labels)
    trainIter = Data.DataLoader(traindataset, batch_size=10, shuffle=True)
    testDataset = Data.TensorDataset(test_features, test_labels)
    testIter  = Data.DataLoader(testDataset, batch_size=10, shuffle=False)
    return trainIter, testIter

## normal fitting
# train_features = _xs[:100, :]
# train_labels = _y[:100]
# test_features = _xs[100:200, :]
# test_labels = _y[100:200]

## under fitting
# train_features = _x[:100, :]
# test_features = _x[100:200, :]
# test_features = _x[100:200, :]
# test_labels = _y[100:200]

## over fitting
train_features = _xs[:2, :]
train_labels   = _y[:2, :]
test_features = _xs[100:200, :]
test_labels = _y[100:200]

trains, tests = dataiter(train_features, train_labels, test_features, test_labels)

trainl = []
testl = []
for i in range(iters):
    for x, y in trains:
        # print(x.shape, y.shape)
        l = loss(net(x), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # normal fitting
    # trainl.append(loss(net(_xs[:100, :]), _y[:100]).item())
    # testl.append(loss(net(_xs[100:200, :]), _y[100:200]).item())

    # under fitting
    # trainl.append(loss(net(_x[:100, :]), _y[:100]).item())
    # testl.append(loss(net(_x[100:200, :]), _y[100:200]).item())

    # over fitting
    trainl.append(loss(net(_xs[:2, :]), _y[:2]).item())
    testl.append(loss(net(_xs[100:200, :]), _y[100:200]).item())

px = [i for i in range(iters)]
plt.figure('scatter points')
ax = plt.gca()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.scatter(px, trainl, label= 'train loss', linestyle=':')
ax.scatter(px, testl, label='test loss', linestyle=':')
ax.legend()

ax.plot(px, trainl, label= 'train loss', linestyle=':')
ax.plot(px, testl, label='test loss', linestyle=':')
ax.legend()

plt.show()