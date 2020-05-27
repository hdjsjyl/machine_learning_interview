# step1 create dataset
# step2 make it iterable
# step3 create model class and instantiate model class
# step4 instantiate loss function
# step5 instantiate optimizer
# step6 train model
# step7 test model
import torch
import numpy as np

num_inputs   = 2
num_examples = 1000
true_w       = [2, -3.4]
true_b       = 4.2
# features     = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
features     = torch.randn((num_examples, num_inputs), dtype=torch.float32)
labels       = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# labels      += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

import torch.utils.data as Data

dataset    = Data.TensorDataset(features, labels)
dataLoader = Data.DataLoader(dataset, batch_size=10, shuffle=True)

for X, y in dataLoader:
    break

class LinearNet(torch.nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

net = LinearNet(num_inputs)

from torch.nn import init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant(net.linear.bias, val=0)

loss = torch.nn.MSELoss()

import torch.optim as optim
optimizer = optim.SGD(net.linear.parameters(), lr=0.03)

for num in range(3):
    for x, y in dataLoader:
        output = net(x)
        l = loss(output, y)
        optimizer.zero_grad() # 梯度清零
        l.backward() # 计算梯度
        optimizer.step() # 更新参数

    print('iteration {}'.format(num), ' loss: ', l.item())

dense = net
print(true_w, dense.linear.weight)
print(true_b, dense.linear.bias)