import torch as tc
import numpy as np
import torchvision as tv
import torch.utils.data as Data
import torchvision.transforms as trans
import sys

# step1 create dataset
train_data = tv.datasets.FashionMNIST(root='./data', train=True, download=False, transform=trans.ToTensor())
test_data  = tv.datasets.FashionMNIST(root='./data', train=False, download=False, transform=trans.ToTensor())

# step2 make it iterable
trainIter = Data.DataLoader(train_data, batch_size=256, shuffle=True)
testIter  = Data.DataLoader(test_data, batch_size=256, shuffle=True)

# step3 create model class
class MLP(tc.nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        self.model = tc.nn.Sequential()
        self.model.add_module('l1', tc.nn.Linear(input_features, 256))
        self.model.add_module('a1', tc.nn.ReLU())
        self.model.add_module('l2', tc.nn.Linear(256, 10))

    def forward(self, x):
        out = self.model(x)
        return out

# step4 instantiate model class
net = MLP(784)

# step5 instantiate loss function
Loss = tc.nn.CrossEntropyLoss()

# step6 instantiate optimizer
# def sgd(parameters, lr):
#     for para in parameters:
#         para.data = para.data - lr*para.grad.data

optimizer = tc.optim.SGD(net.model.parameters(), lr=0.5)

for para in net.model.parameters():
    tc.nn.init.normal_(para, mean=0, std=0.01)

# step7 train mode
iters = 5
# lr    = 0.1

for i in range(iters):
    loss = 0
    num  = 0
    acc  = 0
    for x, y in trainIter:
        x = x.view(-1, 784)
        out = net(x)
        # print(out.shape, y.shape)
        l = Loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss += l.item()
        num += len(x)
        acc += (out.argmax(dim=1) == y).sum().item()
    print('iter {}'.format(i), ' loss: ', loss/num, ' acc: ', acc/num)