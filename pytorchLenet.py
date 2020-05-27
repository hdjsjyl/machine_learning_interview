import torch as tc
import time
import sys
import torchvision as tv
import torchvision.transforms as trans
import torch.utils.data as Data

# load dataset
train_data = tv.datasets.FashionMNIST(root='./data', train=True, download=False, transform=trans.ToTensor())
test_data  = tv.datasets.FashionMNIST(root='./data', train=False, download=False, transform=trans.ToTensor())

# make the dataset iterable
trainIter = Data.DataLoader(train_data, batch_size=256, shuffle=True)
testIter  = Data.DataLoader(test_data, batch_size=256, shuffle=False)

# create the model class
class Lenet(tc.nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv = tc.nn.Sequential()
        # self.conv.add_module('conv1', tc.nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(3, 3), stride=(1, 1)))
        self.conv.add_module('conv1', tc.nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)))
        self.conv.add_module('sig1', tc.nn.Sigmoid())
        self.conv.add_module('max1', tc.nn.MaxPool2d(2, 2))
        self.conv.add_module('conv2', tc.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)))
        self.conv.add_module('sig2', tc.nn.Sigmoid())
        self.conv.add_module('max2', tc.nn.MaxPool2d(2, 2))

        self.fc = tc.nn.Sequential()
        self.fc.add_module('l1', tc.nn.Linear(16*4*4, 120))
        self.fc.add_module('sig1', tc.nn.Sigmoid())
        self.fc.add_module('l2', tc.nn.Linear(120, 84))
        self.fc.add_module('sig2', tc.nn.Sigmoid())
        self.fc.add_module('l3', tc.nn.Linear(84, 10))

    def forward(self, out):
        for m in self.conv:
            # print(out.shape)
            out = m(out)
        out = out.view(out.shape[0], -1)
        for n in self.fc:
            out = n(out)
        return out


def evaluateAcc(iter, net, device):
    acc = 0
    num = 0
    net.eval()
    for x, y in iter:
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        a = (out.argmax(dim=1) == y).sum().item()
        acc += a
        num += len(x)
    return acc/num

# instantiate the model class
net = Lenet()

# instantiate the loss function
loss = tc.nn.CrossEntropyLoss()

# instantiate optimizer
# optimizer = tc.optim.SGD(net.parameters(), lr=0.001)
optimizer = tc.optim.Adam(net.parameters(), lr=0.001)

num_epochs = 5
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

net = net.to(device)
for i in range(num_epochs):
    train_l = 0
    train_acc = 0
    num = 0
    batch = 0
    for x, y in trainIter:
        x = x.to(device)
        y = y.to(device)

        y_hat = net(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_l += l.item()
        acc = (y_hat.argmax(dim=1) == y).sum().item()
        train_acc += acc
        num += len(x)
        batch += 1
    testAcc = evaluateAcc(testIter, net, device)
    print('epoch {}'.format(i), 'loss: {}'.format(train_l/batch), 'trainAcc: {}'.format(train_acc/num), 'testAcc: {}'.format(testAcc))