import torch as tc
import torchvision as tv
import torch.utils.data as Data
import sys
import time

class FlattenLayer(tc.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
        self.shape = None

    def forward(self, input):
        out = input.view(input.shape[0], -1)
        self.shape = out.shape
        return out

class GlobalAvgPool2d(tc.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        self.layer = tc.nn.AdaptiveAvgPool2d((1, 1))
        self.shape = None

    def forward(self, input):
        out = self.layer(input)
        self.shape = out.shape
        return out

class Residul(tc.nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residul, self).__init__()
        self.conv1 = tc.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = tc.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = tc.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = tc.nn.BatchNorm2d(out_channels)
        self.bn2 = tc.nn.BatchNorm2d(out_channels)
        self.relu1 = tc.nn.ReLU()
        self.relu2 = tc.nn.ReLU()

    def forward(self, input):
        out = self.relu1(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            input = self.conv3(input)
        return self.relu2(out + input)

class Residul_block(tc.nn.Module):
    def __init__(self, in_channels, out_channels, num, first_block=False):
        super(Residul_block, self).__init__()
        if first_block:
            assert in_channels == out_channels

        self.block = tc.nn.Sequential()
        for i in range(num):
            if i == 0 and not first_block:
                block = Residul(in_channels, out_channels, True, stride=2)
            else:
                block = Residul(out_channels, out_channels)

            self.block.add_module('block{}'.format(i), block)

    def forward(self, out):
        for x in self.block:
            out = x(out)
        return out

class ResNet18(tc.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.res18 = tc.nn.Sequential()
        self.res18.add_module('conv1', tc.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2))
        self.res18.add_module('bn1', tc.nn.BatchNorm2d(64))
        self.res18.add_module('relu1', tc.nn.ReLU())
        self.res18.add_module('pool1', tc.nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
        self.res18.add_module('block1', Residul_block(64, 64, 2, True))
        self.res18.add_module('block2', Residul_block(64, 128, 2))
        self.res18.add_module('block3', Residul_block(128, 256, 2))
        self.res18.add_module('block4', Residul_block(256, 512, 2))
        self.res18.add_module('global2d', GlobalAvgPool2d())
        self.res18.add_module('flatten', FlattenLayer())
        self.res18.add_module('fc', tc.nn.Linear(512, 10))

    def forward(self, input):
        for x in self.res18:
            # for y in x:
            #     input = y(input)
            input = x(input)
        return input

## instantiate model class
net = ResNet18()

# x = tc.randn((1, 3, 256, 256))
# for name, layer in net.named_children():
#     for name2, layer2 in layer.named_children():
#         x = layer2(x)
#         print(name2, x.shape)

## load dataset
transforms = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tv.transforms.RandomResizedCrop(200, (0.1, 1), ratio=(0.5, 2)),
    tv.transforms.ToTensor()
])

def evaluatAcc(iter, net, device):
    acc = 0
    num = 0
    for x, y in iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        a = (y_hat.argmax(dim=1) == y).sum().item()
        acc += a
        num += len(x)
    return acc/num

## make dataset iterable
train_data = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
test_data  = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=tv.transforms.ToTensor())

trainIter = Data.DataLoader(train_data, batch_size=256, shuffle=True)
testIter  = Data.DataLoader(test_data, batch_size=256, shuffle=False)

## create loss function
loss = tc.nn.CrossEntropyLoss()

## instantiate optimizer
optimizer = tc.optim.Adam(net.parameters(), lr=0.001)

## train model
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
print("training on ", device)
epoch = 10
net = net.to(device)
for i in range(epoch):
    train_l = 0
    train_acc = 0
    num = 0
    batch = 0
    for x, y in trainIter:
        start = time.time()
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        train_l += l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = (y_hat.argmax(dim=1) == y).sum().item()
        train_acc += acc
        num += len(x)
        batch += 1
        # print('batch {}'.format(batch+1), 'time: {}'.format(time.time()-start))
    testAcc = evaluatAcc(testIter, net, device)
    print('epoch {}'.format(i), 'loss: {}'.format(train_l/batch), 'trainAcc: {}'.format(train_acc/num), 'testAcc: {}'.format(testAcc))