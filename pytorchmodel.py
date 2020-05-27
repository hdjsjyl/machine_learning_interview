import torch as tc

class model(tc.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.x = tc.nn.Sequential()
        self.x.add_module('l1', tc.nn.Linear(3, 1))
        self.x.add_module('l2', tc.nn.Linear(1, 1))

    def forward(self, x):
        for m in self.x:
            x = m(x)
        return x

y = model()
for name, pa in y.named_parameters():
    print(name, type(name), pa.shape)

class model2(tc.nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.Linear1 = tc.nn.Linear(3, 1)
        self.Linear2 = tc.nn.Linear(1, 1)

    def forward(self, input):
        out = self.Linear1(input)
        out = self.Linear2(out)
        return out

z = model2()
for name, param in z.named_parameters():
    print(name, param)

class model3(tc.nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.sq1 = tc.nn.Sequential()
        self.sq1.add_module('l1', tc.nn.Linear(3, 1))
        self.sq1.add_module('l2', tc.nn.Linear(1, 1))

        self.sq2 = tc.nn.Sequential()
        self.sq2.add_module('l1', tc.nn.Linear(1, 1))
        self.sq2.add_module('l2', tc.nn.Linear(1, 1))

    def forward(self, input):
        out1 = self.sq1(input)
        out2 = self.sq2(out1)
        return out2

p = model3()
for name, param in p.named_parameters():
    print(name, param.shape)

x = tc.zeros((1, 1, 9, 9))
net = tc.nn.MaxPool2d(2, 2)
y = net(x)
print(x.shape, y.shape)


class test(tc.nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.x = 1
        self.y = 1

    def forward(self, input):
        print('------------', self.x)
        print('------------', self.y)
        return input


class test2(tc.nn.Module):
    def __init__(self):
        super(test2, self).__init__()
        self.x = 1
        self.y = 1

    def forward(self, input):
        print(self.x)
        print(self.y)
        return input


tmpModel = test2()
for name, param in tmpModel.named_parameters():
    print(name, param.shape)
x = tc.zeros((1, 3))
tmpModel(x)
# tmpModel(1)