# step1 load dataset
# step2 make dataset iterable
# step3 create model class
# step4 instantiate model class
# step5 instantiate loss function
# step6 instantiate optimizer
# step7 train model
# step8 test model

import torch as tc
import torchvision as tv
import torchvision.transforms as trans
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
import sys

mnist_train = tv.datasets.FashionMNIST(root='./data', train=True, download=False, transform=trans.ToTensor())
mnist_test  = tv.datasets.FashionMNIST(root='./data', train=False, download=False, transform=trans.ToTensor())

def get_fashionMnist_labels(labels):
    text_labels = {}
    text_labels[0] = 't-shirt'
    text_labels[1] = 'trouser'
    text_labels[2] = 'pullover'
    text_labels[3] = 'dress'
    text_labels[4] = 'coat'
    text_labels[5] = 'sandal'
    text_labels[6] = 'shirt'
    text_labels[7] = 'sneaker'
    text_labels[8] = 'bag'
    text_labels[9] = 'ankle boot'
    res = [text_labels[i] for i in labels]
    return res

trainIter = Data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=4)
testIter  = Data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=4)

start = time.time()
for x, y in trainIter:
    continue
print(time.time() - start)
