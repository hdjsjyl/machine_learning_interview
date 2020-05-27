import numpy as np

class conv:
    def __init__(self, input, output, kernel=3, stride=1, pad=1):
        self.inputC  = input
        self.outputC = output
        self.kernel  = kernel
        self.stride  = stride
        self.pad     = pad
        ## divide (kernel*kernel) to reduce the variance of initial values
        self.filters = np.random.randn(output, input, self.kernel, self.kernel)/(self.kernel*self.kernel)
        self.data = None

    def forward(self, x):
        ## x : N, H, W, channel is 1 for minist
        self.data = x
        N, C, H, W = x.shape
        assert C == self.inputC
        HO = (H+2*self.pad-self.kernel)//self.stride+1
        WO = (W+2*self.pad-self.kernel)//self.stride+1

        pad_data = np.pad(x, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant', constant_values=0)

        output_data = np.zeros((N, self.outputC, HO, WO))
        for i in range(self.outputC):
            for j in range(HO):
                for k in range(WO):
                    x_window = pad_data[:, :, j*self.stride:j*self.stride+self.kernel, k*self.stride:k*self.stride+self.kernel]
                    output_data[:, i, j, k] = np.sum(x_window * self.filters[i, :, :, :], axis=(1, 2, 3))

        return output_data

    def backward(self, dout, lr):
        pad_data = np.pad(self.data, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant', constant_values=0)
        dw = np.zeros_like(self.filters)
        dx = np.zeros_like(pad_data)
        N, C, H, W = self.data.shape
        N1, CO, HD, WD = dout.shape

        HO = (H+2*self.pad-self.kernel)//self.stride+1
        WO = (W+2*self.pad-self.kernel)//self.stride+1

        for n in range(N):
            for i in range(HO):
                for j in range(WO):
                    x_window = pad_data[n, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel]
                    for c in range(CO):
                        dw[c] += x_window*dout[n, c, i, j]

                        dx[n, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel] += self.filters[c, :, :, :]*dout[n, c, i, j]

        self.filters -= lr*dw
        # print(self.filters[0, 0, 0, :], lr, dw[0, 0, 0, :])
        dx = dx[:, :, self.pad:self.pad+H, self.pad:self.pad+W]
        return dx

class maxPooling:
    def __init__(self, kernel=2, stride=2):
        self.kernel = kernel
        self.stride = stride
        self.data = None

    def forward(self, x):
        self.data = x
        N, C, H, W = x.shape
        HO = (H-self.kernel)//self.stride + 1
        WO = (W-self.kernel)//self.stride + 1

        output_data = np.zeros((N, C, HO, WO))

        for i in range(HO):
            for j in range(WO):
                output_data[:, :, i, j] = np.max(self.data[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel], axis=(2, 3))

        return output_data

    def backward(self, dout):
        dy = np.zeros_like(self.data)

        N, C, HO, WO = dout.shape
        for i in range(HO):
            for j in range(WO):
                x_window = self.data[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel]
                maxs     = np.max(x_window, axis=(2, 3))
                masks    = (x_window == (maxs)[:, :, None, None])
                dy[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel] = masks * (dout[:, :, i, j])[:, :, None, None]

        return dy

class averagePooling:
    def __init__(self, kernel=2, stride=1, glo=False):
        self.glo    = glo
        self.kernel = kernel
        self.stride = stride
        self.data   = None

    def forward(self, x):
        N, C, H, W = x.shape
        self.data = x
        if self.glo:
            self.kernel = int(np.sqrt(H*W))
            self.stride = 1

        HO = (H-self.kernel)//self.stride+1
        WO = (W-self.kernel)//self.stride+1
        HO = HO
        WO = WO

        output = np.zeros((N, C, HO, WO))
        for i in range(HO):
            for j in range(WO):
                x_window = self.data[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel]
                output[:, :, i, j] = np.max(x_window, axis=(2, 3))

        return output

    def backward(self, dout):
        dx = np.zeros_like(self.data)

        N, C, H, W = dout.shape
        for i in range(H):
            for j in range(W):
                dx_window = dx[:, :, i * self.stride:i * self.stride + self.kernel, j * self.stride:j * self.stride + self.kernel]
                tmp = np.ones_like(dx_window)
                dx[:, :, i * self.stride:i * self.stride + self.kernel, j * self.stride:j * self.stride + self.kernel] = (dout[:, :, i, j])[:, :, None, None] / (self.kernel * self.kernel)
                # dx[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel] = dout[:, :, i, j]/(self.kernel)

        return dx

class softmaxCrossEntropy:
    def __init__(self):
        self.ps = None
        self.label = None
        self.length = None

    def forward(self, x, y):
        ## x : (num_samples, num_classes)
        ## y : (num_samples, 1)
        N, C, H, W = x.shape
        x = x.reshape((N, C))
        diff = x - np.max(x, axis=0)
        exps = np.exp(diff)
        ps = exps/np.sum(exps, axis=0)
        self.ps = ps
        self.label = y
        length = N
        self.length = length
        tmps = ps[range(length), y]
        loss = -np.log(tmps+1e-4)
        loss = np.sum(loss) / length
        args = np.argmax(ps, axis=1)
        acc = (args == y)
        return self.ps, loss, np.sum(acc)

    def backward(self):
        dx = self.ps
        dx[range(self.length), self.label] -= 1
        dx = (dx)[:, :, None, None]
        return dx/self.length


def dataLoader(x, y, batchsize):
    indices = [i for i in range(len(x))]
    np.random.shuffle(indices)

    for i in range(0, len(x), batchsize):
        x1  = x[indices[i:min(len(x), i+batchsize)]]
        y1  = y[indices[i:min(len(x), i+batchsize)]]
        yield x1, y1


def forward(images, labels, conv1, maxpool, conv2, averagepool, softmaxLoss, lr):
    ## image normalization
    imgs = (images/255-0.5)
    out = conv1.forward(imgs)
    # print(out.shape)
    out = maxpool.forward(out)
    # print(out.shape)
    out = conv2.forward(out)
    # print(out.shape)
    out = averagepool.forward(out)
    ps, loss, acc = softmaxLoss.forward(out, labels)

    gradient = softmaxLoss.backward()
    gradient = averagepool.backward(gradient)
    gradient = conv2.backward(gradient, lr)
    gradient = maxpool.backward(gradient)
    gradient = conv1.backward(gradient, lr)
    return ps, loss, acc

def forward2(images, labels, conv1, maxpool, conv2, averagepool, softmaxLoss, lr):
    ## image normalization
    imgs = (images/255-0.5)
    out = conv1.forward(imgs)
    # print(out.shape)
    out = maxpool.forward(out)
    # print(out.shape)
    out = conv2.forward(out)
    # print(out.shape)
    out = averagepool.forward(out)
    ps, loss, acc = softmaxLoss.forward(out, labels)
    return ps, loss, acc


if __name__ == '__main__':
    from mnist import MNIST

    mndata = MNIST('/Users/shilei/python-mnist/bin/data')
    trains, train_labels = mndata.load_training()
    tests, test_labels = mndata.load_testing()

    trainImgs = np.array(trains[:1000])
    trainLables = np.array(train_labels[:1000])
    testImgs = np.array(tests[:1000])
    testLabels = (np.array(test_labels[:1000]))

    trainImgs = np.reshape(trainImgs, (1000, 1, 28, -1))

    ## parameters setting
    lr        = 0.001
    batchsize = 2
    iters     = 3

    ## define layers
    conv1 = conv(1, 8)
    maxpool = maxPooling()
    conv2 = conv(8, 10)
    averagepool = averagePooling(glo=True)
    softmaxLoss = softmaxCrossEntropy()

    ## training
    for i in range(iters):
        tmp  = 0
        numc = 0
        num  = 0
        for x, y in dataLoader(trainImgs, trainLables, batchsize):
            out, loss, acc = forward(x, y, conv1, maxpool, conv2, averagepool, softmaxLoss, lr)
            tmp += loss
            numc += acc
            num += len(x)
            print(loss, numc, num, numc/num)
        print("iters {}".format(i), tmp, numc)


    ## testing # Test the CNN
    print('\n--- Testing the CNN ---')
    tmp = 0
    numc = 0
    for x, y in dataLoader(testImgs, testLabels, 1):
        out, loss, acc = forward2(x, y, conv1, maxpool, conv2, averagepool, softmaxLoss, lr)
        tmp += loss
        numc += acc
    print("loss: {}".format(tmp/1000))
    print("accuracy: {}".format(numc/1000))
