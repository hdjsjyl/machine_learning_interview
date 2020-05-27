import numpy as np

class conv:
    def __init__(self, input_channel, output_channel, size=3, stride=1, pad=0, bias=None):
        self.input  = input_channel
        self.out    = output_channel
        self.size   = size
        self.stride = stride
        self.pad    = pad
        ## divide 9 tryies to reduce the variance of initial values
        self.filter = np.random.randn(self.out, self.input, self.size, self.size)/(self.size**2)
        if bias:
            self.bias = np.random.rand(self.out)
        self.data = None

    def forward(self, input):
        N, C1, H, W = input.shape
        assert C1 == self.input, 'C1 should be equal to input channel'

        HO = (H+2*self.pad-self.size)//self.stride+1
        WO = (W+2*self.pad-self.size)//self.stride+1

        pad_input = np.pad(input, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant', constant_values=0)

        output = np.zeros((N, self.out, HO, WO))
        self.data = input

        for i in range(self.out):
            for j in range(HO):
                for k in range(WO):
                    tmp = pad_input[:, :, j*self.stride:j*self.stride+self.size, k*self.stride:k*self.stride+self.size] * self.filter[i, :, :, :]
                    output[:, i, j, k] = np.sum(tmp, axis=(1, 2, 3))
            if self.bias:
                output[:, i, :, :] += self.bias[i]

        return output

    def backward(self, dout, lr):
        N, CO, H, W = dout.shape
        assert CO == self.out

        dw = np.zeros_like(self.filter)
        db = np.sum(dout, axis=(0, 2, 3))

        pad_data = np.pad(self.data, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode='constant', constant_values=0)

        for n in range(N):
            for i in range(H):
                for j in range(W):
                    x_window = pad_data[n, :, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size]
                    for c in range(CO):
                        dw[c] = x_window*dout[n, c, i, j]


        self.filter -= lr*dw

        return None

class maxPool:
    def __init__(self, kernel=2, stride=2):
        self.data = None
        self.stride = stride
        self.kernel = kernel


    def forward(self, input):
        N, C, H, W = input.shape
        self.data = input

        HO = (H-self.kernel)//self.stride+1
        WO = (W-self.kernel)//self.stride+1

        output = np.zeros((N, C, HO, WO))

        for i in range(HO):
            for j in range(WO):
                x_window = self.data[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel]
                output[:, :, i, j] = np.max(x_window, axis=(2, 3))

        return output

    def backword(self, dout):
        dinput = np.zeros_like(self.data)
        N, C, H, W = dout.shape

        for i in range(H):
            for j in range(W):
                x_window = self.data[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel]
                maxs = np.max(x_window, axis=(2, 3))
                masks = (x_window == (maxs)[:, :, None, None])
                dinput[:, :, i*self.stride:i*self.stride+self.kernel, j*self.stride:j*self.stride+self.kernel] += masks*(dout[:, :, i, j])[:, :, None, None]

        return dinput


## sigmoid: 1/1+exp(-x)
class sigmoid:
    def __init__(self):
        self.data = None

    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self, out):
        return out*(1-out)


## softmax:      exp(x)/sum(exp(x))
## crossEntropy: -sum(y*log(y_hat))
class softmaxCrossEntropyLoss:
    def __init__(self):
        self.data = None
        self.ps   = None
        self.N    = None

    def forward(self, x, y):
        ## x: (num_samples. num_classes)
        ## y: (num_sampels, 1)
        self.data = x
        N = len(x)
        shiftx = x - np.max(x, axis=0)
        exps = np.exp(shiftx)
        ps = exps/(np.sum(exps, axis=0))
        ## ps is the output of softmax function
        self.ps = ps
        ## CrossEntropyLoss
        loss = -np.log(self.ps[range(N), y])
        loss = np.sum(loss)/self.N
        return loss


    def backward(self, y):
        ## gradients: y_hat-y
        dw = self.ps[range(self.N), y]-1
        dw = dw/self.N
        return dw


# if __name__ == '__main__':























