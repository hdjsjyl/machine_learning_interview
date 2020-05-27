import numpy as np

def conv(data, kernel, bias, pad, stride):
    N, C1, H, W = data.shape
    CO, C2, HK, WK = kernel.shape

    assert C1 == C2, 'the channel of the input data should be equal to the channel of the kernel'

    HO = (H + 2*pad - HK)//stride + 1
    WO = (W + 2*pad - WK)//stride + 1

    pad_data = np.zeros((N, C1, H + 2 * pad, W + 2 * pad))
    pad_data[:, :, pad:pad+H, pad:pad+W] = data
    output_data = np.zeros((N, CO, HO, WO))

    for i in range(CO):
        for j in range(HO):
            for k in range(WO):
                temp = pad_data[:, :, j*stride:j*stride + HK, k*stride:k*stride + WK] * kernel[i, :, :, :]
                output_data[:, i, j, k] = np.sum(temp, axis=(1, 2, 3))

        output_data[:, i, :, :] += bias[i]

    return output_data


def conv_back(dy, data, kernel, bias, stride, pad):
    ## pad(x) * w = z, dz
    ## dx = dz*rotation(w)

    N, C1, H, W = data.shape
    CO, C2, HK, WK = kernel.shape

    HO = (H+2*pad-HK)//stride + 1
    WO = (W+2*pad-WK)//stride + 1

    dx = np.zeros_like(data)
    dw = np.zeros_like(kernel)

    db = np.sum(dy, axis=(0, 2, 3))

    pad_data = np.pad(data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)
    pad_dx   = np.pad(dx, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant', constant_values=0)

    for n in range(N):
        for i in range(HO):
            for j in range(WO):
                x_window = pad_data[n, :, i*stride:i*stride+HK, j*stride:j*stride+WK]
                for c in range(CO):
                    dw[c] += x_window*dy[n, c, i, j]

                    pad_dx[n, :, j*stride:j*stride+HK, i*stride:i*stride+WK] += dy[n, c, i, j]*kernel[c, :, :, :]

        dx = pad_dx[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db


if __name__ == '__main__':
    data = np.ones((2, 3, 4, 4))
    kernel = np.ones((2, 3, 3, 3))
    bias = np.array([1, 2])
    stride = 1
    pad = 0
    out = conv(data, kernel, bias, pad, stride)
    # print(out.shape, out)
    dout = np.ones((data.shape[0], kernel.shape[0], 2, 2))
    out_back = conv_back(dout, data, kernel, bias, stride, pad)
    print(out_back)