import numpy as np

def max_pooling(data, kernel, stride):
    N, C, H, W = data.shape
    HK, WK = kernel.shape

    HO = (H-HK)//stride+1
    WO = (W-WK)//stride+1

    out = np.zeros((N, C, HO, WO))

    for i in range(HO):
        for j in range(WO):
            x_window = data[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK]
            out[:, :, i, j] = np.max(x_window, axis=(2, 3))

    return out

def max_pooling_back(dout, data, kernel, stride):
    N, C, H, W = data.shape
    HK, WK = kernel.shape
    HO = (H-HK)//stride+1
    WO = (W-WK)//stride+1

    dx = np.zeros((N, C, H, W))

    for i in range(HO):
        for j in range(WO):
            x_window = data[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK]
            maxs = np.max(x_window, axis=(2, 3))
            maks = (x_window == (maxs)[:, :, None, None])
            dx[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK] += maks*(dout[:, :, i, j])[:, :, None, None]

    return dx

def average_pooling(data, kernel, stride):
    N, C, H, W = data.shape
    HK, WK = kernel.shape

    HO = (H-HK)//stride+1
    WO = (W-WK)//stride+1

    out = np.zeros((N, C, HO, WO))

    for i in range(HO):
        for j in range(WO):
            out[:, :, i, j] = np.mean(data[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK], axis=(2, 3))

    return out

def average_pooling_back(dout, data, kernel, stride):
    N, C, H, W = data.shape
    HK, WK = kernel.shape

    HO = (H-HK)//stride+1
    WO = (W-WK)//stride+1

    dx = np.zeros((N, C, H, W))

    for i in range(HO):
        for j in range(WO):
            dx_window = dx[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK]
            tmp = np.ones_like(dx_window)
            dx[:, :, i*stride:i*stride+HK, j*stride:j*stride+WK]= dout[:, :, i, j]/(HK*WK) * tmp

    return dx

if __name__ == '__main__':
    data = np.arange(0, 36).reshape(2, 2, 3, 3)
    # print(data)
    # temp  = (data[:, :, 0, 1])[:, :, None, None]
    # print(temp.shape, temp)
    tmp = np.max(data, axis=(2, 3))
    print(tmp.shape)
    print(data == (tmp)[:, :, None, None])