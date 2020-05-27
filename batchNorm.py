import numpy as np


def bn(data, gamma, beta, eps):
    N, C, H, W = data.shape
    # step0
    batch_mean = (1/N)*np.sum(data, axis=(0, 2, 3))
    # step1
    subst_mean = data - batch_mean
    # step2
    subst_mean_square = subst_mean**2
    # step3
    var = (1/N)*np.sum(subst_mean_square, axis=(0, 2, 3))
    # step4
    sqrtvar = np.sqrt(var+eps)
    # step5
    ivar    = 1/sqrtvar
    # step6
    normalized_data = subst_mean * ivar
    # step7
    return gamma*normalized_data+beta


def bn_back(dout, normalized_data, gamma, substract_mean, ivar, sqrtvar, eps):
    N, C, H, W = dout.shape
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout*normalized_data, axis=(0, 2, 3))

    dnormalized_data = gamma

    divar = np.sum(dnormalized_data*substract_mean, axis=(0, 2, 3))
    dsubstract_mean1 = dnormalized_data*ivar

    dsqrtvar = (-1)/(sqrtvar**2)/sqrtvar * divar
    dvar = 0.5*1/(sqrtvar+eps) * dsqrtvar

    dsubstract_mean_squre = (1/N)*np.ones((N, C, H, W))*dvar

    dsubstract_mean2 = 2*substract_mean*dsubstract_mean_squre

    dsubstract_mean = dsubstract_mean1 + dsubstract_mean2

    dbatch_mean = -1*(np.sum(dsubstract_mean, axis=(0, 2, 3)))

    data1 = dsubstract_mean

    data2 = (1/N)*np.ones((N, C, H, W))*dbatch_mean

    return data1 + data2, dgamma, dbeta


if __name__ == '__main__':
    data = np.array([[1,2], [3,4], [5,6], [7,8]])
    print(data.shape)
    print(np.sum(data, axis=0))
    print(np.sum(data, axis=1))
