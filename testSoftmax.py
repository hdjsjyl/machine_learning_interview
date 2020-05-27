import numpy as np

def softMax(a):
    a -= np.max(a)
    exp = np.exp(a)
    return exp/np.sum(exp)

if __name__ == '__main__':
    x = np.array(([1, -1]))
    print(softMax(x))
    x1 = np.array(([1, 1]))
    print(softMax(x1))
    x2 = np.array(([1, 0]))
    print(softMax(x2))