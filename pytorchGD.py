import torch as tc
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.1)
y = x*x

def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta*2*x
        results.append(x)
    return results

res  = gd(0.05)
resy = [i**2 for i in res]

plt.figure('function')
ax = plt.gca()
# ax.scatter(x, y)
ax.plot(x, y)
ax.scatter(res, resy)
plt.show()