import torch as tc

x1 = tc.empty(5, 3)
x2 = tc.ones(5, 3)
x3 = tc.zeros(5, 3)
x4 = tc.rand(5, 3)
x5 = tc.zeros(5, 3, dtype=tc.long)
x6 = tc.tensor([[5.5, 3.0]])
print(x6.shape)