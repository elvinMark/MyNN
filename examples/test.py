import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
from myNN.nn.layer import *
from myNN.nn.optim import SGDOptimizer
from myNN.util.actfun import relu

# y = nn.Conv2d(2,3,3,padding=1,stride=2)
# x = torch.rand((2,2,5,5))

# t = y(x).detach().numpy()

# c = Conv2DLayer(2,3,3,bias=True,padding=1,stride=2)
# c.weight = y.weight.detach().numpy()
# c.bias = y.bias.detach().numpy()

# o = c.forward(x.detach().numpy())

# print(t)
# print(o)
# print(t-o)

# f = c.backward(o)
# c.update(SGDOptimizer())
# print(f.shape)
# x = np.random.random((3,4,2,5)) - 0.5
# print(x)
# print(relu(x,diff=True))

x = np.random.random((2,1,11,11))
t = Conv2DLayer(1,3,kernel_size=3,stride=2)
y = t(x)
print(y.shape)
print(y)

err = 0.01*np.random.random((2,3,5,5))
o = t.backward(err)
print(o.shape)
print(o)
