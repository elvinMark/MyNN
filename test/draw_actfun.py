import sys
sys.path.append("../")
from myNN.util.actfun import *
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = sigmoid(x)

plt.figure(1)
plt.plot(x,y)
plt.grid()
plt.show()


x = np.linspace(-10,10,100)
y = tanh(x)

plt.figure(2)
plt.plot(x,y)
plt.grid()
plt.show()


x = np.linspace(-10,10,100)
y =relu(x)

plt.figure(3)
plt.plot(x,y)
plt.grid()
plt.show()
