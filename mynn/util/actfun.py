import numpy as np

"""
Activation Functions
"""
def sigmoid(x,diff=False):
    if diff:
        return x*(1-x)
    return 1/(1 + np.exp(-x))

def tanh(x,diff=False):
    if diff:
        return (1 - x**2)/2
    return (1 - np.exp(-x))/(1 + np.exp(-x))

def relu(x,diff=False):
    if diff:
        tmp = np.ones(x.shape)*0.01
        tmp[x>0] = 1
        return tmp
    tmp = x*0.01
    tmp[x>0] = x[x>0]
    return tmp
