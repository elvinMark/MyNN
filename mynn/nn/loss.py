import numpy as np

"""
Loss Classes
"""

# Mean Square Loss
class MSELoss:
    def __init__(self):
        self.grad = None

    def calculateLoss(self,o,t):
        self.grad = o - t
        return (self.grad*self.grad).sum()*0.5

    def getGrad(self):
        return self.grad

# Cross Entropy Loss (Using basically the idea of KL-distance [distance bewtween probability distribution])
class CrossEntropyLoss:
    def __init__(self):
        self.grad = None

    def calculateLoss(self,o,t):
        self.grad = -t/o
        return (-t*np.log(o)).sum()

    def getGrad(self):
        return self.grad
