import numpy as np
from tqdm import tqdm
from myNN.nn.layer import *
from myNN.nn.loss import *
from myNN.nn.optim import *

# Sequential Model
class SequentialModel:
    def __init__(self):
        self.layers = []

    def addLayer(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        o = x
        for l in self.layers:
            o = l.forward(o)
        return o

    def backward(self,x):
        o = x
        for l in reversed(self.layers):
            o = l.backward(o)

    def update(self,optim):
        for l in self.layers:
            l.update(optim)
            
    def train(self,dataset,epochs=100,optim=SGDOptimizer(),loss=MSELoss()):
        log_loss = []
        for epoch in tqdm(range(epochs)):
            l = 0
            for x,y in dataset:
                o = self.forward(x)
                l += loss.calculateLoss(o,y)
                e = loss.getGrad()
                self.backward(e)
                self.update(optim)

            log_loss.append(l)
        
        return log_loss
