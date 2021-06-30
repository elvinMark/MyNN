import numpy as np

from tqdm import tqdm
from myNN.nn.layer import *
from myNN.nn.loss import *
from myNN.nn.optim import *

# Sequential Model
class SequentialModel:
    def __init__(self,*kargs):
        self.layers = []
        for l in kargs:
            self.layers.append(l)
    
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
        return o
    
    def update(self,optim):
        for l in self.layers:
            l.update(optim)

    def __call__(self,x):
        return self.forward(x)

    def eval(self):
        for l in self.layers:
            l.clear_all_extra()
