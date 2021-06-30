import numpy as np
import myNN.nn as nn
from myNN.nn.model import *

class MyCNN:
    def __init__(self):
        self.conv1 = SequentialModel(
            nn.Conv2DLayer(1,16),
            nn.BatchNormalization2DLayer(16),
            nn.ReLULayer(),
            nn.MaxPool2DLayer()
        )

        self.conv2 = SequentialModel(
            nn.Conv2DLayer(16,32,padding=1,stride=3),
            nn.BatchNormalization2DLayer(32),
            nn.ReLULayer()
        )

        self.fc1 = SequentialModel(
            nn.LinearLayer(800,100),
            nn.ReLULayer()
        )

        self.fc2 = SequentialModel(
            nn.LinearLayer(100,10),
            nn.SoftmaxLayer()
        )
        
    def forward(self,x):
        o = self.conv1(x)
        o = self.conv2(o)
        o = o.reshape((-1,800))
        o = self.fc1(o)
        o = self.fc2(o)
        return o

    def backward(self,x):
        o = self.fc2.backward(x)
        o = self.fc1.backward(o)
        o = o.reshape((-1,32,5,5))
        o = self.conv2.backward(o)
        o = self.conv1.backward(o)

    def update(self,optim):
        self.conv1.update(optim)
        self.conv2.update(optim)
        self.fc1.update(optim)
        self.fc2.update(optim)

    def __call__(self,x):
        return self.forward(x)

    def eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.fc1.eval()
        self.fc2.eval()
