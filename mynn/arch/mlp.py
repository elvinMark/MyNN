import numpy as np
from myNN.nn.layer import *
from myNN.nn.loss import *
from myNN.nn.optim import *
from myNN.nn.model import SequentialModel

"""
MultiLayer Perceptron (MLP)
"""

# Simple MultiLayer Perceptron Class
# This MLPRegressor has been designed for regression problems (so it is mainly using MSELoss)
class MLPRegressor(SequentialModel):
    def __init__(self,layer_info=[1,1],act_fun="sigmoid"):
        super().__init__()
        prev = layer_info[0]
        for curr in layer_info[1:]:
            self.addLayer(LinearLayer(prev,curr))
            if act_fun == "sigmoid":
                self.addLayer(SigmoidLayer())
            elif act_fun == "tanh":
                self.addLayer(TanhLayer())
            else:
                self.addLayer(ReLULayer())
            prev = curr
    
    def train(self,dataset,epochs=100,optim=SGDOptimizer()):
        loss=MSELoss()
        return super().train(dataset,epochs=epochs,optim=optim,loss=loss)

# Simple MultiLayer Perceptron Class
# This MLPClassifier has been designed for classification problems (so it is mainly using CrossEntropyLoss)
class MLPClassifier(SequentialModel):
    def __init__(self,layer_info=[1,1],act_fun="sigmoid"):
        super().__init__()
        prev = layer_info[0]

        for curr in layer_info[1:]:
            self.addLayer(LinearLayer(prev,curr))
            if act_fun == "sigmoid":
                self.addLayer(SigmoidLayer())
            elif act_fun == "tanh":
                self.addLayer(TanhLayer())
            else:
                self.addLayer(ReLULayer())
            prev = curr
        self.addLayer(SoftmaxLayer())
    
    def train(self,dataset,epochs=100,optim=SGDOptimizer()):
        loss=CrossEntropyLoss()
        return super().train(dataset,epochs=epochs,optim=optim,loss=loss)
