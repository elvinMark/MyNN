import numpy as np
from tqdm import tqdm

from myNN.nn.optim import *
from myNN.nn.loss import *

def train_model(model,dataset,epochs=100,optim=SGDOptimizer(),loss=MSELoss()):
    log_loss = []
    for epoch in tqdm(range(epochs)):
        l = 0
        for x,y in dataset:
            o = model.forward(x)
            l += loss.calculateLoss(o,y)
            e = loss.getGrad()
            model.backward(e)
            model.update(optim)
            
        log_loss.append(l)
            
    return log_loss
