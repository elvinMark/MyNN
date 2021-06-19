import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from myNN.nn.layer import *
from myNN.nn.loss import *
from myNN.nn.optim import *
from myNN.arch.mlp import MLPRegressor
from myNN.util.dataset import myDataset
from myNN.util.misc import generateRandomClassifiedData

if __name__ == "__main__":
    # Create random points classified in different regions
    x,y = generateRandomClassifiedData(regions=[((-0.8,-0.8),(-0.1,-0.1)),((0,0),(0.8,0.8))])
    
    # Create dataset from the generated data 
    ds = myDataset(x,y,batch_size=10)

    # Create model (MLP)
    nn = MLPRegressor(layer_info=[2,3,3],act_fun="relu") 

    # Start training
    print("Training ...")
    log_loss = nn.train(ds,epochs=1000,optim=AdamOptimizer(lr=0.1,alpha=0.5))
    
    print("Test")
    print(nn.forward(x))

    # Plotting loss evolution
    plt.plot(log_loss)
    plt.grid()
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
