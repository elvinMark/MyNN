import sys
sys.path.append("../")

from myNN.nn.layer import *
from myNN.nn.loss import *
from myNN.nn.optim import *
from myNN.util.dataset import myDataset
from myNN.arch.mlp import MLPClassifier
from myNN.util.misc import generateRandomClassifiedData
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # Create random points classified in different regions
    x,y = generateRandomClassifiedData(regions=[((-0.8,-0.8),(-0.1,-0.1)),((0,0),(0.8,0.8))],N=300)

    # Create dataset from the generated data 
    ds = myDataset(x,y,batch_size=20)

    # Create models (MLP)
    nn1 = MLPClassifier(layer_info=[2,5,3,3],act_fun="relu")
    nn2 = MLPClassifier(layer_info=[2,5,3,3],act_fun="relu")
    nn3 = MLPClassifier(layer_info=[2,5,3,3],act_fun="relu")
    nn4 = MLPClassifier(layer_info=[2,5,3,3],act_fun="relu")

    # Train each model using different optimizers
    print("Training ...")
    log_loss1 = nn1.train(ds,epochs=1000,optim=SGDOptimizer(lr=0.1))
    log_loss2 = nn2.train(ds,epochs=1000,optim=MomentumOptimizer(lr=0.1,alpha=0.2))
    log_loss3 = nn3.train(ds,epochs=1000,optim=AdaGradOptimizer(lr=0.1))
    log_loss4 = nn4.train(ds,epochs=1000,optim=AdamOptimizer(lr=0.1,alpha=0.2))
    
    # Plotting loss evolution
    plt.plot(log_loss1,label="SGD Optimizer")
    plt.plot(log_loss2,label="Momentum Optimizer")
    plt.plot(log_loss3,label="AdaGrad Optimizer")
    plt.plot(log_loss4,label="Adam Optimizer")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("../images/compareOptimizers.png")

