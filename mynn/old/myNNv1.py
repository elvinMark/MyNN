import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

"""
Neural Network Layers' Class
"""
# Linear Layer Class
class LinearLayer:
    def __init__(self,num_in,num_out):
        self.weights = np.random.random([num_in,num_out])
        self.bias = np.random.random([num_out])
        self.wv = np.random.random([num_in,num_out])
        self.bv = np.random.random([num_out])
        self.wh = np.random.random([num_in,num_out])
        self.bh = np.random.random([num_out])
        
    def forward(self,x):
        self.cache_input = x
        self.cache_output = x.dot(self.weights) + self.bias
        return self.cache_output

    def backward(self,e):
        self.cache_err = e
        return e.dot(self.weights.T)

    def update(self,optim):
        gradW = self.cache_input.T.dot(self.cache_err)
        gradB = self.cache_err.sum(axis=0)
        if optim.optimType == "SGD":
            self.weights -= optim.lr*gradW
            self.bias -= optim.lr*gradB
            
        elif optim.optimType == "Momentum":
            self.wv = optim.alpha*self.wv - optim.lr*gradW
            self.weights += self.wv

            self.bv = optim.alpha*self.bv - optim.lr*gradB
            self.bias += self.bv

        elif optim.optimType == "AdaGrad":
            self.wh += gradW*gradW
            self.weights -= optim.lr*gradW/(np.sqrt(self.wh) + 1e-7)
            self.bh += gradB*gradB
            self.bias -= optim.lr*gradB/(np.sqrt(self.bh) + 1e-7)
            
        elif optim.optimType == "Adam":
            self.wh += gradW*gradW
            self.wv = optim.alpha*self.wv - optim.lr*gradW/(np.sqrt(self.wh) + 1e-7)
            self.weights += self.wv 
            self.bh += gradB*gradB
            self.bv = optim.alpha*self.bv - optim.lr*gradB/(np.sqrt(self.bh) + 1e-7)
            self.bias += self.bv

# Sigmoid Layer Class
class SigmoidLayer:
    def __init__(self):
        self.cache_input = None
        self.cache_output = None
        self.cache_err = None

    def forward(self,x):
        self.cache_input = x
        self.cache_output = sigmoid(x)
        return self.cache_output

    def backward(self,e):
        self.cache_err = e
        return e*sigmoid(self.cache_output,diff=True)

    def update(self,optim):
        pass

# Hyperbolic Tangent Layer
class TanhLayer:
    def __init__(self):
        self.cache_input = None
        self.cache_output = None
        self.cache_err = None

    def forward(self,x):
        self.cache_input = x
        self.cache_output = tanh(x)
        return self.cache_output

    def backward(self,e):
        self.cache_err = e
        return e*tanh(self.cache_output,diff=True)

    def update(self,optim):
        pass

# Rectified Linear unit Layer
class ReLULayer:
    def __init__(self):
        self.cache_input = None
        self.cache_output = None
        self.cache_err = None

    def forward(self,x):
        self.cache_input = x
        self.cache_output = relu(x)
        return self.cache_output

    def backward(self,e):
        self.cache_err = e
        return e*relu(self.cache_output,diff=True)

    def update(self,optim):
        pass

# Softmax Layer (used frequently with Cross entropy loss)
class SoftmaxLayer:
    def __init__(self):
        self.cache_input = None
        self.cache_output = None
        self.cache_err = None

    def forward(self,x):
        self.cache_input = x
        self.cache_output = np.exp(x)
        tmp = self.cache_output.sum(axis=1).reshape(len(x),1)
        self.cache_output /= tmp
        return self.cache_output

    def backward(self,e):
        o = self.cache_output * e
        tmp = o.sum(axis=1).reshape(len(o),1)
        return self.cache_output*(e - tmp)        

    def update(self,optim):
        pass

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


"""
Optimizers
"""

# Stocasthic Gradient Descent
class SGDOptimizer:
    def __init__(self,lr=1):
        self.lr = lr
        self.optimType = "SGD"

# Momentum
class MomentumOptimizer:
    def __init__(self,lr=1,alpha=0.1):
        self.lr = lr
        self.alpha = alpha
        self.optimType = "Momentum"

# Adaptive Gradient (AdaGrad)
class AdaGradOptimizer:
    def __init__(self,lr=1):
        self.lr = lr
        self.optimType = "AdaGrad"

# Adaptive Gradient + Momentum (Adam)
class AdamOptimizer:
    def __init__(self,lr=1,alpha=0.1):
        self.lr = lr
        self.alpha = alpha
        self.optimType = "Adam"


"""
Dataset

it will help to loop over data using an specific batch size
"""

# Dataset class
class myDataset():
    def __init__(self,x,y,batch_size=1):
        if len(x) != len(y):
            raise Error()
        self.num_elems = len(x)
        self.curr = 0
        self.x = x
        self.y = y
        self.batch_size = batch_size
        
    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr == self.num_elems:
            raise StopIteration()
        
        tmp = self.curr
        if self.curr < self.num_elems - self.batch_size:
            self.curr += self.batch_size
            return self.x[tmp:tmp + self.batch_size], self.y[tmp : tmp + self.batch_size]

        self.curr = self.num_elems
        
        return self.x[tmp:], self.y[tmp:]


"""
Models
"""

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
        # self.layers = []
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


"""
Useful functions
"""

# Function to generate random data (points classified in different regions)
def generateRandomClassifiedData(regions=[((0,0),(1,1))],N=20):
    num_classes = len(regions) + 1
    x = np.random.random([N,2])*2 - 1
    y = np.array([[0]*num_classes for i in range(N)])
    
    inside = lambda x,y,R : x>=R[0][0] and x<=R[1][0] and y>=R[0][1] and y<=R[1][1]
    
    for k,(i,j) in enumerate(x):
        found = -1
        for c in range(num_classes - 1):
            if inside(i,j,regions[c]):
                found = c
                break
        y[k][found] = 1
        
    return x,y

if __name__ == "__main__":
    # x = np.array([[0,0],[1,0],[0,1],[1,1]])
    # y = np.array([[1,0],[0,1],[0,1],[1,0]])

    # Create random points classified in different regions
    x,y = generateRandomClassifiedData(regions=[((-0.8,-0.8),(-0.1,-0.1)),((0,0),(0.8,0.8))])
    
    # Create dataset from the generated data 
    ds = myDataset(x,y,batch_size=10)

    # Create model (MLP)
    # nn = MLPRegressor(layer_info=[2,3,3],act_fun="relu") 
    nn = MLPClassifier(layer_info=[2,5,3,3],act_fun="relu")

    # Start training
    print("Training ...")
    #log_loss = nn.train(ds,epochs=1000,optim=SGDOptimizer(lr=0.01))
    #log_loss = nn.train(ds,epochs=1000,optim=MomentumOptimizer(lr=0.01,alpha=0.5))
    #log_loss = nn.train(ds,epochs=1000,optim=AdaGradOptimizer(lr=0.01))
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
