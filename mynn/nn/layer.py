import numpy as np
from myNN.util.actfun import *
from myNN.util.conv import *
from myNN.util.param_init import *

"""
Neural Network Layers' Class
"""
# Linear Layer Class
class LinearLayer:
    def __init__(self,num_in,num_out):
        std = 1/np.sqrt(num_in)
        # self.weights = std*(np.random.random([num_in,num_out]) - 0.5)
        # self.bias = std*(np.random.random([num_out]) - 0.5)
        self.weights = create_parameters(std,(num_in,num_out))
        self.bias = create_parameters(std,(num_out))
        
        self.wv = np.zeros([num_in,num_out]) 
        self.bv = np.zeros([num_out])
        self.wh = np.zeros([num_in,num_out])
        self.bh = np.zeros([num_out])
        
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

    def __call__(self,x):
        return self.forward(x)


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

    def __call__(self,x):
        return self.forward(x)

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

    def __call__(self,x):
        return self.forward(x)

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

    def __call__(self,x):
        return self.forward(x)

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

    def __call__(self,x):
        return self.forward(x)

# Dropout Layer in progress ...
class DropoutLayer:
    def __init__(self,prob=0.5):
        self.prob = prob

    def forward(self,x):
        pass

    def backward(self,x):
        pass

    def update(self,optim):
        pass

    def __call__(self,x):
        return self.forward(x)

# Batch Normalization Layer in progress ...
class BatchNormalization:
    def __init__(self):
        pass

    def forward(self,x):
        pass

    def backward(self,x):
        pass

    def update(self,optim):
        pass

    def __call__(self,x):
        return self.forward(x)


# Convolutional Layer (2D)
class Conv2DLayer:
    def __init__(self,cin,cout,kernel_size=3,padding=0,stride=1,bias=False):

        std = 1/np.sqrt(cin)
        
        self.cin = cin
        self.cout = cout
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # self.weight = std * (np.random.random((cout,cin,kernel_size,kernel_size)) - 0.5)
        self.weight = create_parameters(std,(cout,cin,kernel_size,kernel_size))

        self.bias = None
        if bias:
            # self.bias = std*(np.random.random(cout) - 0.5)
            self.bias = create_parameters(std,(cout))
        
        self.wv = np.zeros(self.weight.shape)
        self.wh = np.zeros(self.weight.shape)

        if bias:
            self.bv = np.zeros(self.bias.shape)
            self.bh = np.zeros(self.bias.shape)
        
    def forward(self,x):
        self.cache_input = x
        self.cache_output =  myconv2d(x,self.weight,padding=self.padding,stride=self.stride)
        if not self.bias is None:
            tmp = as_strided(self.bias,self.cache_output.shape[1:],(self.bias.strides[0],0,0))
            self.cache_output = self.cache_output + tmp
        return self.cache_output
    
    def backward(self,x):
        self.cache_error = x
        Co,Ci,Fh,Fw = self.weight.shape
        N,Co,H,W = x.shape
        
        self.dilatated_error = np.zeros((N,Co,H + (H - 1)*(self.stride-1),W + (W - 1)*(self.stride-1)))
        self.dilatated_error[:,:,::self.stride,::self.stride] = x

        wT = np.flip(self.weight.transpose((1,0,2,3)),(3,2,1))
        tmp = myconv2d(self.dilatated_error,wT,padding=self.kernel_size - 1)
        
        if self.padding != 0:
            return tmp[:,:,self.padding:-self.padding,self.padding:-self.padding]
        return tmp
    
    def update(self,optim):
        tmpI = self.cache_input.transpose((1,0,2,3))
        tmpE = self.dilatated_error.transpose((1,0,2,3))

        gradW = myconv2d(tmpI,tmpE,padding=self.padding).transpose((1,0,2,3))

        if not self.bias is None:
            gradB = self.dilatated_error.sum(axis=(0,2,3))

        if optim.optimType == "SGD":
            self.weight -= optim.lr * gradW
            if not self.bias is None:
                self.bias -= optim.lr * gradB
            
        elif optim.optimType == "Momentum":
            self.wv = optim.alpha*self.wv - optim.lr * gradW
            self.weight += self.wv

            if not self.bias is None:
                self.bv = optim.alpha * self.bv - optim.lr * gradB
                self.bias += self.bv
            
        elif optim.optimType == "AdaGrad":
            self.wh += gradW * gradW
            self.weight -= optim.lr * gradW / (np.sqrt(self.wh) + 1e-7)

            if not self.bias is None:
                self.bh += gradB*gradB
                self.bias -= optim.lr * gradB / (np.sqrt(self.bh) + 1e-7)
        elif optim.optimType == "Adam":
            self.wh += gradW * gradW
            self.wv = optim.alpha*self.wv - optim.lr * gradW / (np.sqrt(self.wh) + 1e-7)
            self.weight += self.wv
            if not self.bias is None:
                self.bh += gradB * gradB
                self.bv = optim.alpha * self.bv - optim.lr * gradB / (np.sqrt(self.bh) + 1e-7)
                self.bias += self.bv
    
    def __call__(self,x):
        return self.forward(x)

# Max Pooling Layer in progress ... 
class MaxPool2DLayer:
    def __init__(self,kernel_size=2):
        self.kernel_size = kernel_size
        
    def forward(self,x):
        pass

    def backward(self,x):
        pass

    def update(self,optim):
        pass

    def __call__(self,x):
        return self.forward(x)
    
