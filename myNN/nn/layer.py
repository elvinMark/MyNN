import numpy as np
from numpy.lib.stride_tricks import as_strided
from myNN.util.actfun import *
from myNN.util.conv import *
from myNN.util.param_init import *

"""
Neural Network Layers' Class
"""

# Linear Layer Class
class LinearLayer:
    """
    Linear Layer:

    it applies a linear transformation to the input: A*x + b.

    Args
    
    """
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

# Batch Normalization 1D and 2D
class BatchNormalization1D:
    def __init__(self,channels,eps=5e-4):
        self.channels = channels
        self.eps = eps
        self.gamma = np.ones((channels,1))
        self.beta = np.zeros((channels,1))

        self.wv = np.zeros((channels,1))
        self.bv = np.zeros((channels,1))
        self.wh = np.zeros((channels,1))
        self.bh = np.zeros((channels,1))

    def forward(self,x):
        self.cache_input = x
        N,C = x.shape
        x_ = x.transpose((1,0))
        self.mean = x_.sum(axis=1) / N
        self.mean = self.mean.reshape((C,1))
        self.sigma2 = ((x_ - self.mean)**2).sum(axis=1)/ N
        self.sigma2 = self.sigma2.reshape((C,1))
        self.isigma = 1/np.sqrt(self.sigma2 + self.eps)
        self.z = (x_ - self.mean)*self.isigma
        self.cache_output = self.z * self.gamma + self.beta
        self.cache_output = self.cache_output.transpose((1,0))
        return self.cache_output

    def backward(self,err):
        N,C = err.shape
        self.cache_err = err.transpose((1,0))
        self.dgamma = (self.cache_err * self.z).sum(axis=1).reshape((C,1))
        self.dbeta = self.cache_err.sum(axis=1).reshape((C,-1))
        tmp = (self.cache_err - self.dbeta/N - self.z * self.dgamma /N) * self.gamma * self.isigma
        return tmp.transpose((1,0))
        
    def update(self,optim):
        if optim.optimType == "SGD":
            self.gamma -= optim.lr*self.dgamma
            self.beta -= optim.lr*self.dbeta
            
        elif optim.optimType == "Momentum":
            self.wv = optim.alpha*self.wv - optim.lr*self.dgamma
            self.gamma += self.wv

            self.bv = optim.alpha*self.bv - optim.lr*self.dbeta
            self.beta += self.bv

        elif optim.optimType == "AdaGrad":
            self.wh += self.dgamma * self.dgamma
            self.gamma -= optim.lr*self.dgamma/(np.sqrt(self.wh) + 1e-7)
            self.bh += self.dbeta * self.dbeta
            self.beta -= optim.lr*self.dbeta/(np.sqrt(self.bh) + 1e-7)
            
        elif optim.optimType == "Adam":
            self.wh += self.dgamma * self.dgamma
            self.wv = optim.alpha*self.wv - optim.lr*self.dgamma/(np.sqrt(self.wh) + 1e-7)
            self.gamma += self.wv 
            self.bh += self.dbeta * self.dbeta
            self.bv = optim.alpha*self.bv - optim.lr*self.dbeta/(np.sqrt(self.bh) + 1e-7)
            self.beta += self.bv

    def __call__(self,x):
        return self.forward(x)


class BatchNormalization2D:
    def __init__(self,channels,eps=5e-4):
        self.channels = channels
        self.eps = eps
        self.gamma = np.ones((channels,1))
        self.beta = np.zeros((channels,1))

        self.wv = np.zeros((channels,1))
        self.bv = np.zeros((channels,1))
        self.wh = np.zeros((channels,1))
        self.bh = np.zeros((channels,1))

    def forward(self,x):
        self.cache_input = x
        N,C,H,W = x.shape
        M = N*H*W
        x_ = x.transpose((1,0,2,3)).reshape((C,-1))
        self.mean = x_.sum(axis=1) / M
        self.mean = self.mean.reshape((C,1))
        self.sigma2 = ((x_ - self.mean)**2).sum(axis=1)/ M
        self.sigma2 = self.sigma2.reshape((C,1))
        self.isigma = 1/np.sqrt(self.sigma2 + self.eps)
        self.z = (x_ - self.mean)*self.isigma
        self.cache_output = self.z * self.gamma + self.beta
        self.cache_output = self.cache_output.reshape((C,N,H,W)).transpose((1,0,2,3))
        return self.cache_output

    def backward(self,err):
        N,C,H,W = err.shape
        M = N*H*W
        self.cache_err = err.transpose((1,0,2,3)).reshape(C,-1)
        self.dgamma = (self.cache_err * self.z).sum(axis=1).reshape((C,1))
        self.dbeta = self.cache_err.sum(axis=1).reshape((C,-1))
        tmp = (self.cache_err - self.dbeta/M - self.z * self.dgamma /M) * self.gamma * self.isigma
        return tmp.reshape((C,N,H,W)).transpose((1,0,2,3))
        
    def update(self,optim):
        if optim.optimType == "SGD":
            self.gamma -= optim.lr*self.dgamma
            self.beta -= optim.lr*self.dbeta
            
        elif optim.optimType == "Momentum":
            self.wv = optim.alpha*self.wv - optim.lr*self.dgamma
            self.gamma += self.wv

            self.bv = optim.alpha*self.bv - optim.lr*self.dbeta
            self.beta += self.bv

        elif optim.optimType == "AdaGrad":
            self.wh += self.dgamma * self.dgamma
            self.gamma -= optim.lr*self.dgamma/(np.sqrt(self.wh) + 1e-7)
            self.bh += self.dbeta * self.dbeta
            self.beta -= optim.lr*self.dbeta/(np.sqrt(self.bh) + 1e-7)
            
        elif optim.optimType == "Adam":
            self.wh += self.dgamma * self.dgamma
            self.wv = optim.alpha*self.wv - optim.lr*self.dgamma/(np.sqrt(self.wh) + 1e-7)
            self.gamma += self.wv 
            self.bh += self.dbeta * self.dbeta
            self.bv = optim.alpha*self.bv - optim.lr*self.dbeta/(np.sqrt(self.bh) + 1e-7)
            self.beta += self.bv

    def __call__(self,x):
        return self.forward(x)

# Dropout Layer in progress ...
class DropoutLayer1D:
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

# Max Pooling Layer in progress ... 
class MaxPool2DLayer:
    def __init__(self,kernel_size=2):
        self.kernel_size = kernel_size
        
    def forward(self,x):
        N,C,H,W = x.shape
        new_shape = (N,C,H//self.kernel_size,W//self.kernel_size,self.kernel_size,self.kernel_size)
        (s1,s2,s3,s4) = x.strides
        new_stride = (s1,s2,self.kernel_size*s3,self.kernel_size*s4,s3,s4)
        x_ = as_strided(x,new_shape,new_stride).reshape((N,C,H//self.kernel_size,W//self.kernel_size,-1))
        self.max_idx = np.argmax(x_,axis=4)
        return np.argmax(x_,axis=4)
    
    def backward(self,x):
        pass

    def update(self,optim):
        pass

    def __call__(self,x):
        return self.forward(x)
    
