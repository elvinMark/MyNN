# myNN

This is a simple library implementing some simple Neural Network Algorithms. This library is not optimized, it is just for study purposes

---

## Activation Functions

An activation function simulates what real neurons do inside our brains. Decide wether a neuron should trigger a signal to the next neuron or not based on the input signal. 

Based on this definition some functions that simulates this behaviour were deviced. Among all of those the most famous ones are detailed below.

### Sigmoid

Defined as below:
$f(x) = \frac{1}{1 + e^{-x}}$
The plot of this function is as shown in the following graph:
![sigmoid graph](https://github.com/rioyokotalab/startup_program_elvin/blob/master/myNN/images/sigmoid.png)

If look closely to the shape of the graph, we can notice that it resembles a softer version of the step function. That is actually why this function was chosen as an activation function in the first place. The first function used for activation function was actually the step function since we need a function that decides whether to trigger a signal or not in other words a function that given an input gives 1 or 0. But because, as we will see later, we need the derivative of the function for training purposes and the derivative of the step function is 0 almost everywhere and where is not 0 it is actually $\infty$ then it was not so useful for trainig. That is why a "smooth" version with non-zero derivative of the step function was chosen: the sigmoid function.

### Hyperbolic Tangent 

Defined as below:
$f(x) = \frac{1 - e^{-x}}{1 + e^{-x}}$

The plot of this function is as shown in the following graph:
![hyperbolic tangent graph](https://github.com/rioyokotalab/startup_program_elvin/blob/master/myNN/images/tanh.png)

This graph's shape is actually quite the same with the sigmoid. The only difference is that its range is not $[0,1]$ but $[-1,1]$. Why this is necessary you may wonder, well the answer relies as before in the derivative of the function. Since the range is wider, this function offers more pronounce values of the derivative that helps the training of the network to be faster.

### Rectified Linear Unit (ReLU)

Classically is defined as shown in the equation below:
$f(x) = \Bigg\{ \begin{matrix}
x & x>0\\
0 & \text{otherwise}
\end{matrix}$

But for some specific reasons (that will be explain in detail later) it is defined as below (this variation is called Leaky ReLU):

$f(x) = \Bigg\{ \begin{matrix}
x & x>0\\
0.01x & \text{otherwise}
\end{matrix}$

The plot of this function is as shown in the following graph
![relu graph](https://github.com/rioyokotalab/startup_program_elvin/blob/master/myNN/images/relu.png)

This function offers something similar to previous functions: a value for not triggering the signal which in this case is $0$ and a value for triggering the signal which in this case, contrary to the other function where it was taken as $1$, it is the same as the input $x$.

The reason why this function was deviced and why it has become popular over the years is again its derivative. This function is easy to derivate, we know it will be $1$ for positive values and $0$ for negative ones. So it is easy to calculate when training the network. But, as you may recall from the previous detailed activation functions, having derivatives equal to $0$ is not a good thing to have when training, since the training process uses the derivative for the update of the neuron. To cope with this problem, this function was slightly modified in the output for negative input values. Instead of just output $0$, we output a really close value to $0$ which in this case, we have set to $0.01x$. Why we have chosen a linear function you may ask, well because again it is easy to derivate, we know its derivative will $0.01$.

---

## Layers

All layers we can think it as box that has some inputs and returns some outputs.

Basically we have to implement 3 methods for all layers:

- forward : calculate the output base on the given input
- backward : backpropagate the error to the previous layer
- update : updates the inner parameter used in the layer depending on the optimizer being used (this update will occur during training)

All Layer classes have the following structure:

``` python
class Layer:
    
    ...
    
    def forward(self,x):
        ...
        
    def bacward(self,x):
        ...
        
    def update(self,optim):
        ...
```

### Linear Layer

The linear layer is basically just a linear transformation. In other words:
Given a matrix $A \in \mathbb{R}^{n\times m}$ and a vector $\vec{b} \in \mathbb{R}^{1\times m}$
$f: \mathbb{R}^{1\times n} \rightarrow \mathbb{R}^{1\times m}$
$f(\vec{x}) = \vec{x}A + \vec{b} = \vec{y}$

##### note:
Normally linear transformations are written as $A\vec{x} + b = \vec{y}$ given that $\vec{x}$ and $\vec{y}$ are column vectors but in this case we are taking them as row vectors so we take the  $\vec{x}A + \vec{b}= \vec{y}$ shape instead.

The $A_{ij}$ element of the matrix $A$, also called weight matrix, represents how "strong" an $i^{th}$ neuron in the input layer is connected to $j^{th}$ neuron in the output layer.

And the $b_{j}$ element of the vector $\vec{b}$, also called bias,  represents the threshold of the $j^{th}$ neuron in the output layer. This threshold determines from which value the neuron should shoot.

#### Forward

The forward method was implemented following the linear transformation specified before.

$\vec{y} = \vec{x}A + \vec{b}$

Here $\vec{x}$ is the input and $\vec{y}$ is the obtained output.

#### Backward

We will have a function $L$ that depends on all the parameters in the neural network, the input and the desired output. This function called ***Loss Function*** will determined how "far" our current output is from the desired output given the current parameters.

In order to tune the parameters of our neural network we will need the partial derivative of $L$ with respect to those parameters.

In the case of this Linear Layer, our parameters are the matrix $A$ and the vector $b$ so we will need $\frac{\partial{L}}{\partial{A}}$ and $\frac{\partial{L}}{\partial{\vec{b}}}$. Since we now that our loss function is function, among other variables, of the current output, and we also know that our output is function of the parameters and the input. We can calculate the the needed partial derivatives in the following way:

$\frac{\partial{L}}{\partial{A}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{A}} = \vec{x} \frac{\partial{L}}{\partial{\vec{y}}}$

$\frac{\partial{L}}{\partial{\vec{b}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{b}}} = \mathbb{1}\frac{\partial{L}}{\partial{\vec{y}}}$

##### note:
Here $\mathbb{1}$ is a row vector with all its elements equal to 1.

As you can see from above, to get $\frac{\partial{L}}{\partial{A}}$ and $\frac{\partial{L}}{\partial{\vec{b}}}$, we will need $\frac{\partial{L}}{\partial{\vec{y}}}$ which is the partial derivative of the loss function with respect the current output of this layer.

Since the previous layer will need this information as well, we will have to backpropagate this information to it. In order to do so we will use once again the chain rule. 


$\frac{\partial{L}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}A^{T}$

In this way the previous layer will have the information of the partial derivative of the loss function with respect its output (which it was the current input of this layer)
##### note:
Here we are using the Einstein Summation 
Convention:

$\sum_{i} x_iy_i = x_iy_i$

When we have repeated indices we will sum through that index

#### Update

The update of $A$ and $b$ will be done depending on the optimizer that we will see later but basically it will be something like this.

$A = A + g(\frac{\partial{L}}{\partial{A}})$ 
$\vec{b} = \vec{b} + g(\frac{\partial{L}}{\partial{\vec{b}}})$

Where g is a function of the derivative of the loss function with respect the corresponding parameter. The form of this function depends on the optimizer that is being used.

#### How to create a LinearLayer object
``` python
num_in = 3
num_out = 2
layer = LinearLayer(num_in,num_out)
```

### Sigmoid Layer

The Sigmoid layer just apply the sigmoid function to all its inputs.

$sigmoid(x) = \frac{1}{1 + e^{-x}}$

#### Forward

$\vec{y} = sigmoid(\vec{x})$

#### Backward

Following the same idea as in the Linear Layer, to backpropagate the error we just need to calculate:

$\frac{\partial{L}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}sigmoid'(\vec{x})$

##### note:
$sigmoid'(x)$ is the derivative of the sigmoid function.

$sigmoid'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = sigmoid(x)\times (1 - sigmoid(x))$

#### Update

Since this layer actually does not have any parameters to train, there is nothing to update here.

#### How to create a SigmoidLayer object

``` python
layer = SigmoidLayer()
```

### Tanh Layer

The Tanh layer just apply the hyperbolic tangent function to all its inputs.

$tanh(x) = \frac{1 - e^{-x}}{1 + e^{-x}}$

#### Forward

$\vec{y} = tanh(\vec{x})$

#### Backward

Following the same idea as in the Linear Layer, to backpropagate the error we just need to calculate:

$\frac{\partial{L}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}tanh'(\vec{x})$

##### note:
$tanh'(x)$ is the derivative of the tanh function.

$tanh'(x) = \frac{2e^{-x}}{(1 + e^{-x})^2} = \frac{1}{2}(1 - tanh(x)^2))$

#### Update

Since this layer actually does not have any parameters to train, there is nothing to update here.

#### How to create a TanhLayer object

``` python
layer = TanhLayer()
```

### ReLU Layer

The Relu layer just apply the (leaky) rectified linear unit function to all its inputs.

$relu(x) = \Bigg\{ \begin{matrix}
x & x>0\\
0.01x & \text{otherwise}
\end{matrix}$

#### Forward

$\vec{y} = relu(\vec{x})$

#### Backward

Following the same idea as in the Linear Layer, to backpropagate the error we just need to calculate:

$\frac{\partial{L}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}relu'(\vec{x})$

##### note:
$relu'(x)$ is the derivative of the relu function.

$relu'(x) = \Bigg\{ \begin{matrix}
1 & x>0\\
0.01 & \text{otherwise}
\end{matrix}$

#### Update

Since this layer actually does not have any parameters to train, there is nothing to update here.

#### How to create a ReluLayer object

``` python
layer = ReluLayer()
```

### Softmax Layer

Normally for classification problems we will have some predefined classes and we want our ouput will indicate which class the input more likely belongs to. For example if we have 3 classes. and we want to say that our input belongs to the second class, then our output should like something like: $[0,1,0]$ which represents that there is zero probability of the input belonging to class 1 and class 3 is zero but the probabily of the input belonging to class 2 is 1. Actually as you may already noticed, our output will be a probability distribution. Of course in the ideal case we will have just one peak (1) in the corresponding class and zero in the others, but in reallity we will have a probability distribution with a peak in the desired class but still some probability (although small compared to the peak) of the input belonging to other classes. We will see how we can measure how our probability distribution differs from the desired probability distribution (the one with peak 1) in the loss function section. Anyways going back to main point, as we have already stated, we want our output to be a probability distribution, in other words we want our output to have the following property: $\sum_i y_i = 1$ it has to be equal to $1$ since it is a probability distribution and the the probability of the input belonging to one of the class has to be 1. The output of the previous detailed layers will not actually give this kind of distribution. So we have to "adjust" these output to this kind of probability distribution. This is when Softmax layer comes into play. Imagine the we have $\vec{x}$ as the output from a previous layer. If we want to convert this to a certain $\vec{y}$ that follows the previously defined condition we can perform the following operation.

$y_j = f(\vec{x})_j = \frac{e^{x_j}}{\sum_k e^{x_k}}$

As you may have already noticed if we sum all $y_j$ we will get $1$.

$\sum_j y_j = 1$

We may have chosen a different function that kind of give the same result. For example:

$y_j = \frac{x_j}{\sum_k x_k}$

and still get the same property of $\sum_j y_j = 1$. So why did we choose the previous function instead. Well the answer is actually relates with physics (actually I find it interesting how many stuff related to artificial intelligence is somehow related with physics like we will see later in the optimizer section). And that is actually related with partition function which are used in statistical mechanics. In statistical mechanics we deal with tons of particle. So we cannot define an specific function that defines each and every particle. Instead we device some function that can give some information about the system as a whole. That is what partition functions actually tell us. It actually tell us how many particles are in an specific energy level. Normally in physics, particles try to go if possible to lowest energy level possible and that is why the actualy particle distribution looks something like this:

$N(E_k) = \frac{e^{-\alpha E_K}}{\sum_i e^{-\alpha E_i}}$

This formula is not just random picked but it is actually derivated from permutations and limits but we will not go into those details. We will just use that fact that this function will distribute properly the particles in order that we will have more particles in the lowest level. We can rephrase this in a different way, this function will give a distribution probability of finding a particle in an specific energy level. And this function will have a maximum probability for lowest energy levels. 

This is quite similar to what we actually want our layer to do. We will have some outputs from the previous layer and we want to transform those outputs such that where we had higher values (analogy to lowest energy level), it is more likely to have the right class there (analogy to find the particle in that energy level). Since in this case we are not looking for the higuer probability to be in the lowest level but rather in the higher level we dont use minus sign in the exponential but a positive one instead.

#### Forward

Forwarding in this layer is just aplying the said softmax function.

$y_j = softmax(\vec{x})_j = \frac{e^{x_j}}{\sum_k e^{x_k}}$

#### Backward

Following the same idea as in the Linear Layer, to backpropagate the error we just need to calculate:

$\frac{\partial{L}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}\frac{\partial{\vec{y}}}{\partial{\vec{x}}} = \frac{\partial{L}}{\partial{\vec{y}}}softmax'(\vec{x})$

##### note:
$softmax'(x)$ is the derivative of the relu function.

$y_j = [softmax(\vec{x})]_j$
$\frac{\partial y_j}{\partial x_i} = \frac{\delta_{ij} e^{x_j}\sum_k e^{x_k} - e^{x_j} e^{x_i}}{(\sum_k e^{x_k})^2} =\delta_{ij}y_j - y_jy_i$

Replacing $\frac{\partial y_j}{\partial x_i}$ in the previous equation we obtained:

$[\frac{\partial{L}}{\partial{\vec{x}}}]_i= \sum_j \frac{\partial{L}}{\partial{y_j}}\frac{\partial{y_j}}{\partial{x_i}} = \sum_j \frac{\partial{L}}{\partial{y_j}}(\delta_{ij}y_j - y_jy_i) =y_i\frac{\partial{L}}{\partial{y_i}} - y_i (\sum_j y_j\frac{\partial{L}}{\partial{y_j}})$

##### note:
$\delta_{ij}$ is called kronecker delta and it takes the value of $1$ if $i = j$ and $0$ if $i\neq j$.

#### Update

Since this layer actually does not have any parameters to train, there is nothing to update here.

#### How to create a SoftmaxLayer object
``` python
layer = SoftmaxLayer()
```

### Dropout Layer

#### Forward
#### Backward
#### Update

#### How to create a DropoutLayer object
``` python
layer = DropoutLayer()
```

### Batch Normalization Layer

#### Forward
#### Backward
#### Update

#### How to create a BatchNormalizationLayer object
``` python
layer = BatchNormalizationLayer()
```

### Convolution Layer

$I= [I_{ij}]$
$F= [F_{ij}]$
$A= [A_{ij}]$
$A_{mn} = \sum_{k,l} I_{m+k,n+l}F_{kl}$

$\frac{\partial A_{mn}}{\partial I_{ij}} = F_{i-m,j-n}$

$\frac{\partial A_{mn}}{\partial F_{ij}} = I_{m+i,n + j}$




#### Forward

#### Backward

$\frac{\partial L}{\partial I_{ij}} = \sum_{m,n}\frac{\partial L}{\partial A_{mn}}\frac{\partial A_{mn}}{\partial I_{ij}}=\sum_{m,n}\frac{\partial L}{\partial A_{mn}}F_{i-m,j-n}$

#### Update

$\frac{\partial L}{\partial F_{ij}} = \sum_{m,n}\frac{\partial L}{\partial A_{mn}}\frac{\partial A_{mn}}{\partial F_{ij}}=\sum_{m,n}\frac{\partial L}{\partial A_{mn}}I_{m+i,n+j}$

#### How to create a Conv2DLayer object
``` python
layer = Convolution2DLayer(num_kernels,kernel_size=3)
```

### Max Pool Layer

#### Forward
#### Backward
#### Update

#### How to create a MaxPool2DLayer object
``` python
layer = MaxPool2DLayer(kernel_size=3)
```


---

## Loss Functions

``` python
class Loss:
    
    ...

    def calculateLoss(self,o,t):
        ...
        
    def getGrad(self):
        ...
```

### Mean Square Error Loss

#### Loss Calculation
$\vec{o}$ is the obtained output and $\vec{t}$ is the "target" or desired output.

$L(\vec{o},\vec{t}) = \frac{1}{2}\sum_j (o_j - t_j)^2 = \frac{1}{2}(\vec{o} - \vec{t})\cdot(\vec{o} - \vec{t})$

#### Gradient (SGD Optimizer)

$\frac{\partial L}{\partial \vec{o}} = \vec{o} - \vec{t}$

``` python
loss = MSELoss()
```

### Cross Entropy Loss

#### Loss Calculation

$L(\vec{o},\vec{t}) = \sum_i -t_i \log{o_i}$

#### Gradient (SGD Optimizer)

$\frac{\partial L}{\partial o_i} = -\frac{t_i}{o_i}$

``` python
loss = CrossEntropyLoss()
```

---

## Optimizers

``` python
class Optimizer:
    def __init__(self,lr): # and other parameters  if needed
        ...
```

### Stochastic Gradient Descent (SGD)

$W^{k+1} = W^{k} - \alpha \frac{\partial L}{\partial W^{k}}$

### Momentum 

$v^{k+1} = \beta v^{k} - \alpha \frac{\partial L}{\partial W^{k}}$
$W^{k+1} = W^{k} + v^{k+1}$


### Adaptative Gradient (AdaGrad)

$h^{k+1} = h^{k} + \frac{\partial L}{\partial W^{k}} \times \frac{\partial L}{\partial W^{k}}$

$W^{k+1} = W^{k} - \alpha\frac{1}{\sqrt{h^{k+1}}}\frac{\partial L}{\partial W^{k}}$

### Adaptative Gradient + Momentum (Adam)
$h^{k+1} = h^{k} + \frac{\partial L}{\partial W^{k}} \times \frac{\partial L}{\partial W^{k}}$

$v^{k+1} = \beta v^{k} - \alpha\frac{1}{\sqrt{h^{k+1}}} \frac{\partial L}{\partial W^{k}}$

$W^{k+1} = W^{k} + v^{k+1}$


---
## Multilayer Perceptron

### Regressor

``` python
model = MLPRegressor()
```

### Classifier

``` python
model = MLPClassifier()
```

### Sample

---
