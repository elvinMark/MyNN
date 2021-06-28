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
