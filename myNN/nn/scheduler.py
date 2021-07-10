import numpy as np

class ConstantLRScheduler:
    def __init__(self,val):
        self.val = val

    def get_lr(self,epoch):
        return self.val

class StepLRScheduler:
    def __init__(self,init_val):
        self.init_val = init_val

    def get_lr(self,epoch):
        if epoch < 60:
            return self.init_val
        elif epoch < 120:
            return self.init_val / 5
        elif epoch < 160:
            return self.init_val / 25
        elif epoch < 200:
            return self.init_val / 125
        else:
            return self.init_val / 625

class CosineLRScheduler:
    def __init__(self,eta_max=0.1,eta_min=5e-4,t_max=200):
        self.init_val = init_val
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.t_max = t_max
        
    def get_lr(self,epoch):
        return self.eta_min + 0.5*(self.eta_max - self.eta_min)*(1 + np.cos(epoch * np.pi / self.t_max))
