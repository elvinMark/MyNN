import numpy as np

def create_parameters(std,shape):
    return std*(np.random.random(shape) - 0.5)
