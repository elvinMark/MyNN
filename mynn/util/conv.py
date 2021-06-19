import numpy as np
from numpy.lib.stride_tricks import as_strided

def myconv2d(x,y,padding = 0, stride = 1):
    x = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)),mode='constant',constant_values=(0,0))
    N,Cin,H,W = x.shape
    Co,Cin,Fh,Fw = y.shape
    prev_strides = x.strides
    new_shape = (N,Cin,(H - Fh + stride)//stride,(W - Fw + stride)//stride,Fh,Fw)
    new_strides = prev_strides[:2] + (stride*prev_strides[2], stride*prev_strides[3]) + prev_strides[2:]
    new_x = as_strided(x,new_shape,new_strides)
    return np.einsum("ijklmn,ajmn->iakl",new_x,y)
