import numpy as np

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
