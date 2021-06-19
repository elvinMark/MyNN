import numpy as np

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
