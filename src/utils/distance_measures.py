import numpy as np
from math import sqrt

def euclidean_distance(x, y, dims=None):
    if dims is not None:
        x, y = x[:dims], y[:dims]
    
    return np.sqrt(np.sum((x - y) ** 2)) 


def manhattan_distance(x, y, dims=None):
    if dims is not None:
        x, y = x[:dims], y[:dims]
    
    return np.sum(np.absolute(x-y))


def chebyshev_distance(x, y, dims=None):
    if dims is not None:
        x, y = x[:dims], y[:dims]
    
    return np.max(np.absolute(x-y))


def cosine_distance(x, y, dims=None):
    if dims is not None:
        x, y = x[:dims], y[:dims]
    
    n = np.sum(x * y)
    a = np.sqrt(np.sum(x**2))
    b = np.sqrt(np.sum(y**2))
    
    return 1 - (n / (a*b))