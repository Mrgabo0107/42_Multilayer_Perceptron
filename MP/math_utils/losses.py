import numpy as np

def categorical_cross_entropy(v,t):
    return -np.sum(t*np.log(v/np.sum(v)))