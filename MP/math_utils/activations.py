import numpy as np


def relu(n):
    return np.maximum(0, n)

def relu_deriv(n):
    return np.where(n > 0, 1.0, 0.0)

def sigmoid(n):
    return np.where(
        n >= 0,
        1 / (1 + np.exp(-n)),
        #protection against overflow
        np.exp(n) / (1 + np.exp(n))
    )

def sigmoid_deriv(n):
    s = sigmoid(n)
    return s * (1 - s)

def tan_h(n):
    return np.tanh(n)

def tan_h_deriv(n):
    t = np.tanh(n)
    return 1 - t**2
