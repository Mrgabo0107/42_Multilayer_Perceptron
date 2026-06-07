import numpy as np

def softmax(v):
    shift = v - np.max(v, axis=-1, keepdims=True)
    e = np.exp(shift)
    return e / np.sum(e, axis=-1, keepdims=True)

def softmax_cross_entrop(target, pre_activ):
    s = softmax(pre_activ)
    return s - target

def softmax_mse(target, pre_activ):
    s = softmax(pre_activ)
    grd_loss_s = s - target
    sum_dot = np.sum(grd_loss_s * s, axis=-1, keepdims=True)
    return s * (grd_loss_s - sum_dot)

def cross_entropy(target, s):
    # avoid Nan in extreme values (log function)
    s_clipped = np.clip(s, 1e-15, 1.0 - 1e-15)
    return -np.sum(target * np.log(s_clipped)) / target.shape[0]

def mse(target, s):
    return 0.5 * np.sum((s - target) ** 2) / target.shape[0]

def binary_to_one_hot(y_binary):
    num_clases = 2
    return(np.eye(num_clases)[y_binary])
