import numpy as np

class Optimizer:
    def __init__(self, config):
        self.lr = config.learning_rate

    def step(self, layers):
        raise NotImplementedError("Every optimizer must implement its own 'step' method.")


class sgd(Optimizer):
    def __init__(self, config):
        super().__init__(config)

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


class adam(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.t = 0
        self.epsilon = 1e-8
        self.beta1 = config.adam_beta1
        self.beta2 = config.adam_beta2
        self.initialized = False

    def initialize(self, layers):
        #TO do once
        for layer in layers:
            layer.m_W = np.zeros_like(layer.W)
            layer.v_W = np.zeros_like(layer.W)
            layer.m_b = np.zeros_like(layer.b)
            layer.v_b = np.zeros_like(layer.b)
        self.initialized = True

    def step(self, layers):
        if not self.initialized:
            self.initialize(layers)
            
        self.t += 1
        
        for layer in layers:
            #Update W:

            layer.m_W = self.beta1 * layer.m_W + (1 - self.beta1) * layer.dW
            layer.v_W = self.beta2 * layer.v_W + (1 - self.beta2) * (layer.dW ** 2)
            
            m_W_corrected = layer.m_W / (1 - self.beta1 ** self.t)
            v_W_corrected = layer.v_W / (1 - self.beta2 ** self.t)
            
            layer.W -= (self.lr / (np.sqrt(v_W_corrected) + self.epsilon)) * m_W_corrected
            
            #Update b:

            layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.db
            layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.db ** 2)
            
            m_b_corrected = layer.m_b / (1 - self.beta1 ** self.t)
            v_b_corrected = layer.v_b / (1 - self.beta2 ** self.t)
            
            layer.b -= (self.lr / (np.sqrt(v_b_corrected) + self.epsilon)) * m_b_corrected

    #just in case...
    def reset(self):
        self.t = 0
        self.initialized = False