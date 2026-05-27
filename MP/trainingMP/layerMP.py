import numpy as np

class Layer:
    def __init__(self, n_in, n_out, activation="relu", initializer="heUniform"):
        self.activation_name = activation
        
        # 1. Weight (W) and Bias (b) Initialization
        self.W, self.b = self._initialize_weights(n_in, n_out, initializer)
        
        # 2. Cache Space (essential for Backpropagation)
        self.A_in = None  # Activation from the previous layer (or input data X)
        self.Z = None     # Weighted sum (Z = A_in * W + b)
        self.A = None     # Activation of this layer (A = f(Z))
        
        # 3. Gradient Space (computed during the backward pass)
        self.dW = None
        self.db = None
        
        # 4. Bonus Space (history for the Adam optimizer)
        # They remain None for now; the optimizer will initialize them with zeros if needed
        self.m_W, self.v_W = None, None
        self.m_b, self.v_b = None, None

    def _initialize_weights(self, n_in, n_out, initializer):
        """
        Initializes the weight and bias matrices depending on the selected method.
        W has dimensions (n_in, n_out) to work with matrices where rows
        represent samples: Z = A_in . W + b
        """
        # Bias is usually initialized with zeros or very small values
        b = np.zeros((1, n_out))
        
        if initializer == "heUniform":
            # He Uniform / Kaiming Uniform
            limit = np.sqrt(6.0 / n_in)
            W = np.random.uniform(-limit, limit, size=(n_in, n_out))
            
        elif initializer == "heNormal":
            # He Normal / Kaiming Normal
            std = np.sqrt(2.0 / n_in)
            W = np.random.normal(0.0, std, size=(n_in, n_out))
            
        elif initializer == "xavier":
            # Xavier / Glorot Uniform (ideal for tanh and sigmoid)
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, size=(n_in, n_out))
            
        elif initializer == "random":
            # Simple scaled random initialization
            W = np.random.randn(n_in, n_out) * 0.01
            
        else:
            # Safety fallback
            limit = np.sqrt(6.0 / n_in)
            W = np.random.uniform(-limit, limit, size=(n_in, n_out))
            
        return W, b