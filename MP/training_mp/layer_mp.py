import numpy as np
from MP.math_utils.activations import relu, sigmoid, tan_h
from MP.math_utils.out_layer import softmax

SHOW_MATRICES_DEBUG = False

class Layer:
    def __init__(self, n_in, n_out, activation="relu", initializer="heUniform"):
        self.activation_name = activation
        self.initializer = initializer
        
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
    
    def __str__(self):
        w_shape = self.W.shape if self.W is not None else (0, 0)
        b_shape = self.b.shape if self.b is not None else (0, 0)
        
        # Extracción de métricas de los Pesos (W)
        if self.W is not None:
            w_stats = f"min:{self.W.min():.4f} | max:{self.W.max():.4f} | mean:{self.W.mean():.4f}"
            if SHOW_MATRICES_DEBUG:
                w_stats += "\n" + "\n".join(f"        {line}" for line in str(self.W).split("\n"))
        else:
            w_stats = "Uninitialized"
            
        # Extracción de métricas de los Bias (b)
        if self.b is not None:
            b_stats = f"min:{self.b.min():.4f} | max:{self.b.max():.4f} | mean:{self.b.mean():.4f}"
            if SHOW_MATRICES_DEBUG:
                 b_stats += f"\n        {str(self.b)}"
        else:
            b_stats = "Uninitialized"

        return (
            f"Layer Specs:\n"
            f"  |- Configuration: Activation = {self.activation_name} \n"
            f"  |- Weights (W):   Shape = {str(w_shape):<10} | Stats -> {w_stats}\n"
            f"  |- Biases (b):    Shape = {str(b_shape):<10} | Stats -> {b_stats}\n"
            f"  |- Initializer = {self.initializer}"
        )

    def compute_activation(self):
        activations = {
            "relu": relu,
            "sigmoid": sigmoid,
            "tan_h": tan_h,
            "softmax": softmax
        }

        function = activations.get(self.activation_name)

        if function is None:
            raise ValueError(f"activation name not supported: {self.activation_name}")

        return function(self.Z)