import numpy as np
from MP.training_mp.layer_mp import Layer
from MP.math_utils.out_layer import binary_to_one_hot, softmax_crossentropy, softmax_mse, mse, c_crossentropy

class MultiLayerPerceptron:
    def __init__(self, config, data_tuple):
        self.config = config
        self.layers = []
        
        # Desempaquetamos de forma limpia la tupla que viene del main
        self.X_train, self.y_train, self.X_val, self.y_val = data_tuple

        self.Y_train_oh = binary_to_one_hot(self.y_train)
        self.Y_val_oh = binary_to_one_hot(self.y_val)
        
        # 1. Input features are passed directly to the first hidden layer.
        # There is no separate input-layer activation or initializer here.
        # The first hidden layer receives the raw input and applies its own weights.
        
        current_dim = self.X_train.shape[1]  # Number of features in the incoming dataset
        
        # Build hidden layers by iterating through the JSON configuration
        for layer_cfg in self.config.hidden_layers:
            self.layers.append(
                Layer(
                    n_in=current_dim,
                    n_out=layer_cfg["n_neurons"],
                    activation=layer_cfg["activation"],
                    initializer=layer_cfg["initializer"]
                )
            )
            current_dim = layer_cfg["n_neurons"]  # The output of this layer becomes the input of the next one
            
        # Output layer fixed to 2 neurons with Softmax for probabilistic binary classification
        self.layers.append(
            Layer(
                n_in=current_dim,
                n_out=2,
                activation="softmax",
                initializer=self.config.output_layer_initializer
            )
        )
        self.num_layers = len(self.layers)

    def __str__(self):
        divider = "=" * 90
        sub_divider = "-" * 90
        
        res = [
            divider,
            f" MULTILAYER PERCEPTRON - GLOBAL MODEL CONFIGURATION",
            divider
        ]
        
        # 1. Inspección del objeto Config pasados al MLP
        if hasattr(self, 'config') or 'config' in globals() or self.layers:
            # Intentamos extraer de manera segura las variables comunes de tu JSON/Config
            lr = getattr(self.config, 'learning_rate', 'N/A') if hasattr(self, 'config') else 'N/A'
            epochs = getattr(self.config, 'epochs', 'N/A') if hasattr(self, 'config') else 'N/A'
            batch_size = getattr(self.config, 'batch_size', 'N/A') if hasattr(self, 'config') else 'N/A'
            
            res.append(f"  » Learning Rate: {lr}")
            res.append(f"  » Epochs:        {epochs}")
            res.append(f"  » Batch Size:    {batch_size}")
            # Añadimos información útil sobre el tamaño de los datos en la configuración global
            res.append(f"  » Train Samples: {self.X_train.shape[0]}")
            res.append(f"  » Val Samples:   {self.X_val.shape[0]}")
        else:
            res.append("  » No config object metadata could be fetched.")
            
        res.append(divider)
        res.append(f" DETAILED LAYER ARCHITECTURE & INITIALIZED DATA")
        res.append(divider)
        
        # 2. Inspección detallada de cada capa
        for idx, layer in enumerate(self.layers):
            layer_type = f"HIDDEN LAYER [{idx + 1}]" if idx < len(self.layers) - 1 else "OUTPUT LAYER"
            res.append(f"● {layer_type}")
            res.append(str(layer))  # Llama al __str__ detallado de la capa superior
            res.append(sub_divider)
            
        res.append(f" Total Network Depth: {len(self.layers)} layers.")
        res.append(divider)
        
        return "\n".join(res)
    
    def forward(self, batch):
        self.layers[0].A_in = batch
        
        for i, layer in enumerate(self.layers):
            layer.Z = layer.A_in @ layer.W + layer.b
            
            if i < self.num_layers - 1:
                layer.A = layer.compute_activation()
                self.layers[i + 1].A_in = layer.A
        return self.layers[-1].Z
    
    # The parameters has to be general to allow shuffled targets
    def compute_output_gradient(self, target_batch_oh, output_preactiv):
        gradients = {
            "mse": softmax_mse,
            "categorical_crossentropy": softmax_crossentropy
        }

        method = gradients.get(self.config.loss)

        if method is None:
            raise ValueError(f"The method to find the output layer gradient '{self.config.loss}' is not allowed")

        # Operación directa, limpia y ultra-rápida sin conversiones internas
        return method(target_batch_oh, output_preactiv)
    
    def backward(self, output_gradient):
        dZ = output_gradient
        m = dZ.shape()[0]
        for i in range(self.num_layers - 1, -1, -1):
            layer = self.layers[i]
            layer.db = np.sum(dZ, axis=0, keepdims=True) / m
            layer.dW = (layer.A_in.T @ dZ) / m

            if i > 0:
                dA_in = dZ @ layer.W.T
                previous_layer = self.layers[i - 1]
                dZ = dA_in * previous_layer.compute_activation_derivative()
