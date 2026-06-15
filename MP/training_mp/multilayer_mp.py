import numpy as np
from MP.training_mp import Layer
from MP.math_utils import softmax, softmax_crossentropy, softmax_mse, mse, c_crossentropy

class MultiLayerPerceptron:
    def __init__(self, config, input_dim):
        self.config = config
        self.layers = []
        
        current_dim = input_dim
        
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
        
        # 1. Inspección segura de los hiperparámetros esenciales de configuración
        if hasattr(self, 'config') and self.config is not None:
            lr = getattr(self.config, 'learning_rate', 'N/A')
            epochs = getattr(self.config, 'epochs', 'N/A')
            batch_size = getattr(self.config, 'batch_size', 'N/A')
            loss_func = getattr(self.config, 'loss', 'N/A')
            opt_type = getattr(self.config, 'optimizer_type', 'SGD')
            
            res.append(f"  » Optimizer Type: {opt_type.upper()}")
            res.append(f"  » Learning Rate:  {lr}")
            res.append(f"  » Epochs:         {epochs}")
            res.append(f"  » Batch Size:     {batch_size}")
            res.append(f"  » Loss Function:  {loss_func}")
        else:
            res.append("  » No config object metadata could be fetched.")
            
        res.append(divider)
        res.append(f" DETAILED LAYER ARCHITECTURE & INITIALIZED DATA")
        res.append(divider)
        
        # 2. Inspección detallada de la topología de cada capa
        for idx, layer in enumerate(self.layers):
            layer_type = f"HIDDEN LAYER [{idx + 1}]" if idx < self.num_layers - 1 else "OUTPUT LAYER"
            res.append(f"● {layer_type}")
            res.append(str(layer))  # Llama al __str__ de tu clase Layer
            res.append(sub_divider)
            
        res.append(f" Total Network Depth: {self.num_layers} layers.")
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

    def compute_output_loss(self, target_oh, output_preactiv):
        activ = softmax(output_preactiv)

        losses = {
            "mse": mse,
            "categorical_crossentropy": c_crossentropy
        }

        method = losses.get(self.config.loss)

        if method is None:
            raise ValueError(f"The method to find loss: '{self.config.loss}' is not allowed")
        
        return method(target_oh, activ)

    @staticmethod
    def compute_accuracy(target, out_preactiv):
        predictions = np.argmax(out_preactiv, axis=1)
        return np.mean(predictions == target) * 100

    @property
    def get_layers(self):
        return self.layers
