from MP.training_mp.layer_mp import Layer

class MultiLayerPerceptron:
    def __init__(self, config, data):
        self.config = config
        self.layers = []
        
        # 1. Input features are passed directly to the first hidden layer.
        # There is no separate input-layer activation or initializer here.
        # The first hidden layer receives the raw input and applies its own weights.
        
        current_dim = data.shape[1]  # Number of features in the incoming dataset
        
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