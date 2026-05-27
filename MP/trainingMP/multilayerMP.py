from .layerMP import Layer

class MultiLayerPerceptron:
    def __init__(self, config, data):
        self.layers = []
        
        # 1. Input features are passed directly to the first hidden layer.
        # There is no separate input-layer activation or initializer here.
        # The first hidden layer receives the raw input and applies its own weights.
        
        current_dim = data.shape[1]  # Number of features in the breast cancer dataset
        print("aca", current_dim)
        # Build hidden layers by iterating through the JSON configuration
        for layer_cfg in config.hidden_layers:
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
                initializer=config.output_layer_initializer
            )
        )