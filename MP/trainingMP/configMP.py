class Config:
    def __init__(self, json_data=None):
        if json_data is None:
            json_data = {}
        
        self.model_name = json_data.get("model_name", "breast_cancer_mlp")
        
        topology_cfg = json_data.get("topology", {})
        training_cfg = json_data.get("training", {})
        early_cfg = json_data.get("early_stopping", {})
        
        self.input_layer_activation = topology_cfg.get("input_layer_activation", "sigmoid")
        self.output_layer_initializer = topology_cfg.get("output_layer_initializer", "heUniform")
        
        self.hidden_layers = topology_cfg.get("hidden_layers", [
            {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"},
            {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"}
        ])

        if not isinstance(self.hidden_layers, list) or len(self.hidden_layers) < 2:
            self.hidden_layers = [
                {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"},
                {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"}
            ]

        cleaned_layers = []
        for layer in self.hidden_layers:
            if not isinstance(layer, dict):
                continue 
            
            clean_layer = {
                "n_neurons": int(layer.get("n_neurons", 24)),
                "activation": str(layer.get("activation", "sigmoid")),
                "initializer": str(layer.get("initializer", "heUniform"))
            }
            cleaned_layers.append(clean_layer)

        if len(cleaned_layers) < 2:
            cleaned_layers = [
                {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"},
                {"n_neurons": 24, "activation": "sigmoid", "initializer": "heUniform"}
            ]
        
        self.hidden_layers = cleaned_layers
        
        self.epochs = int(training_cfg.get("epochs", 100))
        self.batch_size = int(training_cfg.get("batch_size", 16))
        self.learning_rate = float(training_cfg.get("learning_rate", 0.01))
        self.loss = training_cfg.get("loss", "categorical_crossentropy")
        
        optimizer_cfg = training_cfg.get("optimizer", {})
        self.optimizer_type = optimizer_cfg.get("type", "sgd")
        
        if self.optimizer_type == "adam":
            opt_params = optimizer_cfg.get("params", {})
            self.adam_beta1 = float(opt_params.get("beta1", 0.9))
            self.adam_beta2 = float(opt_params.get("beta2", 0.999))
        
        self.early_stopping_enabled = bool(early_cfg.get("enabled", True))
        self.early_stopping_patience = int(early_cfg.get("patience", 10))
        self.early_stopping_monitor = early_cfg.get("monitor", "val_loss")

        # === VALIDATION OF STRINGS (ALLOWED OPTONS) ===
        valid_activations = ["relu", "sigmoid", "tanh"]
        valid_initializers = ["heUniform", "heNormal", "xavier", "random"]
        
        if self.input_layer_activation not in valid_activations:
            print(f"Warning: Invalid input activation '{self.input_layer_activation}'. Falling back to 'sigmoid'.")
            self.input_layer_activation = "sigmoid"
            
        if self.output_layer_initializer not in valid_initializers:
            print(f"Warning: Invalid output initializer '{self.output_layer_initializer}'. Falling back to 'heUniform'.")
            self.output_layer_initializer = "heUniform"

        if self.loss not in ["categorical_crossentropy", "mse"]:
            print(f"Warning: Invalid loss '{self.loss}'. Falling back to 'categorical_crossentropy'.")
            self.loss = "categorical_crossentropy"

        if self.optimizer_type not in ["sgd", "adam"]:
            print(f"Warning: Invalid optimizer '{self.optimizer_type}'. Falling back to 'sgd'.")
            self.optimizer_type = "sgd"

        if self.early_stopping_monitor not in ["val_loss", "val_accuracy"]:
            print(f"Warning: Invalid monitor '{self.early_stopping_monitor}'. Falling back to 'val_loss'.")
            self.early_stopping_monitor = "val_loss"

        # === NUMERIC LIMITS VALIDATION ===
        if self.epochs <= 0 or self.epochs > 1000:
            print(f"Warning: Epochs {self.epochs} out of bounds (0, 1000]. Falling back to 100.")
            self.epochs = 100

        if self.batch_size < 1:
            print(f"Warning: Batch size {self.batch_size} must be >= 1. Falling back to 16.")
            self.batch_size = 16

        if self.learning_rate <= 0.0 or self.learning_rate > 0.1:
            print(f"Warning: Learning rate {self.learning_rate} dangerous or invalid. Falling back to 0.01.")
            self.learning_rate = 0.01

        if self.optimizer_type == "adam":
            if not (0.0 <= self.adam_beta1 < 1.0):
                print(f"Warning: Beta1 {self.adam_beta1} out of bounds [0, 1). Falling back to 0.9.")
                self.adam_beta1 = 0.9
            if not (0.0 <= self.adam_beta2 < 1.0):
                print(f"Warning: Beta2 {self.adam_beta2} out of bounds [0, 1). Falling back to 0.999.")
                self.adam_beta2 = 0.999

        if self.early_stopping_enabled:
            if self.early_stopping_patience < 1 or self.early_stopping_patience >= self.epochs:
                # A healthly patience is often 10% of the total epochs
                default_patience = min(10, max(1, int(self.epochs * 0.1)))
                print(f"Warning: Patience {self.early_stopping_patience} invalid for {self.epochs} epochs. Falling back to {default_patience}.")
                self.early_stopping_patience = default_patience

        # === CLEANED HIIDEN LAYERS VALIDATION ===
        for i, layer in enumerate(self.hidden_layers):
            if layer["n_neurons"] <= 0 or layer["n_neurons"] > 256:
                print(f"Warning: Layer {i} neurons {layer['n_neurons']} out of bounds (0, 256]. Falling back to 24.")
                layer["n_neurons"] = 24
                
            if layer["activation"] not in valid_activations:
                print(f"Warning: Layer {i} invalid activation '{layer['activation']}'. Falling back to 'sigmoid'.")
                layer["activation"] = "sigmoid"
                
            if layer["initializer"] not in valid_initializers:
                print(f"Warning: Layer {i} invalid initializer '{layer['initializer']}'. Falling back to 'heUniform'.")
                layer["initializer"] = "heUniform"

    def __str__(self):
        """Genera una representación en string limpia y organizada de la configuración actual."""
        lines = []
        lines.append("=" * 50)
        lines.append(f" CONFIGURATION: {self.model_name.upper()} ")
        lines.append("=" * 50)
        
        # 1. Topología
        lines.append("\n[TOPOLOGY]")
        lines.append(f"  Input Layer Activation:  {self.input_layer_activation}")
        lines.append("  Hidden Layers Architecture:")
        for i, layer in enumerate(self.hidden_layers):
            lines.append(f"    - Layer {i+1}: {layer['n_neurons']} neurons | "
                         f"Activation: {layer['activation']} | "
                         f"Initializer: {layer['initializer']}")
        lines.append(f"  Output Layer Initializer: {self.output_layer_initializer}")
        lines.append("  Output Layer Activation:  softmax (FIXED)")
        
        # 2. Entrenamiento General
        lines.append("\n[TRAINING PARAMETERS]")
        lines.append(f"  Epochs:                 {self.epochs}")
        lines.append(f"  Batch Size:             {self.batch_size}")
        lines.append(f"  Learning Rate:          {self.learning_rate}")
        lines.append(f"  Loss Function:          {self.loss}")
        
        # 3. Optimizador Dinámico (Aquí ocurre la magia que pides)
        lines.append(f"  Optimizer Type:         {self.optimizer_type.upper()}")
        if self.optimizer_type == "adam":
            lines.append(f"    -> Adam Beta 1:       {self.adam_beta1}")
            lines.append(f"    -> Adam Beta 2:       {self.adam_beta2}")
            
        # 4. Early Stopping
        lines.append("\n[EARLY STOPPING]")
        lines.append(f"  Enabled:                {self.early_stopping_enabled}")
        if self.early_stopping_enabled:
            lines.append(f"  Patience:               {self.early_stopping_patience} epochs")
            lines.append(f"  Monitor Metric:         {self.early_stopping_monitor}")
            
        lines.append("=" * 50)
        return "\n".join(lines)