import pickle
import pandas as pd
import sys
import numpy as np
import os
import argparse
import textwrap
import json


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

        # === VALIDACIÓN DE STRINGS (OPCIONES PERMITIDAS) ===
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

        # === VALIDACIÓN DE LÍMITES NUMÉRICOS (ANTI-OVERFLOW) ===
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
                # Una paciencia sana por defecto es el 10% de las épocas, o mínimo 10 VER
                default_patience = min(10, max(1, int(self.epochs * 0.1)))
                print(f"Warning: Patience {self.early_stopping_patience} invalid for {self.epochs} epochs. Falling back to {default_patience}.")
                self.early_stopping_patience = default_patience

        # === VALIDACIÓN DE CADA CAPA OCULTA YA LIMPIA ===
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


#a factorizar
def parser():
    description = """\
    This program trains a Multilayer Perceptron (MLP) using a previously
    prepared training dataset generated by the split_data.py script.

    The program accepts two optional arguments:

    1. The name to the .pkl file containing the training dataset.
    2. The name to the .json configuration file describing the neural
       network architecture and training parameters.

    If no dataset path is provided, the program defaults to training_data.pkl.
    All training data must be located inside the "../splitted_data/" directory
    relative to the script's location.

    If no configuration file is provided, or if specific parameters within 
    the file are missing or invalid, the program intelligently falls back to 
    safe default values on a parameter-by-parameter basis, ensuring the 
    training can always execute.

    The absolute global defaults (used if the JSON is empty or missing) are:
    - Input Layer Activation: sigmoid
    - Hidden Layers: 2 layers with 24 neurons each, using sigmoid and heUniform
    - Output Layer Initializer: heUniform
    - Training: 100 epochs, batch_size 16, learning_rate 0.01, loss categorical_crossentropy
    - Optimizer: sgd (Adam defaults: beta1=0.9, beta2=0.999)
    - Early Stopping: enabled, patience 10, monitor val_loss

    All configuration files must be located inside the ../configs/ directory
    relative to the script's location.

    The input layer size is automatically determined by the number of
    features in the dataset (30 features excluding the target variable in this 
    project). The design explicitly allows customization of both the input and
    output layers: the input layer supports configurable activation functions,
    while the output layer supports configurable initialization, while always
    using a fixed softmax activation over two neurons for binary classification.

    Configuration file example:

    {
      "model_name": "breast_cancer_mlp",
      "topology": {
        "input_layer_activation": "sigmoid",
        "hidden_layers": [
          { "n_neurons": 32, "activation": "relu", "initializer": "heUniform" },
          { "n_neurons": 24, "activation": "tanh", "initializer": "xavier" },
          { "n_neurons": 16, "activation": "sigmoid", "initializer": "heNormal" }
        ],
        "output_layer_initializer": "heUniform"
      },
      "training": {
        "epochs": 150,
        "batch_size": 16,
        "learning_rate": 0.01,
        "loss": "categorical_crossentropy",
        "optimizer": {
          "type": "adam",
          "params": { "beta1": 0.9, "beta2": 0.999 }
        }
      },
      "early_stopping": {
        "enabled": true,
        "patience": 10,
        "monitor": "val_loss"
      }
    }

    Valid configuration values & Fallback constraints:

    Activation functions (Input & Hidden):
    ["relu", "sigmoid", "tanh"] -> Invalid values fallback to "sigmoid"

    Initializers (Hidden & Output):
    ["heUniform", "heNormal", "xavier", "random"] -> Invalid values fallback to "heUniform"

    Loss functions:
    ["categorical_crossentropy", "mse"] -> Invalid values fallback to "categorical_crossentropy"

    Optimizers:
    ["sgd", "adam"] -> Invalid values fallback to "sgd"

    Adam parameters:
    beta1 ∈ [0.0, 1.0) -> Out of bounds falls back to 0.9
    beta2 ∈ [0.0, 1.0) -> Out of bounds falls back to 0.999

    Topology constraints:
    - If "hidden_layers" is missing or malformed, it falls back to the default 
      architecture (2 layers of 24 neurons).
    - If provided, each hidden layer must define n_neurons ∈ (0, 256]. Values 
      outside this range will trigger a warning and fallback to 24 neurons.

    Training parameters:
    epochs ∈ (0, 1000] -> Out of bounds falls back to 100
    batch_size ≥ 1 and ≤ dataset size -> Out of bounds falls back to 16
    learning rate > 0.0 (recommended ≤ 0.1) -> Out of bounds falls back to 0.01

    Early stopping:
    enabled: boolean -> Invalid types fallback to true
    patience ≥ 1 and < epochs -> Out of bounds falls back to 10
    monitor: ["val_loss", "val_accuracy"] -> Invalid values fallback to "val_loss"
    """
    parser = argparse.ArgumentParser(
        description = textwrap.dedent(description),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="training_data.pkl",
        metavar="",
        help="name of trainning datafile (default: training_data.pkl)"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        metavar="",
        help="name of configuration file (default: None)"
    )

    args = parser.parse_args()
    return "../splitted_data/"+args.data_name, "../configs/"+ args.config_name 


def set_configuration(path):
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                conf = json.load(f)
                return Config(conf)
        except Exception as e:
            print(f"Error reading config file: {e}\nSetting as default...")
            return Config()
    else:
        return Config()


def load_training_data(path):
    try:
        with open("../splitted_data/training_data.pkl", "rb") as f:
            training_df = pickle.load(f)
            if not isinstance(training_df, pd.DataFrame):
                raise TypeError("The file doesn't contains a pandas dataframe")
    except TypeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print("Error opening training data, Make sure you have splitted data.csv")
        sys.exit(1)
    return training_df


def save_scaler(mean, std):
    os.makedirs("../scaler", exist_ok=True)
    with open("../scaler/scale_values.pkl", "wb") as f:
        pickle.dump((mean, std), f)


def separate_and_normalize(df):
    target_col = df.columns[0]

    y = df[target_col].to_numpy()
    X = df.drop(target_col, axis=1).to_numpy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1

    X = (X - mean) / std

    # save scaler values to avoid data leakage in test:
    save_scaler(mean, std)

    return X, y


if __name__ == "__main__":
    data_path, config_path = parser()
    set_configuration(config_path)
    training_df = load_training_data(data_path)
    X, y = separate_and_normalize(training_df)
    # validar la configuracion si es correcta se crea el objeto. 
    # se debe crear clase layer, protegiendo:
    #     - number of features in imput
    #     - softmax for output
    #     - setteo standar en caso de error