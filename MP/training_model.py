import pickle
import pandas as pd
import sys
import os
import json
from pathlib import Path
from MP.training_mp import Config, parser, MultiLayerPerceptron
from MP.math_utils.out_layer import binary_to_one_hot
from MP.math_utils.optimizers import sgd, adam


MP_PATH = Path(__file__).resolve().parent


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


def load_dataset(path):
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"The file {path.name} does not contain a pandas dataframe")
    except TypeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error opening data from {path.name}. Make sure you have executed split_dataset.py")
        sys.exit(1)
    return df


def separate_and_normalize(train_df, val_df, data_name):
    def save_scaler(mean, std):
        os.makedirs(MP_PATH.parent / "scaler", exist_ok=True)
        with open(MP_PATH.parent / "scaler" / (Path(data_name).stem + "_scale_values.pkl"), "wb") as f:
            pickle.dump((mean, std), f)
    
    target_col = train_df.columns[0]

    # Separación Train
    train_target = train_df[target_col].to_numpy()
    train_data = train_df.drop(target_col, axis=1).to_numpy()

    # Separación Validation
    val_target = val_df[target_col].to_numpy()
    val_data = val_df.drop(target_col, axis=1).to_numpy()

    # Cálculo de métricas ÚNICAMENTE con los datos de Entrenamiento
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    
    # Protección contra datos sin variación
    std[std == 0] = 1

    # Escalamiento de ambos conjuntos con los mismos parámetros de entrenamiento
    train_data_normalized = (train_data - mean) / std
    val_data_normalized = (val_data - mean) / std

    # Guardar scaler del entrenamiento para el programa de testing posterior
    save_scaler(mean, std)

    return train_data_normalized, train_target, val_data_normalized, val_target


def set_optimizer(config):
    optimizers = {
        "sgd" : sgd,
        "adam": adam
    }

    optimizer = optimizers.get(config.optimizer_type)

    if optimizer is None:
        raise ValueError(f"The optimizer {config.optimizer_type} is not allowed")
    
    return optimizer(config)

def fit(config, mlp, optimizer, train_set, val_set):
    pass


if __name__ == "__main__":
    data_names, config_name = parser()
    train_path = MP_PATH.parent / "splitted_data" / data_names[0]
    val_path = MP_PATH.parent / "splitted_data" / data_names[1]
    config_path = MP_PATH.parent / "configs" / config_name if config_name else None
    
    config =  set_configuration(config_path)
    
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    X_train, y_train, X_val, y_val = separate_and_normalize(train_df, val_df, data_names[0])
    
    Y_train_oh = binary_to_one_hot(y_train)
    Y_val_oh = binary_to_one_hot(y_val)

    mlp = MultiLayerPerceptron(config, input_dim=X_train.shape[1])

    optimizer = set_optimizer(config)

    train_set = {"X": X_train, "y": y_train, "y_oh": Y_train_oh}
    val_set = {"X": X_val, "y": y_val, "y_oh": Y_val_oh}

    fit(config, mlp, optimizer, train_set, val_set)