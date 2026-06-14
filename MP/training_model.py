import pickle
import sys
import os
import json
import copy
import numpy as np
import pandas as pd
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
    def _shuffle_set():
        return np.random.permutation(index)

    def _get_batch_set(batch_idx):
        return [train_set["X"][batch_idx], train_set["y_oh"][batch_idx]]
    
    def _report(epoch):
        print(f"epoch {epoch}/{config.epochs} \
              -- loss: {historic['train_loss'][-1]} \
              -- val_loss: {historic['val_loss'][-1]}")

        print('actualizando graficas')

    def _its_better(metric):
        return (config.early_stopping_monitor == 'val_loss' and historic['val_loss'][-1] < metric)\
        or (config.early_stopping_monitor == 'val_accuracy' and historic['val_accuracy'][-1] > metric)
    

    num_samples = train_set["X"].shape[0]
    index = np.arange(num_samples)
    patiente_counter = 0
    stopping_metric = None
    state = None
    historic = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": []
        }
    
    for epoch in range(config.epochs):
        shuffled = _shuffle_set()
        for start_batch in range(0, num_samples, config.batch_size):
            end_batch = min(start_batch + config.batch_size, num_samples)
            batch_set = _get_batch_set(shuffled[start_batch:end_batch])

            output_preactiv = mlp.forward(batch_set[0])
            loss_gradient = mlp.compute_output_gradient(batch_set[1], output_preactiv)
            mlp.backward(loss_gradient)
            optimizer.step(mlp.get_layers)

        # Graphics & report
        train_out_preactiv = mlp.forward(train_set["X"])
        val_out_preactiv = mlp.forward(val_set["X"])

        train_loss = mlp.compute_output_loss(train_set["y_oh"], train_out_preactiv)
        val_loss = mlp.compute_output_loss(val_set["y_oh"], val_out_preactiv)

        # accuracy (true postifs)
        train_acc = MultiLayerPerceptron.compute_accuracy(train_set["y"], train_out_preactiv)
        val_acc = MultiLayerPerceptron.compute_accuracy(val_set["y"], val_out_preactiv)

        historic["train_loss"].append(train_loss)
        historic["val_loss"].append(val_loss)
        historic["train_accuracy"].append(train_acc)
        historic["val_accuracy"].append(val_acc)

        _report(epoch)

        if config.early_stopping_enabled:
            if stopping_metric == None:
                stopping_metric = historic[config.early_stopping_monitor][-1]
                state = copy.deepcopy(mlp)
            elif _its_better(stopping_metric):
                patiente_counter = 0
                state = copy.deepcopy(mlp)
                stopping_metric = historic[config.early_stopping_monitor][-1]
            else:
                patiente_counter += 1
                if patiente_counter == config.early_stopping_patience:
                    return state
    return mlp


def _export_trained(model, config):
    model_dir = MP_PATH.parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = model_dir / f"{config.model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


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

    trained = fit(config, mlp, optimizer, train_set, val_set)

    _export_trained(trained, config)