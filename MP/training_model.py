import pickle
import sys
import os
import json
import copy
import numpy as np
from pathlib import Path
from MP.training_mp import Config, parser, MultiLayerPerceptron
from MP.math_utils import sgd, adam


MP_PATH = Path(__file__).resolve().parent


def _set_configuration(path):
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


def _load_dataset(data_path):
    try:
        with open(data_path, "rb") as f:
            data_capsule = pickle.load(f)
    except FileNotFoundError:
        print(f"Error opening data from {data_path}. Make sure you have executed split_dataset.py")
        sys.exit(1)

    return (data_capsule["train"], data_capsule["val"])


def _load_scaler(scaler_path):
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"Error opening scaler from {scaler_path}. Make sure you have executed split_dataset.py")
        sys.exit(1)

    return scaler


def _set_optimizer(config):
    optimizers = {
        "sgd" : sgd,
        "adam": adam
    }

    optimizer = optimizers.get(config.optimizer_type)

    if optimizer is None:
        raise ValueError(f"The optimizer {config.optimizer_type} is not allowed")
    
    return optimizer(config)


def _fit(config, mlp, optimizer, data):
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
    
    train_set, val_set = data
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
    return historic, mlp


def _export_trained(historic, model, config):
    model_dir = MP_PATH.parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = model_dir / f"{config.model_name}.pkl"
    historic_path = model_dir / f"historic_{config.model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(historic_path, "wb") as f:
        pickle.dump(historic, f)

if __name__ == "__main__":
    data_name, config_name = parser()
    capsule_path = MP_PATH.parent / "splitted_data" / data_name / f"train_val_{data_name}.pkl"
    scaler_path = MP_PATH.parent / "splitted_data" / data_name / f"scaler_{data_name}.pkl"
    config_path = MP_PATH.parent / "configs" / config_name if config_name else None
    
    data = _load_dataset(capsule_path)

    config =  _set_configuration(config_path)
    config.scaler = _load_scaler(scaler_path)

    mlp = MultiLayerPerceptron(config, data[0]["X"].shape[1])

    optimizer = _set_optimizer(config)

    historic, trained = _fit(config, mlp, optimizer, data)

    _export_trained(historic, trained, config)