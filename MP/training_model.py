import pickle
import pandas as pd
import sys
import numpy as np
import os
import json
from pathlib import Path
from trainingMP import Config, parser


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


def load_training_data(path):
    try:
        with open(path, "rb") as f:
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


def save_scaler(mean, std, data_name):
    os.makedirs(MP_PATH.parent / "scaler", exist_ok=True)
    with open(MP_PATH.parent / "scaler" / (Path(data_name).stem + "_scale_values.pkl"), "wb") as f:
        pickle.dump((mean, std), f)


def separate_and_normalize(df, data_name):
    target_col = df.columns[0]

    y = df[target_col].to_numpy()
    X = df.drop(target_col, axis=1).to_numpy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1

    X = (X - mean) / std

    # save scaler values to avoid data leakage in test:
    save_scaler(mean, std, data_name)

    return X, y


if __name__ == "__main__":
    data_name, config_name = parser()
    data_path = MP_PATH.parent / "splitted_data" / data_name
    config_path = MP_PATH.parent / "configs" / config_name if config_name else None
    print(set_configuration(config_path))
    training_df = load_training_data(data_path)
    X, y = separate_and_normalize(training_df, data_name)
    # validar la configuracion si es correcta se crea el objeto. 
    # se debe crear clase layer, protegiendo:
    #     - number of features in imput
    #     - softmax for output
    #     - setteo standar en caso de error