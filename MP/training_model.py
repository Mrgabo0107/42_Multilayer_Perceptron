import pickle
import pandas as pd
import sys
import os
import json
from pathlib import Path
from MP.training_mp import Config, parser, MultiLayerPerceptron


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


def separate_and_normalize(df, data_name):
    def save_scaler(mean, std):
        os.makedirs(MP_PATH.parent / "scaler", exist_ok=True)
        with open(MP_PATH.parent / "scaler" / (Path(data_name).stem + "_scale_values.pkl"), "wb") as f:
            pickle.dump((mean, std), f)
    
    target_col = df.columns[0]

    target = df[target_col].to_numpy()
    data = df.drop(target_col, axis=1).to_numpy()

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    # protection againt data without variation
    std[std == 0] = 1

    data = (data - mean) / std

    # save scaler values to avoid data leakage in test:
    save_scaler(mean, std)

    return data, target


if __name__ == "__main__":
    data_name, config_name = parser()
    data_path = MP_PATH.parent / "splitted_data" / data_name
    config_path = MP_PATH.parent / "configs" / config_name if config_name else None
    training_df = load_training_data(data_path)
    normalized_data, target = separate_and_normalize(training_df, data_name)
    Model = MultiLayerPerceptron(set_configuration(config_path), normalized_data)
    print(f'aca {Model}')