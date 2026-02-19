import pickle
import pandas as pd
import sys
import numpy as np
import os


def load_training_data():
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
    training_df = load_training_data()
    X, y = separate_and_normalize(training_df)