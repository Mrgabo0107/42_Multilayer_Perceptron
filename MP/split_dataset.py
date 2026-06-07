import pandas as pd
import numpy as np
import argparse
import textwrap
import os
import pickle
import sys
from pathlib import Path

MP_PATH = Path(__file__).resolve().parent


def parser():
    description = """\
    This program reads the file 'data.csv' and splits a portion
    of the data into a training set according to the 'training_rate'
    percentage. The split preserves the class distribution in
    the dataset (stratification).

    Example usage:

        python split_dataset.py --training_rate 80 --seed 42

    - 80%% of the samples will go to the training set, 20% to the test set.
    - Providing a specific 'seed' ensures reproducibility, so that each
      time you run the script with the same seed, the same samples are
      selected for training and testing.
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--training_rate",
        type=float,
        default=75,
        metavar="",
        help="Percentage of data to use for training (default: 75%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="",
        help="Seed for randomness (default: None)"
    )

    args = parser.parse_args()
    return args.training_rate / 100, args.seed


def split_data(df, target_col, training_rate, seed=None):
    # If a seed is provided, the randomness in NumPy is fixed to extract
    # the same samples in training_data and test_data.
    if seed is not None:
        np.random.seed(seed)

    # Extract values in the column target as a numpy array
    y = df[target_col].to_numpy()

    # Get the diferent classes in a target column
    classes = np.unique(y)

    # Init the two list of index for split
    train_indx = []
    test_indx = []

    for c in classes:
        idx = np.where(y == c)[0]  # index for each class
        np.random.shuffle(idx)  # shuffle the index before add to a list
        # number of samples of a class that goes to training set
        n_train = int(len(idx) * training_rate)
        # Split the indices into their respective lists.
        train_indx.extend(idx[:n_train])
        test_indx.extend(idx[n_train:])

    # last shuffle to mix classes
    np.random.shuffle(train_indx)
    np.random.shuffle(test_indx)

    # create final dataframes:
    training_df = df.iloc[train_indx].reset_index(drop=True)
    test_df = df.iloc[test_indx].reset_index(drop=True)

    return training_df, test_df


def save_splitted_data(training_df, validation_df):
    os.makedirs(MP_PATH.parent / "splitted_data", exist_ok=True)

    with open(MP_PATH.parent / "splitted_data" / "training_data.pkl", "wb") as f:
        pickle.dump(training_df, f)

    # Ahora se guarda explícitamente como validation_data.pkl
    with open(MP_PATH.parent / "splitted_data" / "validation_data.pkl", "wb") as f:
        pickle.dump(validation_df, f)


if __name__ == "__main__":
    # Get parameters
    training_rate, seed = parser()

    # Read csv
    try:
        df = pd.read_csv(MP_PATH.parent / "data.csv", header=None)
    except FileNotFoundError:
        print("Put data.csv file next to MP folder")
        sys.exit(1)

    # Check for empty entries (I leave it commented out since the dataset
    # provided by the school has no empty values, and these lines only serves
    # to confirm that nothing has changed).
    # print(df.isna().sum())
    # print(df.isnull().sum())

    # Drop ID column (no important information provided):
    df = df.drop(df.columns[0], axis=1)

    # Convert the target variable from categorical to binary B = 0 & M = 1:
    df[df.columns[0]] = df[df.columns[0]].map({'M': 1, 'B': 0})

    # Split data according to training_rate, taking care to save the proportion
    # in the target classes:
    training_df, validation_df = split_data(df, df.columns[0], training_rate, seed)

    save_splitted_data(training_df, validation_df)

    print(training_df.head(10), "\n" , validation_df.head(10))
