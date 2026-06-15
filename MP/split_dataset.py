import pandas as pd
import numpy as np
import argparse
import textwrap
import os
import pickle
import sys
from pathlib import Path
from MP.math_utils.out_layer import binary_to_one_hot

MP_PATH = Path(__file__).resolve().parent


def _parser():
    description = """\
        This program reads a raw dataset file (by default 'data.csv') and splits 
        a portion of the data into a training set according to the 'training_rate'
        percentage. The split preserves the class distribution in the dataset 
        (stratification).

        The script cleans the data, isolates the scaling parameters (scaler) 
        using ONLY the training set to prevent data leakage, and exports 
        ready-to-use matrices (including targets and One-Hot representations) 
        into a dedicated experiment directory.

        By using the '--raw_name' (-rn) flag, you can specify any other CSV file 
        located next to the MP folder, allowing you to easily process different 
        sources, filtered subsets, or variations of the dataset.

        Additionally, it can perform an Initial Exploratory Data Analysis (EDA)
        with statistical summaries and visual graphs of the processed dataset.

        Example usage:

            python split_dataset.py --raw_name data.csv --training_rate 80 --seed 42 --name my_experiment

        - The file 'data.csv' will be loaded and processed.
        - 80%% of the samples will go to the training set, 20% to the validation set.
        - Providing a specific 'seed' ensures reproducibility.
        - All outputs will be stored inside '../splitted_data/my_experiment/'.
        - The --explore (-e) flag will trigger the data visualization and summary.
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
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="default",
        metavar="",
        help="Name of the experiment folder to save the splitted data (default: 'default')"
    )
    parser.add_argument(
        "-e", "--explore",
        action="store_true",
        help="Trigger initial exploratory data analysis (EDA) and render charts (default: False)"
    )
    parser.add_argument(
        "--raw_name", "-rn",
        type=str,
        default="data.csv",
        help="Name of the raw CSV file located next to the MP folder (default: data.csv)"
    )

    args = parser.parse_args()
    
    return args.training_rate / 100, args.seed, args.name, args.explore, args.raw_name


def _split_by_rate(df, target_col, training_rate, seed=None):
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


def _save_formated_data(name, scaler, train_val_set):
    experiment_dir = MP_PATH.parent / "splitted_data" / name
    os.makedirs(experiment_dir, exist_ok=True)

    with open(experiment_dir / f"scaler_{name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(experiment_dir / f"train_val_{name}.pkl", "wb") as f:
        pickle.dump(train_val_set, f)

    print(f"> Capsule '{name}' successfully generated inside '../splitted_data/{name}/'")


def _format_and_save(df, training_df, validation_df, name):
    target_col = df.columns[0]

    # Separation Train
    train_target = training_df[target_col].to_numpy()
    train_target_oh = binary_to_one_hot(train_target)
    train_data = training_df.drop(target_col, axis=1).to_numpy()

    # Separation Validation
    val_target = validation_df[target_col].to_numpy()
    val_target_oh = binary_to_one_hot(val_target)
    val_data = validation_df.drop(target_col, axis=1).to_numpy()

    # Defining scaler
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    
    # Protection agains no variable data
    std[std == 0] = 1
    
    scaler = (mean, std)

    # Scalign 
    train_data_normalized = (train_data - mean) / std
    val_data_normalized = (val_data - mean) / std

    train_val_set = {
        "train" : {
            "X" : train_data_normalized,
            "y" : train_target,
            "y_oh" : train_target_oh
        },
        "val" : {
            "X" : val_data_normalized,
            "y" : val_target,
            "y_oh" : val_target_oh
        }
    }
    _save_formated_data(name, scaler, train_val_set)

    return train_val_set


def _report_init_data(formated_data, raw_df):
    pass


if __name__ == "__main__":
    # Get parameters
    training_rate, seed, name, explore, raw_name = _parser()

    # Read csv
    try:
        raw_df = pd.read_csv(MP_PATH.parent / raw_name, header=None)
    except FileNotFoundError:
        print(f"Put {raw_name} file next to MP folder")
        sys.exit(1)

    # Drop ID column (no important information provided):
    df = raw_df.drop(raw_df.columns[0], axis=1)

    # Convert the target variable from categorical to binary B = 0 & M = 1:
    df[df.columns[0]] = df[df.columns[0]].map({'M': 1, 'B': 0})

    # Split data according to training_rate, taking care to save the proportion
    # in the target classes:
    training_df, validation_df = _split_by_rate(df, df.columns[0], training_rate, seed)

    formated_data = _format_and_save(df, training_df, validation_df, name)

    if explore:
        _report_init_data(formated_data, raw_df)


