import pickle
import pandas as pd
import sys


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


if __name__ == "__main__":
    training_df = load_training_data()
    print(training_df.head(10))