import pickle
import pandas as pd
import sys


def load_test_data():
    try:
        with open("../splitted_data/test_data.pkl", "rb") as f:
            test_df = pickle.load(f)
            if not isinstance(test_df, pd.DataFrame):
                raise TypeError("The file doesn't contains a pandas dataframe")
    except TypeError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print("Error opening test data, Make sure you have splitted data.csv")
        sys.exit(1)
    return test_df

if __name__ == "__main__":
    test_df = load_test_data()
    print(test_df.head(10))