import pandas as pd

if __name__ == "__main__":
    test_proportion = 7.5

    # Read csv
    df = pd.read_csv("../data.csv", header=None)

    # Drop ID column (no important information provided):
    df = df.drop(df.columns[0], axis=1)

    # Convert the target variable from categorical to binary B = 0 & M = 1:
    df[df.columns[0]] = df[df.columns[0]].map({'M': 1, 'B': 0})

    # My stratify: 
    print(df.head(30))