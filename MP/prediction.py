import argparse
import textwrap
import pickle
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from MP.training_mp import MultiLayerPerceptron
from MP.math_utils import softmax

MP_PATH = Path(__file__).resolve().parent


def _parser():
    description = """\
    Program used to evaluate a model created by the training program using
    Binary Cross-Entropy over raw CSV files.

    It accepts:
    - model_name: The name of the trained model file inside the models 
                  directory (default: "breast_cancer_mlp").
    
    - data_name: The name of the raw .csv file located inside the '../test_data/' 
                 directory (default: "data.csv").
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="breast_cancer_mlp",
        metavar="",
        help="Name of the trained model file to load (default: 'breast_cancer_mlp')"
    )
    
    parser.add_argument(
        "-d", "--data_name",
        type=str,
        default="data.csv",
        metavar="",
        help="Name of the raw csv file to evaluate inside test_data folder (default: 'data.csv')"
    )

    args = parser.parse_args()
    
    return args.model_name.strip(), args.data_name.strip()


def _load_all(model_name, data_name):
    models_dir = MP_PATH.parent / "models"
    test_data_dir = MP_PATH.parent / "test_data"

    model_path = models_dir / f"{model_name}.pkl"
    try:
        with open(model_path, "rb") as f:
            mlp = pickle.load(f)
            
        if not isinstance(mlp, MultiLayerPerceptron):
            raise TypeError(f"The file '{model_path.name}' is not a valid MultiLayerPerceptron instance.")
            
    except FileNotFoundError:
        print(f"Error: Model file '{model_path.name}' not found inside '../models/'.")
        sys.exit(1)
    except TypeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not os.path.exists(test_data_dir):
        print(f"Error: The directory '{test_data_dir}' does not exist. Please create it next to the MP folder.")
        sys.exit(1)

    csv_path = test_data_dir / data_name
    try:
        raw_df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"Error: File '{data_name}' not found inside '../test_data/' directory.")
        sys.exit(1)

    return mlp, raw_df


def _process_and_scale_dataset(raw_df, scaler):
    if scaler is None:
        print("Error: The loaded model does not contain a valid scaler in its configuration.")
        sys.exit(1)

    # same cleaning as split dataset expecting same format
    df = raw_df.drop(raw_df.columns[0], axis=1)

    target = df[df.columns[0]].map({'M': 1, 'B': 0}).to_numpy()

    raw = df.drop(df.columns[0], axis=1).to_numpy()

    # Recuperar parámetros del scaler interno de la configuración del modelo
    mean, std = scaler

    # Re-escalar en caliente usando estrictamente los parámetros de entrenamiento
    scaled = (raw - mean) / std

    return scaled, target


def _binary_crossentropy(activ, target):
    num_samples = target.shape[0]
    
    probab_positive = activ[:, 1]
    probab_positive = np.clip(probab_positive, 1e-15, 1 - 1e-15)

    loss = - (1 / num_samples) * np.sum(target * np.log(probab_positive) + (1 - target) * np.log(1 - probab_positive))
    return loss

def _report(model_name, data_name, target, loss, accuracy):
    print("\n" + "=" * 55)
    print(f" EVALUATION METRICS FOR: {model_name}")
    print(f" Raw Source File: test_data/{data_name}")
    print(f" Number of samples processed: {target.shape[0]}")
    print("=" * 55)
    print(f" -> Binary Cross-Entropy Loss : {loss:.6f}")
    print(f" -> Evaluation Accuracy       : {accuracy * 100:.2f}%")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    model_name, data_name = _parser()

    mlp, raw_df = _load_all(model_name, data_name)

    features, target = _process_and_scale_dataset(raw_df, mlp.config.scaler)

    preactiv = mlp.forward(features)
    activ = softmax(preactiv)
    
    loss = _binary_crossentropy(activ, target)
    accuracy = MultiLayerPerceptron.compute_accuracy(target, preactiv)

    _report(model_name, data_name, target, loss, accuracy)
