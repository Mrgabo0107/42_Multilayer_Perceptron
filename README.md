# Multilayer Perceptron (MLP) — School 42

This project consists of a complete implementation from scratch of an Artificial Neural Network of type **Multilayer Perceptron (MLP)** for binary classification diagnosis of breast cancer (Malignant vs. Benign) using the *Wisconsin Breast Cancer* dataset.

The main objective is to build and understand the architecture of deep neural networks without relying on high-level machine learning libraries (such as PyTorch, TensorFlow, or Scikit-Learn), manually implementing the mathematical blocks for *Forward Propagation*, *Backpropagation*, optimization, and metrics evaluation.

---

## Features

* **Implementation From Scratch:** Matrix operations and vector calculation flows developed without Machine Learning libraries.
* **Modularity and Configurable Topology:** Dynamic support for defining architectures with multiple hidden layers, number of neurons, activation functions, and weight initialization via JSON configuration files.
* **Mathematical Weight Initializations:**
    * **Xavier/Glorot:** Preserves variance in layers with symmetric/linear activations (Sigmoid, Tanh).
    * **He/Kaiming:** Adjusts variance to compensate for the 50% deactivation of neurons caused by the ReLU function.


* **Advanced Optimizers:**
    * **SGD (Stochastic Gradient Descent):** Mini-batch stochastic gradient descent.
    * **Adam (Adaptive Moment Estimation):** Adaptive estimation of moments ($m_t$ and $v_t$) with bias correction.


* **Training Control Mechanisms:**
    * **Early Stopping:** Real-time monitoring of `val_loss` or `val_accuracy` with configurable *patience* to prevent overfitting and automatically restore the best model state.


* **Professional Preprocessing:** Data cleaning, One-Hot encoding, and scaler isolation ($\mu, \sigma$) using **only** the training dataset to prevent data leakage.
* **Performance Visualization:** Interactive plot generation for learning curves (*Loss* and *Accuracy* in Train vs. Validation).

---

## Project Structure and Setup (`setup.sh`)

The `setup.sh` script automatically prepares the entire executable workspace for the project.

### Execution of `setup.sh`

`bash setup.sh`

### Directories created by `setup.sh`

1. **`.venv/`**: Isolated Python virtual environment where dependencies (`pandas`, `numpy`, `plotly`) and the local `MP` package are installed in editable mode (`pip install -e .`).
2. **`configs/`**: Root directory intended to store the model training JSON configuration files.
3. **`test_data/`**: Root directory where raw `.csv` files are stored to be evaluated by the prediction script.

### CLI Commands Available After `setup.sh`

Upon activating the virtual environment (`source .venv/bin/activate`), Setuptools registers three executable commands directly in the terminal without needing to invoke them via `python path/to/script.py` syntax:

* **`mlp_split`**: Executes data partitioning, cleaning, and preparation.
* **`mlp_training`**: Initializes, trains, and exports the neural network.
* **`mlp_test`**: Evaluates a trained model against test data.

---

## Program Documentation and Flags

### 1. Data Partitioning Module (`mlp_split`)

Reads a raw CSV file, removes unnecessary identifiers, converts labels (`M` -> 1, `B` -> 0), and performs stratified sampling to preserve class proportions between training and validation sets. Additionally, it calculates normalization parameters (mean and standard deviation) based solely on the training partition.

**Generated Output:** Creates a directory inside `splitted_data/<experiment_name>/` containing Pickled files with scaled matrices, label vectors, and the scaler tuple.

#### Available Flags:

| Short Flag | Long Flag | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `-rn` | `--raw_name` | `str` | `"data.csv"` | Name of the raw CSV file located alongside the `MP` folder. |
|  | `--training_rate` | `float` | `75.0` | Percentage of data assigned to the training set. |
|  | `--seed` | `int` | `None` | Seed for the NumPy random generator to ensure reproducibility. |
| `-n` | `--name` | `str` | `"default"` | Name of the experiment directory inside `splitted_data/`. |
| `-e` | `--explore` | `flag` | `False` | Runs an Exploratory Data Analysis (EDA) and prints statistical summaries. |
| `-h` | `--help` | `flag` | — | Displays the descriptive help menu. |

#### Usage Example:

`mlp_split --raw_name data.csv --training_rate 80 --seed 42 -n exp_01 -e`

---

### 2. Training Module (`mlp_training`)

Loads the data capsule processed by `mlp_split`, constructs the Multilayer Perceptron according to the configuration file specifications, performs forward propagation (*Forward Pass*), calculates loss via *Categorical Cross-Entropy*, backpropagates errors (*Backpropagation*), and updates parameters using the selected optimizer.

**Generated Output:** Exports the trained model and metrics history inside `models/<model_name>/` as Pickle files, and displays graphic performance curves.


#### 2.1 Configuration File Structure (`JSON`)

Below is the supported JSON structure for defining training hyperparameters, optimizer type, network layers, and *Early Stopping* rules:

```json
{
  "model_name": "breast_cancer_mlp",
  "topology": {
    "hidden_layers": [
      { "n_neurons": 32, "activation": "relu", "initializer": "heUniform" },
      { "n_neurons": 12, "activation": "relu", "initializer": "heNormal" }
    ],
    "output_layer_initializer": "xavier"
  },
  "training": {
    "epochs": 150,
    "batch_size": 16,
    "learning_rate": 0.01,
    "loss": "categorical_crossentropy",
    "optimizer": {
      "type": "adam",
      "params": { "beta1": 0.9, "beta2": 0.999 }
    }
  },
  "early_stopping": {
    "enabled": true,
    "patience": 10,
    "monitor": "val_loss"
  }
}

```

#### Valid Configuration Values & Fallback Constraints

* **Activation functions (Hidden):** `["relu", "sigmoid", "tanh"]` $\rightarrow$ Invalid values fallback to `"relu"`.
* **Initializers (Hidden & Output):** `["heUniform", "heNormal", "xavier", "random"]` $\rightarrow$ Invalid values fallback to `"heUniform"`.
* **Loss functions:** `["categorical_crossentropy", "mse"]` $\rightarrow$ Invalid values fallback to `"categorical_crossentropy"`.
* **Optimizers:** `["sgd", "adam"]` $\rightarrow$ Invalid values fallback to `"sgd"`.
* **Adam parameters:**
    * `beta1` $\in [0.0, 1.0)$ $\rightarrow$ Out of bounds falls back to `0.9`.
    * `beta2` $\in [0.0, 1.0)$ $\rightarrow$ Out of bounds falls back to `0.999`.


* **Topology constraints:**
    * If `"hidden_layers"` is missing or malformed, it falls back to the default architecture (2 layers of 24 neurons).
    * If provided, each hidden layer must define `n_neurons` $\in (0, 256]$. Values outside this range will trigger a warning and fallback to 24 neurons.


* **Training parameters:**
    * `epochs` $\in (0, 1000]$ $\rightarrow$ Out of bounds falls back to `100`.
    * `batch_size` $\ge 1$ and $\le$ dataset size $\rightarrow$ Out of bounds falls back to `16`.
    * `learning_rate` $> 0.0$ (recommended $\le 0.1$) $\rightarrow$ Out of bounds falls back to `0.01`.


* **Early stopping:**
    * `enabled`: boolean $\rightarrow$ Invalid types fallback to `true`.
    * `patience` $\ge 1$ and $<$ `epochs` $\rightarrow$ Out of bounds falls back to `10`.
    * `monitor`: `["val_loss", "val_accuracy"]` $\rightarrow$ Invalid values fallback to `"val_loss"`.


#### 2.2 Positional Arguments:

1. **`data_name`**: Name of the experiment generated previously in `splitted_data/` (default: `"default"`).
2. **`config_name`**: Name of the JSON file located inside the `configs/` folder (default: `None`, loading standard configuration).

#### Available Flags:

| Short Flag | Long Flag | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `-c` | `--config_name` | `str` | `None` | Name of a valid configuration file (JSON) located in configs/ directory |
| `-d` | `--data_name` | `str` | `default` | Name of a data capsule calculated by mlp_split command in splitted_data/ directory |
| `-h` | `--help` | `flag` | — | Displays help for the training program. |

#### Usage Example:

`mlp_training -d splitted -c my_config.json`

---

### 3. Evaluation and Prediction Module (`mlp_test`)

Loads a previously trained model from the `models/` directory, accepts a raw `.csv` data file located in `test_data/`, processes and scales input data using the original metrics saved inside the model, performs batch predictions, and displays the performance report.

**Generated Output:** Informative report printed to the terminal showing *Binary Cross-Entropy* loss and evaluation accuracy percentage (*Accuracy*).

#### Available Flags:

| Short Flag | Long Flag | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `-m` | `--model_name` | `str` | `"breast_cancer_mlp"` | Name of the saved model directory inside `models/`. |
| `-d` | `--data_name` | `str` | `"data.csv"` | Name of the raw CSV file located inside `test_data/`. |
| `-h` | `--help` | `flag` | — | Displays help for the evaluation program. |

#### Usage Example:

`mlp_test -m my_mlp_model -d data.csv`

---

## Theoretical and Mathematical Foundations (Supplementary Material)

The project includes theoretical technical documentation in the `others/` directory aimed at project defense:

* **Weight Initialization (`activations_&_init_in_layers.pdf`):** Mathematical proof of the variance problem $Var(z) = n_{in} \cdot Var(W) \cdot Var(a)$ and derivation of uniform distribution limits based on activation type (Glorot vs. Kaiming).
* **Backpropagation Equations (`Backpropagation_fundamentals.pdf`):** Formal derivation of the four fundamental equations ($\delta^L$, $\delta^l$, $\frac{\partial L}{\partial b^l}$, $\frac{\partial L}{\partial W^l}$) and vectorization using outer products and Hadamard products ($\odot$).
* **Step-by-Step Calculation Trace (`forward_backward_adam_exemple.pdf`):** Numerical example verified with a mini step-by-step trace of forward pass, simplified error derivation, gradient projections, and Adam optimizer updates.

---
