# Configuration Reference Guide
This document explains all fields in **ANNConfig** (network architecture) and **TrainingConfig** (training procedure).  
It is intended as a clear, concise reference for users configuring models in the `ANN_pipeline` project.

---

## 1. ANNConfig (in `ann/model.py`)
'ANNConfig' defines the **architecture**, **activations**, **initializers**, **regularization**, **output head**, and **compile options** for the neural network.

---

### 1.1 Network Topology

#### `input_dim: int`
Number of input features.

---

#### `hidden_layers: List[int]`
Units for each hidden layer. Length = number of hidden layers.

**Examples:**
- `[64, 64]` → two layers of 64.
- `[32, 64, 128]` → three hidden layers.

---

### 1.2 Activations and Layer Options

#### `activations`
Activation(s) for hidden layers. Broadcast to all layers if a single value.
Accepts:
- String: `"relu"`, `"tanh`, `"gelu"`, `"silu"`, etc.
- Dict: `{"name": "leaky_relu", "alpha": 0.1}`, `{"name": "elu", "alpha": 1.0}`, `{"name": "prelu"}`
- Keras layer instance: `tf.keras.layers.LeakyReLU(alpha=0.1)`
- Callable: `tf.nn.gelu`
- List of the above for a per-layer setup

**Examples:**
- same activation:
 ```
 activations="relu"
 ```
- per layer:
 ```
 activations=[ "relu", "gelu", {"name": "leaky_relu", "alpha": 0.1} ]
 ```

---

#### `use_batchnorm: bool | Sequence[bool] = False`
Apply BatchNorm after Dense layers. Broadcastable.

**Examples:**
- all layers: `use_batchnorm=True`
- per-layer: `use_batchnorm=[False, True, True]`

---

#### `bn_momentum: float = 0.99`
BatchNorm momentum. Usually left at default (0.9–0.99 common).

---

#### `bn_epsilon: float = 1e-3`
BatchNorm numerical stability epsilon. Normally not changed.

---

#### `dropout_rates: float | Sequence[float] = 0.0`
Dropout rate(s) for hidden layers. Broadcastable.
Accepts:
- scalar: used for all hidden layers (unless overridden by split logic for last layer)
- list length = n_layers: one per hidden layer
- separate_last_group_dropout=True: may give length n_layers - 1 (last is per-group)

**Examples:**
- No dropout: `dropout_rates=0.0`
- Same for all: `dropout_rates=0.2`
- Per layer: `dropout_rates=[0.0, 0.2, 0.2]`
- With split head and per-group dropout on last layer:
```
hidden_layers=[64, 64, 64]
dropout_rates=[0.0, 0.2]  # last layer handled by group_dropout_rates
separate_last_group_dropout=True
```
---

### 1.3 Regularization & Initialization

#### `kernel_regularizers`
Kernel regularization for Dense layers.

Accepts:
- `None`: no regularization
- float: interpreted as L2 factor
- dict: `{"l1": 1e-4, "l2": 1e-3}`
- list for per-layer

**Examples:**
- same L2 on all: `kernel_regularizers=1e-4`
- per layers:
```
kernel_regularizers=[
    None,
    {"l2": 1e-4},
    {"l1": 1e-5},
]
```

---

#### `kernel_initializers`
Initializers for Dense kernels.
Accepts:
- String: `"glorot_uniform"`, `"he_normal"`, `"lecun_normal"`, etc.
- Keras initializer instance: `tf.keras.initializers.HeNormal()`
- List of those, one per hidden layer.

**Examples:**
- Default: `kernel_initializers="glorot_uniform"`
- Per layer: `kernel_initializers=["he_uniform", "he_uniform", "glorot_uniform"]`

---

#### `bias_initializers`
Bias initializers. Commonly `"zeros"` for all layers.

---

### 1.4 Output Layer & Split-Head Options

#### `output_dim: int`
Number of outputs.

---

#### `output_activation`
Activation applied to the final outputs
Accepts:
- 'None': no activation (pure linear regression)
- String: e.g. `"sigmoid"`, `"softmax"`
- Keras layer: `tf.keras.layers.Softmax`
- Callable: `tf.nn.softmax`

---

#### `output_kernel_initializer` / `output_bias_initializer`
Initializers for final Dense layer. Same semantics as for hidden layers

---

Split final connection (optional advanced feature)

Below fields are only relevant if you want to split the last hidden layer into two branches with separate outputs
---

#### `output_groups: Optional[Sequence[int]] = None`
Two integers summing to total output dimension. Enables split head if set.

**Examples:**
- `output_dim=100; output_groups=[45, 55]`

---

#### `last_hidden_split: Optional[int] = None`
Split the final hidden layer into two parts. 
Accepts:
- `None`: uses half of neurons `hidden_layers[-1] // 2`
- int: Must be between 1 and `last_hidden_units-1` e.g. `hidden_layers=[128]; last_hidden_split=64`

---

#### `separate_last_group_bn: bool = False`
If `True` and `output_groups` is set, BatchNorm and activation for the last hidden layer are applied separately to each slice after the split, instead of once before splitting.

---

#### `separate_last_group_dropout: bool = False`
If `True` and `output_groups` is set, dropout for the last hidden layer is applied separately to each group after splitting using `group_dropout_rates`. The last element of `dropout_rates` is ignored (and set to 0) in that case.

---

#### `group_dropout_rates: float | Sequence[float] = 0.0`
Dropout rates for each group when `separate_last_group_dropout=True`.
Accepts:
- scalar: same for both groups, e.g. `0.3`
- sequence `[d1, d2]` per group

**Example:**
```
separate_last_group_dropout=True
group_dropout_rates=[0.2, 0.1]
```

---

### 1.5 Compile Options

#### `optimizer`
Optimizer name or instance. 
Accepts:
- String: `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`
- Keras optimizer: `tf.keras.optimizers.Adam(learning_rate=1e-3)`

---

#### `lr: float = 1e-3`
Learning rate.

---

#### `loss`
Loss name or instance, e.g. `"mse"`, `"binary_crossentropy"`, `"sparse_categorical_crossentropy"`

---

#### `metrics: Tuple `= ("mae",)`
Sequence of metric names or metric objects.
**Examples:**
- `metrics=("mae",)`
- `metrics=["mse", "mae"]`
- `metrics=[tf.keras.metrics.RootMeanSquaredError()]`
---

---

## 2. TrainingConfig (in `ann/training.py`)
Controls **data loading**, **preprocessing**, **callbacks**, **output folders**, and **run settings**.

---

### 2.1 Run & Output

#### `output_dir: str = "Models"`
Root directory where runs are saved (each run in a subfolder): Models/<run_name>/...

---

#### `run_name: Optional[str] = None`
Name of the subdirectory under output_dir.
Accepts:
- `None`: auto timestamp like `training_20251116-123456`
- String

---

#### `save_best_only: bool = True`
For the best checkpoint callback: only save when monitored metric improves.

---

#### `save_last: bool = True`
Whether to also save a `"last.weights.h5"` after last epoch.

---

#### `save_weights_only: bool = True`
Save only weights (.h5). Faster and more portable.

---

#### `save_config: bool = True`
Save ANN and training config JSON files in `<output_dir>/<run_name>/configurations/`.

---

#### `resume_from: Optional[str] = None`
Resume training from directory.

**Example:**
- `resume_from="Models/training_20251110-100000"`

---

#### `base_cfg: Optional[str] = None`
Path to base config directory containing ann_base.json and train_base.json (as generated in cfg.py): `ANN_pipeline/config/`

---

### 2.2 Data Loading

#### `X_path: Optional[str] = None`
Path to feature data. Supported extensions: `.npy`, `.npz`, `.csv`

---

#### `y_path: Optional[str] = None`
Path to target data (same formats as `X_path`).

---

#### `batch_size: int = 128`
Mini-batch size. Typical values: 32, 64, 128, 256.

---

#### `epochs: int = 100`
Maximum number of training epochs.

---

#### `validation_split: float = 0.2`
Fraction of data used for validation (random split).

---

#### `shuffle_buffer: int = 10000`
Buffer size for `Dataset.shuffle`. Larger → better shuffling, more memory.

---

#### `prefetch: int = tf.data.AUTOTUNE`
How many batches to prefetch.
Accepts:
- `0` or `False`: no prefetch
- Positive int: fixed number of batches, e.g. `perfetch=2`
- String: `"auto"`, 'AUTOTUNE"`
- Callable: `tf.data.AUTOTUNE`

---

#### `cache_in_memory: bool = True`
If `True`, dataset is cached in memory after first epoch. Good for small/medium datasets.

---

### 2.3 Preprocessing

#### `use_sklearn_standard_scaler: bool = False`
If `True`, fits `sklearn.preprocessing.StandardScaler` on training `X` only, then applies to train and validation, and saves the scaler.

---

#### `scaler_pathname: str = "scaler.pkl"`
Filename under `<output_dir>/<run_name>` where the fitted scaler is saved via pickle.

---

#### `outlier_removal: bool = True`
If `True`, performs two-step PCA-based outlier removal on the training data.

---

#### `outlier_pathname: str = "outlier_index.pkl"`
Filename under `<output_dir>/<run_name>` where indices of removed samples (relative to training subset) are saved.

---

### 2.4 Callbacks

#### `early_stopping_patience: int = 10`
If > 0, enables `EarlyStopping` on monitor metric with given patience.

---

#### `reduce_lr_on_plateau: bool = True`
If `True`, use ReduceLROnPlateau on monitor.

---

#### `reduce_lr_patience: int = 5`
Patience (epochs) before reducing learning rate when metric plateaus. Only used if `reduce_lr_on_plateau=True`.

---

#### `monitor: str = "val_loss"`
Metric name for `EarlyStopping` and `ReduceLROnPlateau` and best checkpoint.

---

#### `mode: str = "min"`
Direction for `monitor`: `"min"` for loss, `"max"` for accuracy-type metrics.

---

#### `tensorboard: bool = True`
Whether to enable TensorBoard logging to `<output_dir>/<run_name>/tb`

---

### 2.5 Miscellaneous

#### `seed: int = 42`
Random seed for shuffling, initializations, etc.

---

#### `mixed_precision: bool = False`
If `True`, tries to enable mixed precision (`mixed_bfloat16` policy). Useful on GPUs/TPUs with good mixed-precision support.
---
