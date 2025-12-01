# pdfANN Pipeline (TensorFlow)

This repository provides a modular pipeline for constructing, training, evaluating, and deploying Artificial Neural Network (ANN) models using TensorFlow/Keras.  

## Project Structure

```
pdfANN/
├── ann/
│   ├── model.py             # ANN architecture and TensorFlow model builder
│   ├── training.py          # Training loop, dataset handling, callbacks, outlier removal
│   ├── config_utils.py      # Configuration loading/saving, JSON utilities, dataclass helpers
│   ├── utilities.py         # PCA tools, centring & scaling, outlier-removal routines
│
├── data/
│   ├── <DNS_timestep>/      # DNS raw data
│   ├── conAvg.m             # Calculate averaged reaction rate and density conditioned on c and Z
│   ├── filQty.m             # DNS raw data processing (.h5)
│   ├── pdfCZ.py             # DNS data processing → input features/target PDF extraction (KDE or histogram)
│
├── config/
│   ├── ann_base.json        # Baseline ANN configuration (from cfg.py)
│   ├── train_base.json      # Baseline training configuration
│
├── docs/
│   ├── config_guide.md      # Full reference guide to ANNConfig and TrainingConfig fields
│   ├── environment.yml      # Full Conda environment for reproducibility (Python + all dependencies)
│   ├── requirements.txt     # Minimal pip requirement list for running the ANN pipeline
│
├── train.py                 # Main training entry point
├── inference.py             # Run inference using a trained ANN
├── cfg.py                   # Generate baseline configuration JSON files
├── postprocessing.py        # Post-processing results (plots of marginal PDF and JSD values, scatter plot of reaction rate)
├── func.py                  # Post-processing utilities (KL, JS divergence, Beta PDF, Copula PDF)
```

## Components Overview

### 1. Raw Data processing — `data/filQty.m`

Filter the DNS snapshots:
- Snapshots of lifted H2 flame at 2 time steps: 489 and 801 
- Filter type: box (imboxfilt3), can change to Gaussian filter with imgaussfilt3
- Filter width: 1 laminar flame thermal thickness: ~ 8 dx (0.44 mm)
- Collects data at every 'Jump=18' points along each dimension of DNS domain

### 2. Input/target dataset — `data/pdfCZ.py`

Generates ANN-ready input–target datasets from 3D DNS data: 
- Supports KDE- and histogram-based PDF extraction for (c, lnZ)
- Discretisation in c: 45 uniform grids
- Discretisation in Z: 2 uniform grids in [0, left flammability], 
		       49 uniform grids in [left flammability, right flammability], 
 		       and 4 uniform grids in [right flammability, 1] 
		       → lnZ space
- Outputs target dataset of probability over each discretisation bins with range ([0, 1])
			
### 3. Conditional averaging reaction rate — `data/conAvg.m`

Averaged reaction rate/density on discretised c-Z space for integration with PDF during post-processing. 
This quantity was obtained by averaging over 10 DNS snapshots

### 4. ANN Model — `ann/model.py`

Defines the ANN architecture through the `ANNConfig` dataclass and an `ANN` class.  
Key features:

- Configurable hidden-layer layout  
- Customisable activations, batch normalisation, dropout, regularisation  
- Optional split-output structure  
- Custom kernel/bias initialisers  
- User-defined optimiser, loss, and metrics

### 5. Training System — `ann/training.py`

Implements the complete training workflow:

- `TrainingConfig` dataclass storing all hyperparameters  
- Loading datasets from `.npy`, `.npz`, `.csv`  
- Train/validation split with reproducible shuffling  
- Optional PCA-based outlier removal (two-stage)  
- Optional `StandardScaler` preprocessing  
- Efficient `tf.data` pipeline with caching, shuffling, batching, and prefetch  
- Callbacks: EarlyStopping, ReduceLROnPlateau, TensorBoard, checkpointing  
- Logging to text files and CSV  
- Training-loss plot generation  
- Prediction and evaluation helpers  
- Ability to resume from a previous run

### 6. Configuration Tools — `ann/config_utils.py`

Provides utilities for configuration management:

- JSON-safe saving and loading  
- Dataclass-to-dictionary conversions  
- Merging base configurations with CLI overrides  
- Filtering unknown keys  
- Default-value normalisation (e.g. AUTOTUNE handling)

### 7. Preprocessing & PCA Tools — `ann/utilities.py`

Only PCA-related mathematical tools are used in training:

- Centring and scaling methods (mean, min, AUTO)  
- PCA computation  
- Outlier detection using leverage and orthogonal distance    

### 8. Training Entry Point — `train.py`

Provides a command-line interface for model training:

- Builds configurations from scratch, JSON files, or CLI overrides  
- Constructs the ANN model  
- Runs the full training loop  
- Loads best checkpoint for final evaluation  
- Produces a prediction-versus-target scatter plot

### 9. Inference — `inference.py`

Loads a trained model and performs ANN inference:

- Loads saved configurations  
- Restores the best checkpoint  
- Applies saved scaler when required  
- Saves output predictions to `prediction.npy`

### 10. Base Configuration Generator — `cfg.py`

Creates example configuration files:

- `config/ann_base.json`  
- `config/train_base.json`

These serve as baseline templates for training and can be modified or used with CLI overrides. 
Please refer to `docs/config_guide` for full description of fields in configurations

### 11. Post-Processing Functions — `func.py`

Standalone utilities for analysis:

- **KL divergence** (`KLDiv`)  
- **Jensen–Shannon divergence** (`JSDiv`)  
- **Beta PDF evaluation** (`betaPDF`)  
- **Copula-based PDF interpolation** (`copulaPDF`)

### 12. Running environment

#### `environment.yml`

Conda environment file containing the full development environment: exact Python version,
TensorFlow build, NumPy/SciPy stack, and all support libraries needed to reproduce the
original training environment.

#### `requirement.md`

Minimal pip-based dependency list with only the essential packages required to run the
ANN model, training pipeline, and inference scripts. Intended for lightweight installation.

## Data Generation Workflow

### 1. Process DNS raw data
Run `filQty.m`

Output: 
```
data/<DNS_timestep>/<saved .mat file>
```

### 2. Extract input and target dataset
```bash
python pdfCZ.py
```

Output:
```
data/<target .npy>

(Optional)
data/<input .npy>
data/<filtered qty .npy>
```

## Training Workflow

### 1. (Optional) Generate base configurations

```bash
python cfg.py
```

### 2. Train a model using CLI arguments

```bash
python train.py -X data/<input.npy> -y data/<target.npy>
```

### 3. Train using configuration directory

```bash
python train.py --base-cfg ./config
```

### Output Files

Training results are saved in:

```
Models/<run_name>/
    checkpoints/                 # best and last weights
    configurations/              # saved ANN and training configs
    loss_history.png
    training_log.txt
    history.csv
    Scatter_pred_vs_target.png
```

## Inference Workflow

```bash
python inference.py --resume-from Models/<run_name> --test-path data/X_test.npy
```

Output:

```
Models/<run_name>/prediction.npy
```

## Post-processing Workfliw
```bash
python postprocessing.py
```

## Minimal Dependencies

The core pipeline requires:

```
tensorflow
numpy
scipy
matplotlib
scikit-learn
h5py
tensorboard
```

Python **3.10 or later** is recommended.

## Licence

This project is provided without warranty. You may use or modify it at your own discretion.
