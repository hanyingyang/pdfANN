# pdfANN/inference.py
from __future__ import annotations
from pathlib import Path

import argparse, pickle

import numpy as np

# Local imports
from ann.model import ANN
from ann.training import (
    _load_array, 
    get_train_parser, 
    restart_from_run, 
    predict,
)

parser = get_train_parser()
parser.add_argument("-tp", "--test-path", type=str, default=str, help="directory to testing dataset")
args = parser.parse_args()

test_path = Path(args.test_path)
if not test_path.exists():
    raise FileNotFoundError(f"Test dataset does not exist: {test_path}.")

run_dir = Path(args.resume_from)
if not run_dir.exists():
    raise FileNotFoundError(f"Trained model does not exist: {run_dir}.")

# Load trained model's configs and weights
_, ann_cfg, train_cfg, _ = restart_from_run(run_dir)
best_ckpt = run_dir / "checkpoints" / "best.weights.h5"
if not best_ckpt.exists():
    raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
ann = ANN(ann_cfg)
ann.model.load_weights(best_ckpt) 

# Load test input
X = _load_array(test_path)
if train_cfg.use_sklearn_standard_scaler:
    scaler_path = run_dir / train_cfg.scaler_pathname
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_input = scaler.transform(X)
else:
    X_input = X

# Predict
y_pred = predict(ann.model, X_input)

# Save
np.save(run_dir / "prediction.npy", y_pred)
print("Inference finished")
