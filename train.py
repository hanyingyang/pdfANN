# ANN_pipeline/train.py
from __future__ import annotations
from pathlib import Path

import argparse, pickle

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from ann.model import ANNConfig, ANN
from ann.training import (
    _load_array, 
    get_train_parser, 
    restart_from_run, 
    train, 
    TrainingConfig, 
    build_configs,
)

args = get_train_parser().parse_args()

# Resume path: load saved configs, then apply CLI flags
if getattr(args, "resume_from", None):
    ann, ann_cfg, train_cfg, _ = restart_from_run(args.resume_from)
    # allow normal CLI to override common knobs
    # Example: epochs, lr, batch_size; feel free to add others if your parser supports them.
    if hasattr(args, "epochs") and args.epochs is not None:
        train_cfg.epochs = args.epochs
    if hasattr(args, "lr") and args.lr is not None:
        ann_cfg.lr = args.lr
    if hasattr(args, "batch_size") and args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if hasattr(args, "run_name") and args.run_name is not None:
        train_cfg.run_name = args.run_name

# Build configs from scratch, then apply CLI flags
elif getattr(args, "base_cfg", None) is None:
    ann_cfg = ANNConfig(input_dim=4, hidden_layers=[32,64,64,64], output_dim=100)  # example setup from scratch; feel free to add others
    train_cfg = TrainingConfig()  
    ann_cfg, train_cfg = build_configs(args, ann_cfg, train_cfg)

# Load config from base configurations (JSON), then apply CLI flags
else:
    ann_cfg, train_cfg = build_configs(args)

ann = ANN(ann_cfg)
results = train(ann.model, ann_cfg, train_cfg)

# Evaluation: prediction against target
run_dir = Path(results["run_dir"])

X = _load_array(train_cfg.X_path)
y = _load_array(train_cfg.y_path)
if train_cfg.use_sklearn_standard_scaler:
    with open(run_dir / train_cfg.scaler_pathname, "rb") as f:
        scaler = pickle.load(f)
    X_input = scaler.transform(X)
else:
    X_input = X

best_ckpt = Path(results["best_ckpt"])
ann.model.load_weights(best_ckpt)     # load best weights
y_pred = ann.model.predict(X_input)   # prediction

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.scatter(y[:,:-1].ravel(), y_pred[:,:-1].ravel(), s=2, edgecolors='black', linewidths=0.1)
ax.set_xlabel('Target_true')
ax.set_ylabel('Target_pred')
lims = [np.min(y[:,:-1]), np.max(y[:,:-1])]
ax.plot(lims, lims, 'r')
fig.tight_layout()
fig.savefig(run_dir / 'Scatter_pred_vs_target.png', dpi=500)
plt.show()
plt.close(fig)
