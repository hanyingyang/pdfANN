# ANN_pipeline/ann/training.py
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import argparse, time, pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Local imports
from .model import ANNConfig, ANN
from .config_utils import (
    save_configs, 
    load_configs,
    load_json,
    normalize_training_defaults,
    dataclass_merge, 
    args_to_overrides,
)
from .utilities import *

# ------------------ Training config ------------------
@dataclass
class TrainingConfig:
    # Run / IO
    output_dir: str = "Models"
    run_name: Optional[str] = None                # if None -> timestamped
    save_best_only: bool = True
    save_last: bool = True
    save_weights_only: bool = True                # weights-only is faster & portable
    save_config: bool = True
    resume_from: Optional[str] = None             # path to a checkpoint to resume from
    base_cfg: Optional[str] = None                # path to base configurations

    # Data
    X_path: Optional[str] = None                  # path to input_features .npy/.npz/.csv
    y_path: Optional[str] = None                  # path to target .npy/.npz/.csv
    batch_size: int = 128
    epochs: int = 100
    validation_split: float = 0.2               
    shuffle_buffer: int = 10_000
    prefetch: Union[int, Any] = tf.data.AUTOTUNE  # used for gpu training: cpu prepares the next batch while gpu is training on current batch
    cache_in_memory: bool = True                  # catches data in ram after the first epoch, and no reloading and preprocessing for subsequent epoches

    # Preprocessing (optional sklearn scaler on X)
    use_sklearn_standard_scaler: bool = False     # if True, fit on train only and save .pkl
    scaler_pathname: str = "scaler.pkl"           # relative to run_dir
    outlier_removal: bool = True
    outlier_pathname: str = "outlier_index.pkl"

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_on_plateau: bool = True
    reduce_lr_patience: int = 5                   # works if reduce_lr_on_plateau = true
    monitor: str = "val_loss"
    mode: str = "min"                             # "min" or "max"
    tensorboard: bool = True                      # web-based dashboard for visualizing model training

    # Reproducibility / perf
    seed: int = 42
    mixed_precision: bool = False                 # enables certain part of computation in lower precision; set bf16/fp16 if your hardware supports it

# ------------------ CLI helpers ------------------
def _parse_int_list(s: str) -> list[int]:
    # "32,64,128" -> [32,64,128]
    return [int(x) for x in s.split(",") if x.strip()]

def _parse_float_list(s: str) -> list[float]:
    # "0,0.1,0" -> [0.0,0.1,0.0]
    return [float(x) for x in s.split(",") if x.strip()]

def _parse_activation_spec(s: str):
    """
    Accepts:
      "relu" -> "relu"
      "gelu" -> "gelu"
      "leaky_relu:0.1" -> {"name": "leaky_relu", "alpha": 0.1}
      "elu:1.0" -> {"name": "elu", "alpha": 1.0}
      "prelu" -> {"name": "prelu"}
    """
    s = s.strip().lower()
    if ":" not in s:
        if s in ("prelu",):
            return {"name": s}
        return s
    name, val = s.split(":", 1)
    if name in ("leaky_relu", "elu"):
        try:
            alpha = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid alpha for {name}: {val}")
        return {"name": name, "alpha": alpha}
    # fallback: treat as plain string if unknown
    return name

def get_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ANN with simple CLI")

    # IO
    p.add_argument("-od", "--output-dir", type=str, default="Models", help="directory to save training output")
    p.add_argument("-rn", "--run-name", type=str, default=argparse.SUPPRESS, help="name of folder saving traning output")
    p.add_argument("-rf", "--resume-from", type=str, default=argparse.SUPPRESS, help="training directory to resume training")
    p.add_argument("-bc", "--base-cfg", type=str, default=argparse.SUPPRESS, help="base configurations directory to initialize model and training")

    # Data
    p.add_argument("-X", "--X-path", type=str, default=argparse.SUPPRESS, help="path to features (.npy/.npz/.csv)")
    p.add_argument("-y", "--y-path", type=str, default=argparse.SUPPRESS, help="path to targets (.npy/.npz/.csv)")
    p.add_argument("-v", "--validation-split", type=float, default=argparse.SUPPRESS, help="validation split fraction (e.g. 0.2)")
    p.add_argument("-b", "--batch-size", type=int, default=argparse.SUPPRESS, help="size of batch used in training")
    p.add_argument("-e", "--epochs", type=int, default=argparse.SUPPRESS, help="number of training epochs")
    
    # Model
    p.add_argument("-idim", "--input-dim", type=int, default=argparse.SUPPRESS, help="dimension of input layer")
    p.add_argument("-hl", "--hidden-layers", type=_parse_int_list, default=argparse.SUPPRESS,
                   help="neurons at each hidden layer")
    p.add_argument("-a", "--activations", type=_parse_activation_spec, default=argparse.SUPPRESS,
                   help="activation function, e.g. relu | gelu | silu | leaky_relu:0.1 | prelu")
    p.add_argument("-bn", "--use-batchnorm", action="store_true", default=argparse.SUPPRESS, help="Enable BatchNorm on hidden layers")
    p.add_argument("-d", "--dropout-rates", type=_parse_float_list, default=argparse.SUPPRESS,
                   help="Enable dropout on hidden layers, e.g. 0,0.1,0.2")
    p.add_argument("-odim", "--output-dim", type=int, default=argparse.SUPPRESS, help="dimension of output layer")
    p.add_argument("-oa", "--output-activation", type=str, default=argparse.SUPPRESS,
                   help="activation function at output layer")

    # Training knobs   
    p.add_argument("-opt", "--optimizer", type=str, default=argparse.SUPPRESS,
                   help="method to decrease training loss, e.g. adam | adamw | rmsprop | sgd ...")
    p.add_argument("-l", "--lr", type=float, default=argparse.SUPPRESS, help="learning rate")
    p.add_argument("-es", "--early-stopping", dest="early_stopping_patience", type=int, default=argparse.SUPPRESS,
                   help="patience to stop training if no decrease in loss (0 to disable early stopping)")
    p.add_argument("-rl", "--reduce-lr", dest="reduce_lr_on_plateau", action="store_true", default=argparse.SUPPRESS, 
                   help="enable ReduceLROnPlateau callback")
    p.add_argument("-rp", "--reduce-lr-patience", type=int, default=argparse.SUPPRESS, 
                   help="patience for ReduceLROnPlateau")

    p.add_argument("-sd", "--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("-m", "--mixed-precision", action="store_true", default=argparse.SUPPRESS, 
                   help="enable lower precision computation")

    return p

# ------------------- Logging -------------------
class EpochLossLogger(tf.keras.callbacks.Callback):
    """Append per-epoch metrics to a text file and keep running bests."""
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = Path(log_path)
        self.best_train = float("inf")
        self.best_val = float("inf")

        # fresh file with header
        self.log_path.write_text(
            "epoch | loss | val_loss | best_loss_so_far | best_val_loss_so_far\n"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if loss is not None:
            self.best_train = min(self.best_train, float(loss))
        if val_loss is not None:
            self.best_val = min(self.best_val, float(val_loss))

        line = (
            f"{epoch+1:04d} | "
            f"{(loss if loss is not None else float('nan')):.6f} | "
            f"{(val_loss if val_loss is not None else float('nan')):.6f} | "
            f"{self.best_train:.6f} | "
            f"{self.best_val:.6f}\n"
        )
        with self.log_path.open("a") as f:
            f.write(line)

# ------------------ Utilities ------------------
def set_global_seed(seed: int):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def maybe_enable_mixed_precision(enabled: bool):
    if not enabled:
        return
    try:
        from tensorflow.keras import mixed_precision
        # Prefer bfloat16 on modern accelerators; fallback fp16 if you want
        policy = mixed_precision.Policy("mixed_bfloat16")
        mixed_precision.set_global_policy(policy)
        print(f"[INFO] Mixed precision enabled: {policy.name}")
    except Exception as e:
        print(f"[WARN] Mixed precision not enabled: {e}")

def _timestamp_name() -> str:
    return "training_" + time.strftime("%Y%m%d-%H%M%S")

def make_run_dir(cfg: TrainingConfig) -> Path:
    base = Path(cfg.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = cfg.run_name or _timestamp_name()
    run_dir = base / name

    # if run_dir already exists, append a counter
    counter = 1
    while run_dir.exists():
        run_dir = base / f"{name}_{counter:02d}"
        counter += 1  
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "configurations").mkdir(parents=True, exist_ok=True)
    return run_dir

def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".npy",):
        return np.load(p)
    if p.suffix.lower() in (".npz",):
        return np.load(p)["arr_0"]
    if p.suffix.lower() in (".csv",):
        return np.loadtxt(p, delimiter=",")
    raise ValueError(f"Unsupported data file: {p.suffix}")

def _remove_outliers(X: np.ndarray, y: Optional[np.ndarray]):
    """
    Two-step PCA outlier removal, returns kept X,y and boolean mask over original indices.
    """
    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    print("Original training samples: {}".format(n))

    X1, _, mask1 = outlier_removal_leverage(X, 2, "mean", "auto")   # returns removed indices as mask
    removed1 = np.zeros(n, dtype=bool); removed1[mask1] = True
    keep &= ~removed1
    print("Training samples after first outlier removal: {}".format(X1.shape[0]))

    X2, _, mask2 = outlier_removal_orthogonal(X[keep], 2, "mean", "auto")
    # mask2 indexes over compressed space; map back:
    kept_idx = np.where(keep)[0]
    removed2 = np.zeros(n, dtype=bool); removed2[kept_idx[mask2]] = True
    keep &= ~removed2
    print("Training samples after second outlier removal: {}".format(X2.shape[0]))

    y2 = y[keep] if y is not None else None
    return X[keep], y2, ~keep

def build_datasets_from_arrays(cfg: TrainingConfig, run_dir: Path) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if cfg.X_path is None or cfg.y_path is None:
        raise ValueError("X_path and y_path must be set before training.")
    X = _load_array(cfg.X_path)
    y = _load_array(cfg.y_path)
    
    """
    Split X/y into train/val by cfg.validation_split and return tf.data datasets.
    """
    assert X.ndim >= 2, "X must be a 2D array [n_samples, n_features] (or higher)."
    assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
    n = X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(n); rng.shuffle(idx)
    split = int(n * (1.0 - cfg.validation_split))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]  

    # Optional: Removing outlier in training dataset
    if cfg.outlier_removal:
        X_train, y_train, removed_mask = _remove_outliers(X_train, y_train)
        with open(run_dir / cfg.outlier_pathname, "wb") as f:
            pickle.dump(train_idx[removed_mask], f)

    # Optional: Standardize features using sklearn
    if cfg.use_sklearn_standard_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        with open(run_dir / cfg.scaler_pathname, "wb") as f:
            pickle.dump(scaler, f)

    def wrap(x, y, *, shuffle: bool):
        ds = tf.data.Dataset.from_tensor_slices((x, y)) 
        if cfg.cache_in_memory:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)
        ds = ds.batch(cfg.batch_size)
        if cfg.prefetch:
            ds = ds.prefetch(cfg.prefetch)
        return ds

    return wrap(X_train, y_train, shuffle=True), wrap(X_val, y_val, shuffle=False)

def make_callbacks(run_dir: Path, cfg: TrainingConfig) -> List[tf.keras.callbacks.Callback]:
    ckpt_dir = run_dir / "checkpoints"

    # Best checkpoint
    cbs: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_dir / "best.weights.h5"),
            monitor=cfg.monitor, 
            mode=cfg.mode,
            save_best_only=cfg.save_best_only,
            save_weights_only=cfg.save_weights_only,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        
        # per-epoch text logger
        EpochLossLogger(run_dir / "training_log.txt"),
    ]

    if cfg.early_stopping_patience and cfg.early_stopping_patience > 0:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor=cfg.monitor, 
            mode=cfg.mode,
            patience=cfg.early_stopping_patience,
            restore_best_weights=True, 
            verbose=1
        ))

    if cfg.reduce_lr_on_plateau:
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=cfg.monitor, 
                mode=cfg.mode, 
                patience=cfg.reduce_lr_patience, 
                factor=0.5, 
                min_lr=1e-7, 
                verbose=1
        ))

    if cfg.tensorboard:
        tb = tf.keras.callbacks.TensorBoard(
            log_dir=str(run_dir / "tb"),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        )
        cbs.append(tb)

    # Save "last" checkpoint each epoch if requested
    if cfg.save_last:
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                str(ckpt_dir / "last.weights.h5"),
                save_best_only=False,
                save_weights_only=cfg.save_weights_only,
                verbose=0,
        ))

    return cbs

def _resolve_resume_path(resume_from: Optional[str]) -> Optional[Path]:
    if not resume_from:
        return None
    p = Path(resume_from)
    if not p.exists():
        print(f"[WARN] resume_from not found: {p}")
        return None
    if p.is_dir():
        # prefer last, fallback best
        last = p / "checkpoints" / "last.weights.h5"
        best = p / "checkpoints" / "best.weights.h5"
        if last.exists():
            return last
        if best.exists():
            return best
        print(f"[WARN] no checkpoints found under: {p}")
        return None
    return p  # assume file

def _readable_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def _trainable_params(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))

# ------------------ Train / Evaluate ------------------
def train(model: tf.keras.Model, ann_cfg: ANNConfig, cfg: TrainingConfig):
    """Train a compiled model."""
    set_global_seed(cfg.seed)
    maybe_enable_mixed_precision(cfg.mixed_precision)

    run_dir = make_run_dir(cfg)
    if cfg.save_config:
        save_configs(run_dir, ann_cfg, cfg)
   
    train_ds, val_ds = build_datasets_from_arrays(cfg, run_dir) # build datasets
    callbacks = make_callbacks(run_dir, cfg) # Callbacks

    # --- start timing ---
    t0 = time.perf_counter()
    
    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    # --- stop timing ---
    t1 = time.perf_counter()

    # Plot loss
    plt.figure()
    plt.plot(history.history.get("loss",[]))
    plt.plot(history.history.get("val_loss",[]))
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(run_dir / 'loss_history.png', dpi=300)
    plt.close()

    # Save final weights (optional, already saved 'last' via callback)
    final_path = run_dir / "checkpoints" / "final.weights.h5"
    model.save_weights(str(final_path))
    print(f"[INFO] Final weights saved to: {final_path}")

    # --- append model size & timing to training_log.txt ---
    log_path = run_dir / "training_log.txt"

    # params + approximate memory footprint (dtype-aware best-effort)
    total_params = int(model.count_params())
    trainable_params = _trainable_params(model)
    # try to infer dtype bytes (bfloat16/float16=2, float32=4)
    dtype_bytes = 2 if "float16" in str(model.dtype_policy) or "bfloat16" in str(model.dtype_policy) else 4
    approx_param_bytes = total_params * dtype_bytes

    best_ckpt = run_dir / "checkpoints" / "best.weights.h5"
    last_ckpt = run_dir / "checkpoints" / "last.weights.h5"

    def file_size(p: Path) -> str:
        return _readable_bytes(p.stat().st_size) if p.exists() else "N/A"

    with log_path.open("a") as f:
        f.write("\n--- Summary ---\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Approx. in-memory weights size: {_readable_bytes(approx_param_bytes)}\n")
        f.write(f"Final checkpoint size: {file_size(final_path)}\n")
        f.write(f"Best checkpoint size:  {file_size(best_ckpt)}\n")
        f.write(f"Last checkpoint size:  {file_size(last_ckpt)}\n")
        # best losses from history (if present)
        best_train = min(history.history.get('loss', [float('inf')]))
        best_val = min(history.history.get('val_loss', [float('inf')]))
        f.write(f"Best training loss: {best_train:.6f}\n")
        f.write(f"Best validation loss: {best_val:.6f}\n")
        f.write(f"Total training time: {t1 - t0:.2f} seconds\n")

    print(f"[INFO] Wrote training log to: {log_path}")   

    return {
        "run_dir": str(run_dir),
        "history": history.history,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "final_ckpt": str(final_path),
        "log_txt": str(log_path),
    }

def evaluate(model: tf.keras.Model, ds: tf.data.Dataset):
    """Simple evaluate wrapper that returns (loss, metrics...)."""
    return model.evaluate(ds, verbose=0)

def predict(model: tf.keras.Model, ds_or_array: Union[tf.data.Dataset, np.ndarray], batch_size: Optional[int] = None):
    if isinstance(ds_or_array, tf.data.Dataset):
        return model.predict(ds_or_array, verbose=0)
    return model.predict(ds_or_array, batch_size=batch_size or 256, verbose=0)

# -------- Restart training from a saved run --------
def restart_from_run(run_dir: Union[str, Path]) -> Tuple[ANN, ANNConfig, TrainingConfig, Path]:
    ann_d, tr_d = load_configs(run_dir)
    ann_cfg = ANNConfig.from_dict(ann_d)  
    train_cfg = TrainingConfig(**tr_d)
    normalize_training_defaults(train_cfg)
    ann = ANN(ann_cfg)
    ckpt = _resolve_resume_path(run_dir)
    if ckpt:
        ann.model.load_weights(str(ckpt))
        print(f"[INFO] Loaded weights from: {ckpt}")
    return ann, ann_cfg, train_cfg, Path(run_dir)

# --------- Load and override configurations ---------
def build_configs(
    args, 
    ann_cfg: Optional[ANNConfig] = None, 
    train_cfg: Optional[TrainingConfig] = None
) -> tuple[ANNConfig, TrainingConfig]:    
    
    base_dir = getattr(args, "base_cfg", None)
    
    # JSON bases
    if base_dir is not None:
        ann_json = load_json(Path(base_dir) / "ann_base.json")
        train_json = load_json(Path(base_dir) / "train_base.json")
        ann_cfg = ANNConfig.from_dict(ann_json)
        train_cfg = TrainingConfig(**train_json)
        normalize_training_defaults(train_cfg)

    # apply CLI flags actually provided
    overrides = args_to_overrides(args)
    ann_field_names = {f.name for f in fields(ANNConfig)}
    ann_override = {k: v for k, v in overrides.items() if k in ann_field_names}
    train_override = {k: v for k, v in overrides.items() if k not in ann_field_names}
    ann_cfg = dataclass_merge(ann_cfg, ann_override)
    train_cfg = dataclass_merge(train_cfg, train_override)
    normalize_training_defaults(train_cfg)

    # final sanity: ensure required ANN fields are set
    for req in ("input_dim","output_dim","hidden_layers"):
        if getattr(ann_cfg, req, None) in (None, [], 0):
            raise ValueError(f"ANNConfig.{req} must be provided (via explicit cfg, JSON, or CLI).")

    return ann_cfg, train_cfg