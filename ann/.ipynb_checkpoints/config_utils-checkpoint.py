# pdfANN/ann/config_utils.py
from __future__ import annotations
from dataclasses import asdict, is_dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, Type, TypeVar
import json
import argparse

T = TypeVar("T")

def _json_safe_value(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe_value(vv) for k, vv in v.items()}
    if is_dataclass(v):
        return _json_safe_value(asdict(v))
    # fallback: string representation
    return str(v)

def json_safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable dict (recursively)."""
    return {str(k): _json_safe_value(v) for k, v in d.items()}

def save_json(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(json_safe_dict(obj), f, indent=2)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)

def save_configs(run_dir: Union[str, Path], ann_cfg, train_cfg) -> None:
    """Save `ann_config.json` and `training_config.json` under configurations/."""
    cfg_dir = Path(run_dir) / "configurations"
    save_json(cfg_dir / "ann_config.json", asdict(ann_cfg))
    save_json(cfg_dir / "training_config.json", asdict(train_cfg))

def load_configs(run_dir: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load configs back to plain dicts (caller can reconstruct dataclasses)."""
    cfg_dir = Path(run_dir) / "configurations"
    ann_d = load_json(cfg_dir / "ann_config.json")
    tr_d  = load_json(cfg_dir / "training_config.json")
    return ann_d, tr_d

def only_known_keys(d: Dict[str, Any], DC: Type[T]) -> Dict[str, Any]:
    if not is_dataclass(DC):
        raise TypeError("DC must be a dataclass type")
    allowed = {f.name for f in fields(DC)}
    return {k: v for k, v in d.items() if k in allowed}

def dataclass_merge(base: T, override: Dict[str, Any]) -> T:
    """Return a copy of dataclass 'base' with 'override' fields applied (unknown keys ignored)."""
    clean = only_known_keys(override, type(base))
    return replace(base, **clean)

def args_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert argparse. Namespace to a dict of user-provided arguments.
    Assumes all relevant add_argument() calls used default=argparse.SUPPRESS,
    so only user-set arguments appear in vars(args), plus out_dir.
    """
    return vars(args).copy()

def normalize_training_defaults(train_cfg: Any) -> None:
    """
    Normalize a few fields that might come as strings/aliases from JSON or CLI.
    Mutates train_cfg in place.
    """
    # Example: map 'auto' → AUTOTUNE if your TrainingConfig.prefetch expects an int
    try:
        import tensorflow as tf
        if getattr(train_cfg, "prefetch", None) in ("auto", "AUTOTUNE"):
            train_cfg.prefetch = tf.data.AUTOTUNE
    except Exception:
        pass
