# pdfANN/ann/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Union, List, Optional, Dict
from collections.abc import Callable, Mapping, Sequence 

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# ----------------------------
# Config
# ----------------------------
@dataclass
class ANNConfig:
    # --- Model topology ---
    input_dim: int
    hidden_layers: List[int]  # e.g. [32, 64, 128]

    # Activations can be:
    # - None / "relu" / "gelu" / "silu" / ...
    # - a Keras Layer (e.g., LeakyReLU(alpha=0.1), PReLU())
    # - a callable (e.g., tf.nn.gelu)
    # - per-layer list/tuple of the above
    activations: Union[
        None,
        str,
        tf.keras.layers.Layer,
        Callable[[tf.Tensor], tf.Tensor],
        Sequence[Union[None, str, tf.keras.layers.Layer, Callable[[tf.Tensor], tf.Tensor], Mapping[str, Any]]],
    ] = "relu"    

    # --- Per-layer options (broadcastable) ---
    use_batchnorm: Union[bool, Sequence[bool]] = False
    # BatchNorm params (if used)
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    dropout_rates: Union[float, Sequence[float]] = 0.0
    
    # Kernel regularizer per layer:
    # - None
    # - float (interpreted as L2 factor)
    # - {"l1": <float>, "l2": <float>}  (either key optional)
    # - list of the above, one per hidden layer
    kernel_regularizers: Union[
        None, float, Mapping[str, Any], Sequence[Optional[Union[float, Mapping[str, Any]]]]
    ] = None
    
    # Kernel and bias initializer per layer:
    kernel_initializers: Union[
        str, tf.keras.initializers.Initializer,
        Sequence[Union[str, tf.keras.initializers.Initializer]]
    ] = "glorot_uniform"
    bias_initializers: Union[
        str, tf.keras.initializers.Initializer,
        Sequence[Union[str, tf.keras.initializers.Initializer]]
    ] = "zeros"   

    # --- Output layer ---
    output_dim: int=1
    output_activation: Union[None, str, tf.keras.layers.Layer, Callable[[tf.Tensor], tf.Tensor]] = None

    # Output initializers (separate options)
    output_kernel_initializer: Union[str, tf.keras.initializers.Initializer] = "glorot_uniform"
    output_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"    

    # >>> SPECIAL SPLIT FINAL CONNECTION <<<
    # Output split: e.g., output_dim=100, output_groups=[45, 55]
    output_groups: Optional[Sequence[int]] = None          # length 2 if provided
    last_hidden_split: Optional[int] = None                # defaults to hidden_layers[-1]//2

    # Control normalization/dropout positioning for LAST hidden layer when using split:
    # - If False (default): BN/Dropout for last hidden layer happen BEFORE split (shared).
    # - If True: suppress pre-split BN/Dropout on last hidden, and apply them
    #   independently to each slice AFTER split.
    separate_last_group_bn: bool = False
    separate_last_group_dropout: bool = False
    # Per-group dropout rates when separate_last_group_dropout=True
    # Can be a scalar (same for both slices) or a 2-element sequence [d1, d2].
    group_dropout_rates: Union[float, Sequence[float]] = 0.0

    # --- Compile ---
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    lr: float = 1e-3
    loss: Union[str, tf.keras.losses.Loss] = "mse"
    metrics: Tuple = ("mae",)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ANNConfig":
        # minimal normalization for keys that might be json-ified
        return ANNConfig(**d)

# ----------------------------
# Model
# ----------------------------
class ANN:
    def __init__(self, cfg: ANNConfig):
        self.cfg = cfg
        self.model = self._build()

    # ---------- helpers ----------
    @staticmethod
    def _as_list(x, n, *, name: str):
        """Broadcast scalars/strings/bools to length n; validate lists/tuples length."""
        if isinstance(x, (list, tuple)):
            if len(x) != n:
                raise ValueError(f"{name} must have length {n}, got {len(x)}")
            return list(x)
        return [x] * n

    @staticmethod
    def _make_regularizer(spec):
        """Turn user spec into a Keras regularizer or None."""
        if spec is None:
            return None
        if isinstance(spec, (int, float)):
            # interpret as L2
            return tf.keras.regularizers.l2(float(spec))
        if isinstance(spec, Mapping):
            l1 = float(spec.get("l1", 0.0)) if "l1" in spec else 0.0
            l2 = float(spec.get("l2", 0.0)) if "l2" in spec else 0.0
            # if user provides {"value": x} treat as l2=x
            if "value" in spec and l1 == 0.0 and l2 == 0.0:
                l2 = float(spec["value"])
            if l1 == 0.0 and l2 == 0.0:
                return None
            return tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
        raise TypeError(
            "kernel_regularizers items must be None, float (l2), or dict with keys 'l1' and/or 'l2'."
        )

    @staticmethod
    def _make_initializer(spec):
        # Accept strings ("he_uniform"), dict configs, or initializer instances
        return tf.keras.initializers.get(spec)

    @staticmethod
    def _maybe_int(x):
        try:
            return int(x)
        except Exception:
            return x

    def _normalize_dropouts(self, dropouts, n_layers, *, separate_last_group_dropout: bool) -> list[float]:
        # broadcast scalar
        if not isinstance(dropouts, (list, tuple)):
            dropouts = [float(dropouts)] * n_layers
            
        # exact length — keep as-is
        elif len(dropouts) == n_layers:
            dropouts = [float(d) for d in dropouts]

        # allow len == n-1 when last layer uses per-group dropout
        elif separate_last_group_dropout and len(dropouts) == n_layers - 1:
            dropouts = [float(d) for d in dropouts] + [0.0]  # last value unused; set to 0

        else:
            raise ValueError(
                f"dropout_rates length must be 1, {n_layers}, or {n_layers-1} " 
                f"(when separate_last_group_dropout=True); got {len(dropouts)}."
            )

        # Warn if last-layer dropout is ignored when using per-group dropout
        if separate_last_group_dropout:
            last_dr = dropouts[-1]
            if last_dr != 0.0:
                logger.warning(
                    "dropout_rates[-1]=%s is ignored because separate_last_group_dropout=True; "
                    "using group_dropout_rates instead.", last_dr
                )
                dropouts[-1] = 0.0  # ensure pre-split dropout not applied
        
        return dropouts
        

    def _clone_layer_with_name(self, layer: tf.keras.layers.Layer, name: Optional[str]):
        """Clone a Keras layer, giving it 'name' if provided, otherwise let Keras auto-name."""
        cfg = layer.get_config()
        if name is None:
            cfg.pop("name", None)  # remove to allow auto-naming
        else:
            cfg["name"] = name
        return layer.__class__.from_config(cfg)

    def _make_activation_layer(self, spec, name: Optional[str] = None):
        """
        Returns a Keras layer for the given activation spec.
        Accepts:
          - None
          - string name (e.g., "relu", "gelu", "silu"/"swish")
          - a Keras Layer instance (e.g., LeakyReLU(alpha=0.1), PReLU())
          - a callable (e.g., tf.nn.gelu)
          - a dict/Mapping like {"name": "leaky_relu", "alpha": 0.1}
        """
        if spec is None:
            return None

        # Layer instance
        if isinstance(spec, tf.keras.layers.Layer):
            return self._clone_layer_with_name(spec, name)

        # Callable function: wrap
        if callable(spec) and not isinstance(spec, str):
            return tf.keras.layers.Activation(spec, name=name)

        # Plain string name (special-case softmax for nicer summary label)
        if isinstance(spec, str):
            s = spec.lower()
            if s == "softmax":
                return tf.keras.layers.Softmax(name=name)
            return tf.keras.layers.Activation(spec, name=name)

        # Dict/Mapping (parametric cases + fallback-to-string)
        if isinstance(spec, Mapping) or isinstance(spec, dict) or hasattr(spec, "keys"):
            act_name = str(spec.get("name", "")).lower()
            if act_name in ("leaky_relu", "lrelu"):
                alpha = float(spec.get("alpha", 0.3))
                return tf.keras.layers.LeakyReLU(alpha=alpha, name=name)
            if act_name == "elu":
                alpha = float(spec.get("alpha", 1.0))
                return tf.keras.layers.ELU(alpha=alpha, name=name)
            if act_name == "prelu":
                return tf.keras.layers.PReLU(name=name)
            # Fallback: try as a string name (covers "gelu", "silu"/"swish", etc.)
            return tf.keras.layers.Activation(act_name, name=name)

        raise TypeError(
            "Activation must be None, a string, a Keras Layer, a callable, "
            "or a dict like {'name': 'leaky_relu', 'alpha': 0.1}."
        )  

    # ------------- build ------------
    def _build(self) -> tf.keras.Model:
        cfg = self.cfg
        n_layers = len(cfg.hidden_layers)
        
        act_specs = self._as_list(cfg.activations, n_layers, name="activations")
        use_bn = self._as_list(cfg.use_batchnorm, n_layers, name="use_batchnorm")
        dropouts = self._normalize_dropouts(
            cfg.dropout_rates, n_layers,
            separate_last_group_dropout=bool(cfg.output_groups and cfg.separate_last_group_dropout)
        )

        # Regularizaers
        if isinstance(cfg.kernel_regularizers, (list, tuple)):
            kregs_spec = cfg.kernel_regularizers
            if len(kregs_spec) != n_layers:
                raise ValueError(
                    f"kernel_regularizers must have length {n_layers}, got {len(kregs_spec)}"
                )
        else:
            kregs_spec = [cfg.kernel_regularizers] * n_layers
        kregs = [self._make_regularizer(s) for s in kregs_spec]

        # Initializers
        kinits = self._as_list(cfg.kernel_initializers, n_layers, name="kernel_initializers")
        binits = self._as_list(cfg.bias_initializers,   n_layers, name="bias_initializers")
        kinits = [self._make_initializer(k) for k in kinits]
        binits = [self._make_initializer(b) for b in binits]

        # -------Build hidden trunk ------
        inputs = tf.keras.Input(shape=(cfg.input_dim,))
        x = inputs

        last_idx = n_layers - 1
        
        for i, (units, act_spec, bn_flag, dr, kr, kinit, binit) in enumerate(
            zip(cfg.hidden_layers, act_specs, use_bn, dropouts, kregs, kinits, binits)
        ):
            # Decide whether to defer norm or dropout for the LAST hidden layer only
            defer_norm   = (cfg.output_groups is not None) and cfg.separate_last_group_bn and (i == last_idx)
            defer_dropout = (cfg.output_groups is not None) and cfg.separate_last_group_dropout and (i == last_idx)

            apply_norm_now = bn_flag and not defer_norm
            apply_dropout_now = (dr > 0.0) and not defer_dropout

            # Always set Dense activation=None; we apply activation explicitly as a layer.
            # If BN happens now or later (defer), drop bias because BN's beta handles the shift.
            dense_activation = None
            use_bias = not (apply_norm_now or defer_norm)          

            x = tf.keras.layers.Dense(
                units,
                activation=dense_activation,
                use_bias=use_bias,
                kernel_regularizer=kr,
                kernel_initializer=kinit,
                bias_initializer=binit,
                name=f"dense_{i+1}",
            )(x)

            if apply_norm_now:
                x = tf.keras.layers.BatchNormalization(
                    momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, name=f"bn_{i+1}"
                )(x)

            # Apply activation now unless we're deferring BN (and thus activation) to after the split
            if not defer_norm:
                act_layer = self._make_activation_layer(act_spec, name=f"act_{i+1}")
                if act_layer is not None:
                    x = act_layer(x)

            if apply_dropout_now:
                x = tf.keras.layers.Dropout(dr, name=f"dropout_{i+1}")(x)

        # ---- Output head(s) ----
        out_kinit = self._make_initializer(cfg.output_kernel_initializer)
        out_binit = self._make_initializer(cfg.output_bias_initializer)
        
        if cfg.output_groups is None:
            # Standard fully-connected output
            out_act = self._make_activation_layer(cfg.output_activation, name="act_output")
            outputs = tf.keras.layers.Dense(
                cfg.output_dim, 
                activation=None,  # apply explicit activation layer for consistency 
                kernel_initializer=out_kinit,
                bias_initializer=out_binit,
                name="output",
            )(x)
            if out_act is not None:
                outputs = out_act(outputs)
        else:
            if len(cfg.output_groups) != 2:
                raise ValueError("output_groups must be a sequence of two integers, e.g., [45, 55].")
            if sum(cfg.output_groups) != cfg.output_dim:
                raise ValueError(f"sum(output_groups) must equal output_dim ({cfg.output_dim}).")

            last_units = self._maybe_int(cfg.hidden_layers[-1])
            split_idx = cfg.last_hidden_split if cfg.last_hidden_split is not None else last_units // 2
            if not (1 <= split_idx <= last_units - 1):
                raise ValueError(f"last_hidden_split must be in [1, {last_units-1}], got {split_idx}")

            # Split the last hidden activations
            x1 = tf.keras.layers.Lambda(lambda t: t[:, :split_idx], name="last_hidden_slice_1")(x)
            x2 = tf.keras.layers.Lambda(lambda t: t[:, split_idx:], name="last_hidden_slice_2")(x)

            # If we deferred BN for the last hidden, apply BN (+ the last layer's activation) per group now
            if cfg.separate_last_group_bn:
                x1 = tf.keras.layers.BatchNormalization(
                    momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, name="bn_last_group_1"
                )(x1)
                x2 = tf.keras.layers.BatchNormalization(
                    momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, name="bn_last_group_2"
                )(x2)
                
                # Apply activation ONCE here (since Dense had activation=None when defer_norm=True)
                act1 = self._make_activation_layer(act_specs[last_idx], name="act_last_group_1")
                act2 = self._make_activation_layer(act_specs[last_idx], name="act_last_group_2")
                if act1 is not None: 
                    x1 = act1(x1)
                if act2 is not None:
                    x2 = act2(x2)

            # Optionally apply per-group dropout
            if cfg.separate_last_group_dropout:
                # group_dropout_rates can be scalar or [d1, d2]
                if isinstance(cfg.group_dropout_rates, (list, tuple)):
                    if len(cfg.group_dropout_rates) != 2:
                        raise ValueError("group_dropout_rates must be scalar or a 2-element sequence [d1, d2].")
                    d1, d2 = float(cfg.group_dropout_rates[0]), float(cfg.group_dropout_rates[1])
                else:
                    d1 = d2 = float(cfg.group_dropout_rates)
                if d1 > 0.0:
                    x1 = tf.keras.layers.Dropout(d1, name="dropout_last_group_1")(x1)
                if d2 > 0.0:
                    x2 = tf.keras.layers.Dropout(d2, name="dropout_last_group_2")(x2)

            # Group-specific outputs (same activation for both groups)
            out_act1 = self._make_activation_layer(cfg.output_activation, name="act_output_group_1")
            out_act2 = self._make_activation_layer(cfg.output_activation, name="act_output_group_2")

            g1_units, g2_units = int(cfg.output_groups[0]), int(cfg.output_groups[1])
            out1 = tf.keras.layers.Dense(
                g1_units,
                activation=None,
                kernel_initializer=out_kinit,
                bias_initializer=out_binit,
                name="output_group_1",
            )(x1)
            out2 = tf.keras.layers.Dense(
                g2_units,
                activation=None,
                kernel_initializer=out_kinit,
                bias_initializer=out_binit,
                name="output_group_2",
            )(x2)

            if out_act1 is not None:                
                out1 = out_act1(out1)
            if out_act2 is not None:
                out2 = out_act2(out2)

            outputs = tf.keras.layers.Concatenate(name="output")([out1, out2])

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ANN")

        # ----- Compile -----
        if isinstance(cfg.optimizer, str):
            optimizer = tf.keras.optimizers.get(cfg.optimizer)
            if hasattr(optimizer, "learning_rate"):
                optimizer.learning_rate = cfg.lr
        elif isinstance(cfg.optimizer, tf.keras.optimizers.Optimizer):
            optimizer = cfg.optimizer
            # override its LR with cfg.lr
            if hasattr(cfg.optimizer, "learning_rate"):
                cfg.optimizer.learning_rate = cfg.lr
        else:
            # fallback: instantiate Adam with cfg.lr
            optimizer = tf.keras.optimizers.Adam(cfg.lr)

        model.compile(optimizer=optimizer, loss=cfg.loss, metrics=list(cfg.metrics))
        return model
        