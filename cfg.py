# pdfANN/cfg.py
from dataclasses import asdict
from pathlib import Path
import json

from ann.model import ANNConfig
from ann.training import TrainingConfig

'''
Full configurations of ANN architecture and training setup, saved to ./config/
See docs/config_guide.md for meanings of ANNConfig / TrainingConfig fields
'''

ann_cfg = ANNConfig(
        input_dim=4,                 
        hidden_layers=[32, 64, 64, 64],
        activations={"name": "leaky_relu", "alpha": 0.01},        
        use_batchnorm=[False, True, True, True],
        
        bn_momentum=0.9,
        bn_epsilon=1e-5,
        dropout_rates=[0.0, 0.2, 0.2, 0.2],
        
        kernel_regularizers={"l2": 0.01},
        kernel_initializers="normal",
        bias_initializers="zeros",
        
        output_dim=100,                 
        output_activation="softmax",
        output_kernel_initializer="glorot_uniform",
        output_bias_initializer="zeros",

        output_groups=[45,55],
        last_hidden_split=32,

        separate_last_group_bn=True,
        separate_last_group_dropout=True,
        group_dropout_rates=0.2,

        optimizer="adam",
        lr=1e-2,
        loss="binary_crossentropy",
        metrics=["mse", "mae"],
    )

train_cfg = TrainingConfig(
        save_best_only=True,
        save_last=True,
        save_weights_only=True,
        save_config=True,
        resume_from=None,
        base_cfg=None,
    
        X_path="data/X.npy",          
        y_path="data/y.npy",
        batch_size=256,
        epochs=500,
        validation_split=0.2,
        shuffle_buffer=10000,
        prefetch=0,
        cache_in_memory=True,

        use_sklearn_standard_scaler=True,
        scaler_pathname="scaler.pkl",
        outlier_removal=True,
        outlier_pathname="outlier_index.pkl",
    
        early_stopping_patience=20,   
        reduce_lr_on_plateau=True,
        reduce_lr_patience=10,
        monitor="val_loss",
        mode="min",
        tensorboard=True,

        seed=42,
        mixed_precision=False,
    )

out_dir = Path("./config")
(out_dir / "ann_base.json").write_text(json.dumps(asdict(ann_cfg), indent=2))
(out_dir / "train_base.json").write_text(json.dumps(asdict(train_cfg), indent=2))



print(f"✓ Wrote {'ann_base.json'}")
print(f"✓ Wrote {'train_base.json'}")

