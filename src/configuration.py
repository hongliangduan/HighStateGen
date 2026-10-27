"""Translate the manuscript configuration into executable trainer arguments."""

import json
import os
from pathlib import Path


PROTOCOL_MODEL_SETTINGS = {
    "manuscript_20261019": {"translation_scale_mode": "radial_rms",
        "equivariant_translations": True, "topology_graph_scope": "peptide"},
    "jnj2113_45000_v1": {"translation_scale_mode": "coordinate_rms",
        "equivariant_translations": False, "topology_graph_scope": "complex"},
}

MODEL_KEYS = ("c_s", "c_z", "n_layers", "n_heads", "condition_dim", "dropout",
              "n_topo_heads", "khop", "slow_feat_dim", "sigma_min", "cyclic_peptide",
              "equivariant_translations", "topology_graph_scope")
TRAIN_KEYS = ("batch_size", "lr", "weight_decay", "warmup_ratio", "grad_clip",
              "cfg_dropout", "ema_decay", "patience", "basin_boost",
              "translation_scale_mode")


def training_defaults(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol") not in ("manuscript_20261019", "jnj2113_45000_v1"):
        raise ValueError("Unknown JNJ-2113 training configuration")
    if config.get("system") != "IL-23R–JNJ-2113":
        raise ValueError("This training profile supports IL-23R–JNJ-2113 only")
    defaults = {key: config["model"][key] for key in MODEL_KEYS}
    defaults.update({key: config["training"][key] for key in TRAIN_KEYS})
    defaults.update(config["training"]["loss_weights"])
    for key, value in PROTOCOL_MODEL_SETTINGS[config["protocol"]].items():
        if defaults[key] != value:
            raise ValueError(f"Configuration {key} must be {value!r} for {config['protocol']}")
    defaults["epochs"] = config["training"].get("maximum_epochs")
    defaults["seed"] = config["training"].get("seed")
    defaults["manuscript_protocol"] = config["protocol"]
    defaults["finetune_extra_epochs"] = 0
    defaults["reset_scheduler"] = False
    return defaults


def sampling_defaults(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = config["sampling"]
    defaults = {key: settings[key] for key in ("num_steps", "cfg_scale", "cfg_warmup_ratio")}
    defaults.update(settings["geometry_guidance"])
    defaults["tica_guidance_weight"] = settings["tica_gradient_guidance_weight"]
    return defaults


def validate_manuscript_arguments(args):
    if not getattr(args, "manuscript_protocol", None):
        return
    expected_settings = PROTOCOL_MODEL_SETTINGS[args.manuscript_protocol]
    for key, value in expected_settings.items():
        if getattr(args, key, None) != value:
            raise ValueError(f"Training {key} must be {value!r} for {args.manuscript_protocol}")
    if args.epochs is None or args.epochs < 1 or args.seed is None:
        raise ValueError("Supply --epochs and --seed for this training run")
    if args.manuscript_protocol == "manuscript_20261019" and (args.resume or args.finetune_extra_epochs):
        raise ValueError("The 20261019 profile uses random initialization")
    if args.resume and not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    if args.finetune_extra_epochs < 0 or (args.finetune_extra_epochs and not args.resume):
        raise ValueError("Extra epochs require an existing --resume checkpoint")
    if not args.val_data or not args.split_manifest:
        raise ValueError("Manuscript training requires --val_data and --split_manifest")
    if not args.slow_feat_path or not args.slow_feat_val_path or not args.train_basin_labels:
        raise ValueError("Supply prepared train/validation TICA conditions and training basin labels")
    if sum(bool(getattr(args, flag)) for flag in (
            "disable_tica_conditioning", "disable_topology_bias", "disable_interface_attention")) > 1:
        raise ValueError("The reported component ablations remove one component at a time")
