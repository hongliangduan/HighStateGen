"""Bind model semantics and inference inputs to their training provenance."""

from pathlib import Path
import warnings
from data_protocol import sha256_file
from configuration import PROTOCOL_MODEL_SETTINGS


def validate_checkpoint_protocol(model_args, run_metadata, allow_legacy=False):
    protocol = model_args.get("manuscript_protocol")
    legacy = protocol not in ("manuscript_20261019", "jnj2113_45000_v1")
    if legacy:
        if not allow_legacy:
            raise ValueError("Archived/unversioned checkpoint: pass --allow_legacy_checkpoint for an explicitly labelled legacy run. This is not a manuscript-protocol checkpoint.")
        warnings.warn("Legacy checkpoint: retain its original scale, vector-head and topology semantics; this run does not verify the revised manuscript protocol.", stacklevel=2)
        return "archived_unverified"
    split = run_metadata.get("split") or {}
    current = protocol == "jnj2113_45000_v1"
    expected = ({"train": 45000, "validation": 5000, "excluded": 0} if current
                else {"train": 30000, "validation": 7500, "excluded": 12500})
    development = [1, 2, 3, 4] if current else [1, 3, 4]
    if (split.get("counts") != expected or split.get("system") != "jnj2113"
            or split.get("development_replicas") != development):
        raise ValueError("Checkpoint has no verified JNJ-2113 manuscript split")
    expected_settings = PROTOCOL_MODEL_SETTINGS[protocol]
    if any(model_args.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Checkpoint model semantics disagree with the manuscript configuration")
    if not current and (model_args.get("resume") or model_args.get("finetune_extra_epochs", 0)):
        raise ValueError("A resumed checkpoint cannot be labelled a from-scratch manuscript run")
    return "manuscript_configured_run"


def verify_inference_inputs(run_metadata, data, topology, condition_stats=None):
    expected = run_metadata.get("inference_input_sha256")
    if not expected:
        raise ValueError("No bound inference metadata in this checkpoint; recover the matching files and metadata, or use an explicitly labelled legacy checkpoint")
    actual = {"train_data": data, "topology": topology}
    if condition_stats is not None:
        actual["condition_stats"] = condition_stats
    for role, path in actual.items():
        if not path or not Path(path).is_file() or sha256_file(path) != expected.get(role):
            raise ValueError(f"Inference input {role} differs from the checkpoint's training artifact")
