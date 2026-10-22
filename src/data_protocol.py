"""Manuscript data partitions and training-only condition statistics.

Replica IDs are one-based. Frame indices identify chronological order within
each replica; arrays may be interleaved, but duplicate frame IDs are rejected.
The excluded replica participates in the fixed TICA fit, not optimization.
"""

from pathlib import Path
import hashlib
import json

import numpy as np


DEVELOPMENT_REPLICAS = {"jnj2113": (1, 3, 4), "luna18": (1, 2, 3)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(replica_ids, frame_indices, system,
                frames_per_replica=12500, validation_tail=2500):
    """Return disjoint original-row indices, ordered by replica and time."""
    if system not in DEVELOPMENT_REPLICAS:
        raise ValueError(f"Unknown system: {system}")
    replica_ids = np.asarray(replica_ids)
    frame_indices = np.asarray(frame_indices)
    if (replica_ids.ndim != 1 or frame_indices.shape != replica_ids.shape
            or not np.issubdtype(replica_ids.dtype, np.integer)
            or not np.issubdtype(frame_indices.dtype, np.integer)):
        raise ValueError("replica_ids and frame_indices must be equal-length integer vectors")
    if set(replica_ids.tolist()) != {1, 2, 3, 4}:
        raise ValueError("Exactly replicas 1, 2, 3, and 4 are required")
    if not 0 < validation_tail < frames_per_replica:
        raise ValueError("validation_tail must be inside each replica")
    split = {"train": [], "validation": [], "excluded": []}
    for replica in (1, 2, 3, 4):
        rows = np.flatnonzero(replica_ids == replica)
        if len(rows) != frames_per_replica:
            raise ValueError(f"Replica {replica}: {len(rows)} frames; expected {frames_per_replica}")
        times = frame_indices[rows]
        if len(np.unique(times)) != len(times):
            raise ValueError(f"Duplicate frame index in replica {replica}")
        rows = rows[np.argsort(times, kind="stable")]
        if replica in DEVELOPMENT_REPLICAS[system]:
            split["train"].extend(rows[:-validation_tail])
            split["validation"].extend(rows[-validation_tail:])
        else:
            split["excluded"].extend(rows)
    return {name: np.asarray(rows, dtype=np.int64) for name, rows in split.items()}


def standardize_conditions(scores, train_indices):
    """Fit mean/std on training rows and transform all rows with frozen stats."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("TICA scores must be a finite [n_frames, 2] array")
    train_scores = scores[train_indices]
    mean, std = train_scores.mean(axis=0), train_scores.std(axis=0)
    if np.any(std <= 0):
        raise ValueError("Each TICA condition must have nonzero training-set variance")
    return ((scores - mean) / std).astype(np.float32), mean, std


def nearest_centers(scores, centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("State centers must be a finite [n_states, 2] array")
    return np.square(np.asarray(scores)[:, None, :] - centers[None, :, :]).sum(-1).argmin(-1)


def verify_manifest(manifest_path, supplied_files):
    """Bind the trainer's actual input files to one prepared split."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "highstategen.trajectory_split.v1":
        raise ValueError("Unsupported split manifest")
    if manifest.get("counts") != {"train": 30000, "validation": 7500, "excluded": 12500}:
        raise ValueError("Manifest does not implement the manuscript partition")
    for role, actual_path in supplied_files.items():
        if not actual_path:
            raise ValueError(f"Missing manuscript input: {role}")
        expected = manifest["outputs"][role]["sha256"]
        if sha256_file(actual_path) != expected:
            raise ValueError(f"Input {role} does not match the prepared split manifest")
    index_entry = manifest["outputs"]["split_indices"]
    index_path = path.parent / index_entry["file"]
    if sha256_file(index_path) != index_entry["sha256"]:
        raise ValueError("Split indices were changed after preparation")
    with np.load(index_path, allow_pickle=False) as index_data:
        expected_split = build_split(index_data["replica_ids"], index_data["frame_indices"], manifest["system"])
        for role, expected in expected_split.items():
            if not np.array_equal(index_data[role], expected):
                raise ValueError(f"Invalid {role} membership in split manifest")
    return {"manifest_sha256": sha256_file(path), "system": manifest["system"],
            "counts": manifest["counts"], "development_replicas": manifest["development_replicas"]}
