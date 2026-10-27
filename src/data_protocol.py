"""Manuscript data partitions and training-only condition statistics.

Replica IDs are one-based. Frame indices identify chronological order within
each replica; arrays may be interleaved, but duplicate frame IDs are rejected.
JNJ-2113 uses the final 1,250 frames of each replica for validation.
"""

from pathlib import Path
import hashlib
import json

import numpy as np


DEVELOPMENT_REPLICAS = {"jnj2113": (1, 2, 3, 4), "luna18": (1, 2, 3)}
VALIDATION_TAIL = {"jnj2113": 1250, "luna18": 2500}
SPLIT_COUNTS = {"jnj2113": {"train": 45000, "validation": 5000, "excluded": 0},
                "luna18": {"train": 30000, "validation": 7500, "excluded": 12500}}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(replica_ids, frame_indices, system,
                frames_per_replica=12500, validation_tail=None):
    """Return disjoint original-row indices, ordered by replica and time."""
    if system not in DEVELOPMENT_REPLICAS:
        raise ValueError(f"Unknown system: {system}")
    if validation_tail is None:
        validation_tail = VALIDATION_TAIL[system]
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


def validate_tica_metadata(metadata, scores_path, system, centers_path=None):
    """Reject old concatenated/peptide-inclusive fits in manuscript preparation."""
    details = metadata.get("tica_fit_details") or {}
    expected_align = 294 if system == "jnj2113" else 169
    expected_features = 307 if system == "jnj2113" else 180
    if (metadata.get("reduction_method") != "tica" or metadata.get("lagtime") != 5
            or details.get("trajectory_lengths") != [12500] * 4
            or details.get("cross_replica_pairs") != 0
            or metadata.get("n_align_ca") != expected_align
            or metadata.get("n_feature_ca") != expected_features
            or metadata.get("n_frames") != 50000):
        raise ValueError("TICA metadata do not establish receptor-only alignment and within-replica fitting for the manuscript protocol")
    expected = metadata.get("fixed_artifact_sha256", {}).get("projection_scores.npy")
    if not expected or sha256_file(scores_path) != expected:
        raise ValueError("TICA scores are not bound to this fixed transform")
    if centers_path is not None:
        centers_hash = metadata.get("fixed_artifact_sha256", {}).get("state_centers.npy")
        if not centers_hash or sha256_file(centers_path) != centers_hash:
            raise ValueError("Ordered state centers are not bound to this fixed transform")
    return {"alignment_ca_count": expected_align, "lagtime": 5,
            "trajectory_lengths": [12500] * 4, "cross_replica_pairs": 0,
            "score_sha256": expected}


def verify_manifest(manifest_path, supplied_files):
    """Bind the trainer's actual input files to one prepared split."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "highstategen.trajectory_split.v2":
        raise ValueError("Unsupported split manifest")
    if manifest.get("counts") != SPLIT_COUNTS.get(manifest.get("system")):
        raise ValueError("Manifest does not implement the manuscript partition")
    if not manifest.get("tica_protocol"):
        raise ValueError("Manifest is missing verified TICA fit provenance")
    tica_entry = manifest["outputs"]["tica_metadata"]
    if sha256_file(path.parent / tica_entry["file"]) != tica_entry["sha256"]:
        raise ValueError("TICA fit metadata were changed after preparation")
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
    if manifest.get("development_replicas") != list(DEVELOPMENT_REPLICAS[manifest["system"]]):
        raise ValueError("Development replicas disagree with the supplied partition")
    return {"manifest_sha256": sha256_file(path), "system": manifest["system"],
            "counts": manifest["counts"], "development_replicas": manifest["development_replicas"]}
