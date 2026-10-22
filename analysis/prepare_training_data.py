#!/usr/bin/env python3
"""Prepare the manuscript split from aligned physical coordinates and fixed TICA scores."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_protocol import (DEVELOPMENT_REPLICAS, build_split, nearest_centers,
                           sha256_file, standardize_conditions)


def prepare(args):
    inputs = {name: Path(getattr(args, name)).resolve() for name in
              ("coords_nm", "replica_ids", "frame_indices", "tica_scores", "state_centers")}
    coords = np.load(inputs["coords_nm"], mmap_mode="r", allow_pickle=False)
    replica_ids = np.load(inputs["replica_ids"], allow_pickle=False)
    frame_indices = np.load(inputs["frame_indices"], allow_pickle=False)
    scores = np.load(inputs["tica_scores"], allow_pickle=False)
    centers = np.load(inputs["state_centers"], allow_pickle=False)
    split = build_split(replica_ids, frame_indices, args.system)
    if coords.ndim != 3 or coords.shape[0] != 50000 or coords.shape[2] != 3:
        raise ValueError("Aligned physical coordinates must have shape [50000, n_atoms, 3], in nm")
    if scores.shape != (50000, 2):
        raise ValueError("TICA scores must have shape [50000, 2] in the same frame order")
    expected_states = 5 if args.system == "jnj2113" else 3
    if centers.shape != (expected_states, 2):
        raise ValueError(f"Supply the fixed, ordered {expected_states} state centers; do not refit them")
    for start in range(0, len(coords), 256):
        if not np.isfinite(coords[start:start + 256]).all():
            raise ValueError("Nonfinite input coordinates")
    normalized, mean, std = standardize_conditions(scores, split["train"])
    labels = nearest_centers(scores, centers)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    outputs = {}

    def record(role, path):
        outputs[role] = {"file": path.name, "sha256": sha256_file(path)}

    for role, indices in split.items():
        filename = {"train": "train_set.npz", "validation": "val_set.npz",
                    "excluded": "excluded_set.npz"}[role]
        path = output / filename
        # Physical coordinates use identity atom-level normalization. The frame
        # dataset subsequently fits its scalar C-alpha scale on training only.
        np.savez_compressed(path, coords=np.asarray(coords[indices], dtype=np.float32),
                            mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32),
                            source_rows=indices, replica_ids=replica_ids[indices],
                            frame_indices=frame_indices[indices], units="nm")
        record(f"{role}_data", path)
        path = output / f"condition_{role}_normalized.npy"
        np.save(path, normalized[indices])
        record(f"{role}_conditions", path)
    path = output / "basin_labels_train.npy"
    np.save(path, labels[split["train"]])
    record("train_labels", path)
    path = output / "condition_stats.npz"
    np.savez(path, mean=mean, std=std, fitted_on="training_frames_only")
    record("condition_stats", path)
    path = output / "state_centers.npy"
    np.save(path, centers)
    record("state_centers", path)
    path = output / "split_indices.npz"
    np.savez(path, **split, replica_ids=replica_ids, frame_indices=frame_indices)
    record("split_indices", path)
    manifest = {
        "schema": "highstategen.trajectory_split.v1", "system": args.system,
        "development_replicas": list(DEVELOPMENT_REPLICAS[args.system]),
        "excluded_replica": 2 if args.system == "jnj2113" else 4,
        "frames_per_replica": 12500, "validation_tail_per_development_replica": 2500,
        "counts": {name: len(rows) for name, rows in split.items()},
        "tica_fit_scope": "complete 50000-frame ensemble, lagged pairs within replicas only",
        "excluded_scope": "excluded from network optimization and checkpoint selection",
        "condition_statistics_scope": "30000 training frames only",
        "coordinate_units": "nm", "state_center_order": "input row order (M1, M2, ...)",
        "sources": {name: {"file": path.name, "sha256": sha256_file(path)}
                    for name, path in inputs.items()}, "outputs": outputs,
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "counts": manifest["counts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(DEVELOPMENT_REPLICAS))
    for flag in ("coords-nm", "replica-ids", "frame-indices", "tica-scores", "state-centers"):
        parser.add_argument("--" + flag, required=True, help="Input .npy array")
    parser.add_argument("--output-dir", required=True, help="New directory; existing data are never overwritten")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
