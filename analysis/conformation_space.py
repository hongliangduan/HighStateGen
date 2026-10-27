#!/usr/bin/env python3
"""
Build and use a 2D conformational projection space from AMBER trajectories.

This tool supports:
1) fit:
   - load one or multiple trajectories
   - remove water/ions
   - align all frames to receptor backbone CA reference
   - extract CA coordinates and flatten
   - dimensionality reduction by method={auto,tica,pca}
     * auto/tica: regularized TICA with lagged pairs formed within each trajectory
   - save model and projection artifacts
   - plot scatter (time-colored) and FES contour map
   - detect and annotate minima (M1-M5 by default)

2) project:
   - project new structures into the saved 2D space

Required runtime stack:
  MDTraj, NumPy, matplotlib
Optional:
  joblib (model serialization), scikit-learn (legacy PCA backend),
  scipy (Gaussian smoothing)
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from tica_core import fit_tica

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

BOLTZ_KCAL_PER_MOL_K = 0.0019872041


def _apply_publication_style() -> None:
    """
    Global matplotlib style for journal-ready figures.

    Notes:
    - Prefer Times New Roman; fallback to Times/DejaVu Serif when unavailable.
    - Keep defaults deterministic and readable for both PNG/PDF outputs.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=5)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)


_apply_publication_style()


@dataclass
class NumpyPCAModel:
    """Simple PCA model fallback when sklearn is unavailable."""

    mean_: np.ndarray
    components_: np.ndarray  # (k, d)
    explained_variance_ratio_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_[None, :]
        return Xc @ self.components_.T


@dataclass
class NumpyTICAModel:
    """Pure NumPy TICA fallback model when deeptime is unavailable."""

    mean_: np.ndarray
    components_: np.ndarray  # (d, k)
    eigenvalues_: np.ndarray
    lagtime_: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_[None, :]
        return Xc @ self.components_


def _comma_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_sklearn_pca_class():
    try:
        from sklearn.decomposition import PCA as SklearnPCA
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import scikit-learn PCA. "
            "Please install/fix compatible numpy/scipy/scikit-learn versions."
        ) from exc
    return SklearnPCA




def _load_traj(path: Path, topology: Optional[Path], stride: int = 1) -> md.Trajectory:
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".cif", ".mmcif", ".gro"}:
        return md.load(str(path), stride=stride)
    if topology is None:
        raise ValueError(f"Topology is required to load trajectory file: {path}")
    return md.load(str(path), top=str(topology), stride=stride)


def _build_non_solvent_atom_mask(topology: md.Topology, exclude_resnames: Sequence[str]) -> np.ndarray:
    exclude_set = {x.upper() for x in exclude_resnames}
    keep = []
    for atom in topology.atoms:
        res = atom.residue
        if res.is_water:
            keep.append(False)
            continue
        if res.name.upper() in exclude_set:
            keep.append(False)
            continue
        keep.append(True)
    return np.asarray(keep, dtype=bool)


def _auto_receptor_chain_and_align_indices(topology: md.Topology) -> Tuple[int, np.ndarray]:
    best_chain = None
    best_indices = None
    for chain in topology.chains:
        ca_idx = [
            atom.index
            for atom in chain.atoms
            if atom.name == "CA" and atom.residue.is_protein
        ]
        if len(ca_idx) == 0:
            continue
        if best_indices is None or len(ca_idx) > len(best_indices):
            best_indices = ca_idx
            best_chain = chain.index
    if best_indices is None:
        raise ValueError("Could not auto-detect receptor chain with protein CA atoms.")
    return int(best_chain), np.asarray(best_indices, dtype=np.int64)


def _select_align_indices(
    topology: md.Topology,
    align_selection: Optional[str],
    receptor_chain: Optional[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {}

    if align_selection:
        idx = topology.select(align_selection)
        if len(idx) == 0:
            raise ValueError(f"Alignment selection matched 0 atoms: {align_selection}")
        meta["align_mode"] = "selection"
        meta["align_selection"] = align_selection
        return np.asarray(idx, dtype=np.int64), meta

    if receptor_chain is not None:
        sel = f"protein and backbone and name CA and chainid {receptor_chain}"
        idx = topology.select(sel)
        if len(idx) == 0:
            raise ValueError(
                f"Alignment selection matched 0 atoms for receptor chain {receptor_chain}"
            )
        meta["align_mode"] = "receptor_chain"
        meta["receptor_chain"] = int(receptor_chain)
        meta["align_selection"] = sel
        return np.asarray(idx, dtype=np.int64), meta

    chain_id, idx = _auto_receptor_chain_and_align_indices(topology)
    meta["align_mode"] = "auto_largest_protein_chain"
    meta["receptor_chain"] = int(chain_id)
    meta["align_selection"] = f"protein and backbone and name CA and chainid {chain_id}"
    return idx, meta


def _select_feature_ca_indices(topology: md.Topology, feature_selection: Optional[str]) -> np.ndarray:
    sel = feature_selection or "name CA"
    idx = topology.select(sel)
    if len(idx) == 0:
        raise ValueError(f"Feature selection matched 0 atoms: {sel}")
    return np.asarray(idx, dtype=np.int64)


def _save_model(model: object, out_path: Path) -> str:
    if joblib is not None:
        joblib.dump(model, out_path)
        return "joblib"
    with out_path.open("wb") as f:
        pickle.dump(model, f)
    return "pickle"


def _load_model(path: Path) -> object:
    # Historical files were pickled under both script module names. This
    # compatibility mapping is for trusted author-supplied transforms only.
    import sys
    import __main__
    module = sys.modules[__name__]
    sys.modules.setdefault("pca_conformation_space", module)
    for name in ("NumpyPCAModel", "NumpyTICAModel"):
        if not hasattr(__main__, name):
            setattr(__main__, name, globals()[name])
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


def _extract_flattened_features(traj: md.Trajectory, ca_indices: np.ndarray) -> np.ndarray:
    xyz = traj.xyz[:, ca_indices, :]
    return xyz.reshape(xyz.shape[0], -1)


def _transform_with_model(model: object, X: np.ndarray) -> np.ndarray:
    out = model.transform(X)
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            raise ValueError("Model transform returned an empty sequence.")
        out = out[0]
    out = np.asarray(out)
    if out.ndim == 1:
        out = out[:, None]
    return out


def _fit_pca(X: np.ndarray, n_components: int, random_state: int) -> Tuple[object, np.ndarray, Dict[str, object]]:
    try:
        PCAClass = _get_sklearn_pca_class()
        pca = PCAClass(n_components=n_components, random_state=random_state)
        scores = pca.fit_transform(X)
        info = {
            "reduction_method": "pca",
            "backend": "sklearn",
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        }
        return pca, np.asarray(scores), info
    except Exception as exc:
        # Robust fallback: NumPy SVD PCA
        print(f"[WARN] sklearn PCA unavailable ({exc}); fallback to NumPy SVD PCA")
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        Xc = X - mean[None, :]
        _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
        comps = vt[:n_components, :]
        scores = Xc @ comps.T
        var = (svals ** 2) / max(1, (X.shape[0] - 1))
        var_ratio = var / (var.sum() + 1e-12)
        model = NumpyPCAModel(mean_=mean, components_=comps, explained_variance_ratio_=var_ratio[:n_components])
        info = {
            "reduction_method": "pca",
            "backend": "numpy_svd",
            "explained_variance_ratio": np.asarray(model.explained_variance_ratio_, dtype=float),
        }
        return model, np.asarray(scores), info




def _fit_tica_numpy(
    X: np.ndarray,
    n_components: int,
    lagtime: int,
    regularization: float = 1e-6,
    trajectory_lengths=None,
) -> Tuple[object, np.ndarray, Dict[str, object]]:
    """
    Regularized reversible TICA with independent-trajectory covariances.
    The symmetric generalized eigenproblem is solved by whitening C_0.
    """
    lengths = trajectory_lengths if trajectory_lengths is not None else [len(X)]
    return fit_tica(X, lengths, lagtime, n_components, regularization)




def _fit_reduction(
    X: np.ndarray,
    method: str,
    n_components: int,
    random_state: int,
    tica_lag: int,
    trajectory_lengths=None,
) -> Tuple[object, np.ndarray, Dict[str, object]]:
    method = method.lower()
    if method not in {"auto", "tica", "pca"}:
        raise ValueError(f"Unsupported method: {method}")

    if method in {"auto", "tica"}:
        return _fit_tica_numpy(X, n_components=n_components, lagtime=tica_lag,
                               trajectory_lengths=trajectory_lengths)

    return _fit_pca(X, n_components=n_components, random_state=random_state)


def _format_axis_label(base: str, explained_ratio: Optional[np.ndarray], index: int) -> str:
    if explained_ratio is None or index >= len(explained_ratio):
        return base
    return f"{base} ({float(explained_ratio[index]) * 100:.2f}% variance)"


def _plot_projection_scores(
    scores: np.ndarray,
    component_labels: Sequence[str],
    explained_ratio: Optional[np.ndarray],
    color_values: np.ndarray,
    color_label: str,
    cmap: str,
    out_png: Path,
    out_pdf: Optional[Path],
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=dpi)
    sc = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=color_values,
        cmap=cmap,
        s=14,
        alpha=0.9,
        linewidths=0.0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12, width=1.2)

    ax.set_xlabel(_format_axis_label(component_labels[0], explained_ratio, 0), fontsize=14)
    ax.set_ylabel(_format_axis_label(component_labels[1], explained_ratio, 1), fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()

    fig.savefig(out_png, dpi=dpi)
    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)


def _compute_fes(
    points2d: np.ndarray,
    bins: int,
    sigma: float,
    temperature: float,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D histogram probability and free energy surface.

    Returns:
      x_centers, y_centers, probability, free_energy, smoothed_probability
    """
    x = points2d[:, 0]
    y = points2d[:, 1]

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    prob = hist / max(hist.sum(), 1.0)

    prob_smooth = prob.copy()
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            prob_smooth = gaussian_filter(prob_smooth, sigma=sigma, mode="nearest")
        except Exception as exc:
            print(f"[WARN] scipy gaussian_filter unavailable ({exc}); skip smoothing")

    prob_smooth = prob_smooth / max(prob_smooth.sum(), 1e-30)

    kBT = BOLTZ_KCAL_PER_MOL_K * float(temperature)
    free = -kBT * np.log(prob_smooth + float(epsilon))
    free = free - np.nanmin(free)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, prob, free, prob_smooth


def _find_local_minima(
    free_energy: np.ndarray,
    n_minima: int,
    min_separation_bins: int,
) -> List[Tuple[int, int, float]]:
    """
    Detect local minima on 2D grid and return top-N separated minima.

    Returns list of (i, j, G) sorted by increasing G.
    """
    nx, ny = free_energy.shape
    candidates: List[Tuple[int, int, float]] = []

    # Always include the global minimum (can lie on boundaries).
    gi, gj = np.unravel_index(np.nanargmin(free_energy), free_energy.shape)
    gmin = float(free_energy[gi, gj])
    if np.isfinite(gmin):
        candidates.append((int(gi), int(gj), gmin))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            v = float(free_energy[i, j])
            if not np.isfinite(v):
                continue
            block = free_energy[i - 1 : i + 2, j - 1 : j + 2]
            if np.all(v <= block) and np.any(v < block):
                candidates.append((i, j, v))

    if not candidates:
        i0, j0 = np.unravel_index(np.argmin(free_energy), free_energy.shape)
        return [(int(i0), int(j0), float(free_energy[i0, j0]))]

    candidates.sort(key=lambda t: t[2])

    chosen: List[Tuple[int, int, float]] = []
    min_sep = float(max(0, min_separation_bins))
    for i, j, g in candidates:
        keep = True
        for ic, jc, _ in chosen:
            d = np.sqrt((i - ic) ** 2 + (j - jc) ** 2)
            if d < min_sep:
                keep = False
                break
        if keep:
            chosen.append((i, j, g))
        if len(chosen) >= n_minima:
            break

    return chosen


def _parse_xy_pair(text: Optional[str], arg_name: str) -> Optional[Tuple[float, float]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{arg_name} must be in 'x,y' format, got: {text}")
    try:
        x = float(parts[0])
        y = float(parts[1])
    except Exception as exc:
        raise ValueError(f"{arg_name} contains non-numeric value(s): {text}") from exc
    return (x, y)


def _resolve_minima_separation_bins(user_value: Optional[int], bins: int) -> int:
    if user_value is not None:
        return int(max(0, user_value))
    # Emergency default: force a large basin-to-basin distance.
    return int(max(30, int(bins * 0.3)))


def _manual_minima_rows(
    manual_points: Sequence[Tuple[str, Tuple[float, float]]],
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    free_energy: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for label, (x_user, y_user) in manual_points:
        i = int(np.argmin(np.abs(x_centers - float(x_user))))
        j = int(np.argmin(np.abs(y_centers - float(y_user))))
        g = float(free_energy[i, j])
        rows.append(
            {
                "label": str(label),
                "x": float(x_user),
                "y": float(y_user),
                "G_kcal_mol": g,
                "bin_i": i,
                "bin_j": j,
                "manual": True,
            }
        )
    return rows


def _save_minima_csv(
    minima: Sequence[Tuple[int, int, float]],
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    out_csv: Path,
) -> List[Dict[str, float]]:
    rows = []
    for idx, (i, j, g) in enumerate(minima, start=1):
        rows.append(
            {
                "label": f"M{idx}",
                "x": float(x_centers[i]),
                "y": float(y_centers[j]),
                "G_kcal_mol": float(g),
                "bin_i": int(i),
                "bin_j": int(j),
                "manual": False,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "x", "y", "G_kcal_mol", "bin_i", "bin_j"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _write_minima_rows_csv(rows: Sequence[Dict[str, float]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "x", "y", "G_kcal_mol", "bin_i", "bin_j"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "G_kcal_mol": float(row["G_kcal_mol"]),
                    "bin_i": int(row["bin_i"]),
                    "bin_j": int(row["bin_j"]),
                }
            )


def _plot_fes_contour(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    free_energy: np.ndarray,
    minima_rows: Sequence[Dict[str, float]],
    component_labels: Sequence[str],
    cmap: str,
    levels: int,
    dpi: int,
    title: str,
    out_png: Path,
    out_pdf: Optional[Path],
) -> None:
    Xg, Yg = np.meshgrid(x_centers, y_centers, indexing="ij")

    emax = float(np.nanpercentile(free_energy, 99.0))
    if not np.isfinite(emax) or emax <= 0:
        emax = float(np.nanmax(free_energy))
    if not np.isfinite(emax) or emax <= 0:
        emax = 1.0
    contour_levels = np.linspace(0.0, emax, max(3, int(levels)))

    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=dpi)
    cf = ax.contourf(Xg, Yg, free_energy, levels=contour_levels, cmap=cmap, extend="max")
    ax.contour(Xg, Yg, free_energy, levels=contour_levels, colors="k", linewidths=0.35, alpha=0.35)

    for row in minima_rows:
        if bool(row.get("manual", False)):
            ax.scatter(
                row["x"],
                row["y"],
                marker="*",
                s=260,
                c="yellow",
                edgecolors="black",
                linewidths=1.2,
                zorder=6,
            )
            ax.text(
                row["x"],
                row["y"],
                f" {row['label']}",
                color="black",
                fontsize=10,
                fontweight="bold",
                zorder=7,
            )
        else:
            ax.plot(row["x"], row["y"], marker="+", color="black", markersize=12, markeredgewidth=2.0)
            ax.text(row["x"], row["y"], f" {row['label']}", color="black", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Free energy (kcal/mol)", fontsize=14)
    cbar.ax.tick_params(labelsize=12, width=1.2)

    ax.set_xlabel(component_labels[0], fontsize=14)
    ax.set_ylabel(component_labels[1], fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.4)
    _style_axes(ax)
    fig.tight_layout()

    fig.savefig(out_png, dpi=dpi)
    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)


def _fit(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    topology_path = Path(args.topology).resolve()
    traj_paths = [Path(x).resolve() for x in args.traj]
    exclude_resnames = _comma_list(args.exclude_resnames)

    print("=== Conformation-Space Fit: Loading trajectories ===")
    print(f"Topology: {topology_path}")
    print(f"Trajectories ({len(traj_paths)}):")
    for p in traj_paths:
        print(f"  - {p}")

    traj0 = _load_traj(traj_paths[0], topology_path, stride=args.stride)
    keep_mask = _build_non_solvent_atom_mask(traj0.topology, exclude_resnames)
    keep_indices = np.where(keep_mask)[0]
    if args.export_training_inputs and len(keep_indices) != traj0.n_atoms:
        raise ValueError("For training export, supply dry trajectories and the matching dry topology; atom stripping would change indexing")
    print(f"Keep atoms after water/ion removal: {len(keep_indices)} / {traj0.n_atoms}")

    filtered_trajs = []
    frame_offsets = []
    total_frames = 0
    for path in traj_paths:
        t = _load_traj(path, topology_path, stride=args.stride)
        if t.n_atoms != traj0.n_atoms:
            raise ValueError(
                f"Atom count mismatch among trajectories: {path} has {t.n_atoms}, expected {traj0.n_atoms}"
            )
        tf = t.atom_slice(keep_indices)
        filtered_trajs.append(tf)
        frame_offsets.append((path.name, total_frames, total_frames + tf.n_frames - 1))
        total_frames += tf.n_frames

    lengths = args.trajectory_lengths or [t.n_frames for t in filtered_trajs]
    if sum(lengths) != total_frames or any(length <= args.tica_lag for length in lengths):
        raise ValueError("Trajectory lengths must partition all loaded frames and exceed the lag")
    if args.system == "jnj2113":
        if lengths != [12500, 12500, 12500, 12500] or args.method == "pca":
            raise ValueError("JNJ-2113 protocol requires four 12500-frame replicas and TICA; for a concatenated file supply --trajectory-lengths 12500 12500 12500 12500")
        if args.receptor_chain is not None and args.align_selection is None:
            raise ValueError("AMBER chain IDs can merge receptor and peptide; supply an explicit receptor-only --align-selection")
        if args.align_selection is None:
            args.align_selection = "name CA and resid 0 to 293"
    if args.export_training_inputs and lengths != [12500, 12500, 12500, 12500]:
        raise ValueError("Training export requires four 12500-frame replicas in numerical order")
    traj = md.join(filtered_trajs, check_topology=True)
    print(f"Joined frames: {traj.n_frames}, atoms: {traj.n_atoms}")

    align_indices, align_meta = _select_align_indices(
        traj.topology, args.align_selection, args.receptor_chain
    )
    if args.system == "jnj2113":
        expected_align = np.asarray([a.index for a in traj.topology.atoms
                                     if a.name == "CA" and 0 <= a.residue.index <= 293])
        if len(expected_align) != 294 or not np.array_equal(align_indices, expected_align):
            raise ValueError("JNJ-2113 alignment must contain exactly the 294 receptor C-alpha atoms, excluding the peptide")
    print(f"Alignment mode: {align_meta.get('align_mode')}")
    print(f"Alignment atoms: {len(align_indices)}")
    if "align_selection" in align_meta:
        print(f"Alignment selection: {align_meta['align_selection']}")

    if args.reference is None:
        reference = traj[args.reference_frame]
        reference_source = f"joined_trajectory_frame_{args.reference_frame}"
    else:
        ref_path = Path(args.reference).resolve()
        ref_raw = _load_traj(ref_path, topology_path, stride=1)
        if ref_raw.n_frames < 1:
            raise ValueError(f"Reference has no frames: {ref_path}")
        if ref_raw.n_atoms == traj0.n_atoms:
            reference = ref_raw.atom_slice(keep_indices)[0]
        elif ref_raw.n_atoms == traj.n_atoms:
            reference = ref_raw[0]
        else:
            raise ValueError(
                f"Reference atom count ({ref_raw.n_atoms}) does not match raw ({traj0.n_atoms}) or filtered ({traj.n_atoms})."
            )
        reference_source = str(ref_path)

    print("Superposing all frames to reference using receptor backbone CA...")
    traj.superpose(reference, frame=0, atom_indices=align_indices, ref_atom_indices=align_indices)

    feature_indices = _select_feature_ca_indices(traj.topology, args.feature_selection)
    print(f"Feature CA atoms: {len(feature_indices)}")
    print(f"Feature selection: {args.feature_selection or 'name CA'}")

    index_to_ca_pos = {int(a): i for i, a in enumerate(feature_indices.tolist())}
    align_ca_positions = np.asarray(
        [index_to_ca_pos[int(a)] for a in align_indices if int(a) in index_to_ca_pos],
        dtype=np.int64,
    )
    if len(align_ca_positions) != len(align_indices):
        raise ValueError(
            "Alignment CA atoms are not a subset of projection feature CA atoms. "
            "Adjust --feature-selection to include all alignment CA atoms."
        )

    X = _extract_flattened_features(traj, feature_indices)
    print(f"Input matrix shape: {X.shape} = [n_frames, n_ca*3]")

    if int(args.n_components) < 2:
        raise ValueError("--n-components must be >= 2 for 2D FES plotting.")

    model, scores, red_info = _fit_reduction(
        X,
        method=args.method,
        n_components=args.n_components,
        random_state=args.random_state,
        tica_lag=args.tica_lag,
        trajectory_lengths=lengths,
    )

    reduction_method = str(red_info.get("reduction_method", "pca"))
    backend = str(red_info.get("backend", "unknown"))
    explained = red_info.get("explained_variance_ratio")
    explained_arr = None if explained is None else np.asarray(explained, dtype=float)

    component_labels = ["TIC1", "TIC2"] if reduction_method == "tica" else ["PC1", "PC2"]

    print(f"Reduction method: {reduction_method} (backend={backend})")
    if reduction_method == "tica":
        print(f"TICA lagtime: {args.tica_lag}")
    if explained_arr is not None:
        for i, ratio in enumerate(explained_arr[:2], start=1):
            print(f"  {component_labels[i-1]} explained variance: {ratio:.6f} ({ratio*100:.2f}%)")

    # Save model and references
    model_path = out_dir / "pca_model.pkl"
    model_serializer = _save_model(model, model_path)

    ref_pdb_path = out_dir / "reference.pdb"
    reference.save_pdb(str(ref_pdb_path))

    ref_ca_path = out_dir / "reference_ca.pdb"
    reference.atom_slice(feature_indices).save_pdb(str(ref_ca_path))

    np.save(out_dir / "atom_indices.npy", feature_indices)
    np.save(out_dir / "align_atom_indices.npy", align_indices)
    np.save(out_dir / "align_ca_positions.npy", align_ca_positions)
    np.save(out_dir / "projection_scores.npy", scores)
    if args.export_training_inputs:
        np.save(out_dir / "aligned_coords_nm.npy", traj.xyz)
        np.save(out_dir / "replica_ids.npy", np.concatenate([
            np.full(length, replica, dtype=np.int64)
            for replica, length in enumerate(lengths, start=1)]))
        np.save(out_dir / "frame_indices.npy", np.concatenate([
            np.arange(length, dtype=np.int64) for length in lengths]))
    # Backward-compatible numpy score filename used by older downstream scripts.
    np.save(out_dir / "pca_scores.npy", scores)

    # Backward-compatible score CSV naming
    score_csv_path = out_dir / "pca_scores.csv"
    with score_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", component_labels[0].lower(), component_labels[1].lower()])
        for i in range(scores.shape[0]):
            writer.writerow([i, float(scores[i, 0]), float(scores[i, 1])])

    # Scatter (legacy + still useful)
    scatter_png = out_dir / f"{args.figure_prefix}.png"
    scatter_pdf = out_dir / f"{args.figure_prefix}.pdf" if args.save_pdf else None
    color_values = np.arange(scores.shape[0], dtype=float)
    _plot_projection_scores(
        scores=scores,
        component_labels=component_labels,
        explained_ratio=explained_arr,
        color_values=color_values,
        color_label="Frame index",
        cmap=args.cmap,
        out_png=scatter_png,
        out_pdf=scatter_pdf,
        dpi=args.dpi,
        title=args.title or f"{reduction_method.upper()} Projection (time-colored)",
    )

    # FES computation
    x_centers, y_centers, prob_raw, fes, prob_smooth = _compute_fes(
        points2d=np.asarray(scores[:, :2], dtype=np.float64),
        bins=args.fes_bins,
        sigma=args.fes_sigma,
        temperature=args.temperature,
        epsilon=1e-12,
    )

    manual_m1 = _parse_xy_pair(args.manual_m1, "--manual-m1")
    manual_m2 = _parse_xy_pair(args.manual_m2, "--manual-m2")
    if (manual_m1 is None) != (manual_m2 is None):
        raise ValueError("Please provide both --manual-m1 and --manual-m2, or neither.")

    minima_mode = "auto"
    effective_min_sep_bins = _resolve_minima_separation_bins(args.minima_separation_bins, args.fes_bins)
    if manual_m1 is not None and manual_m2 is not None:
        minima_mode = "manual"
        minima_rows = _manual_minima_rows(
            manual_points=[("M1", manual_m1), ("M2", manual_m2)],
            x_centers=x_centers,
            y_centers=y_centers,
            free_energy=fes,
        )
        print("[INFO] Manual minima mode enabled; auto minima detection is skipped.")
        print(
            f"[INFO] Manual points: M1=({manual_m1[0]:.4f}, {manual_m1[1]:.4f}), "
            f"M2=({manual_m2[0]:.4f}, {manual_m2[1]:.4f})"
        )
    else:
        minima = _find_local_minima(
            free_energy=fes,
            n_minima=args.n_minima,
            min_separation_bins=effective_min_sep_bins,
        )
        print(f"[INFO] Auto minima mode with effective separation = {effective_min_sep_bins} bins")
        minima_rows = []
        for idx, (i, j, g) in enumerate(minima, start=1):
            minima_rows.append(
                {
                    "label": f"M{idx}",
                    "x": float(x_centers[i]),
                    "y": float(y_centers[j]),
                    "G_kcal_mol": float(g),
                    "bin_i": int(i),
                    "bin_j": int(j),
                    "manual": False,
                }
            )

    minima_csv_path = out_dir / "minima.csv"
    _write_minima_rows_csv(minima_rows, minima_csv_path)
    np.save(out_dir / "state_centers.npy", np.asarray([[row["x"], row["y"]] for row in minima_rows], dtype=np.float64))

    fes_grid_path = out_dir / "fes_grid.npz"
    np.savez(
        fes_grid_path,
        x_centers=x_centers,
        y_centers=y_centers,
        probability=prob_raw,
        probability_smoothed=prob_smooth,
        free_energy_kcal_mol=fes,
        temperature=float(args.temperature),
    )

    fes_png = out_dir / f"{args.figure_prefix}_fes.png"
    fes_pdf = out_dir / f"{args.figure_prefix}_fes.pdf" if args.save_pdf else None
    _plot_fes_contour(
        x_centers=x_centers,
        y_centers=y_centers,
        free_energy=fes,
        minima_rows=minima_rows,
        component_labels=component_labels,
        cmap=args.fes_cmap,
        levels=args.fes_levels,
        dpi=args.dpi,
        title=args.fes_title or f"FES on {component_labels[0]}/{component_labels[1]} (kcal/mol @ {args.temperature:.1f}K)",
        out_png=fes_png,
        out_pdf=fes_pdf,
    )

    metadata = {
        "topology": str(topology_path),
        "trajectories": [str(p) for p in traj_paths],
        "frame_offsets": frame_offsets,
        "stride": int(args.stride),
        "exclude_resnames": exclude_resnames,
        "reference_source": reference_source,
        "n_frames": int(traj.n_frames),
        "n_atoms_filtered": int(traj.n_atoms),
        "n_feature_ca": int(len(feature_indices)),
        "n_align_ca": int(len(align_indices)),
        "align_meta": align_meta,
        "feature_selection": args.feature_selection or "name CA",
        "requested_method": args.method,
        "reduction_method": reduction_method,
        "backend": backend,
        "tica_fit_details": red_info if reduction_method == "tica" else None,
        "lagtime": int(args.tica_lag) if reduction_method == "tica" else None,
        "component_labels": component_labels,
        "explained_variance_ratio": (
            None if explained_arr is None else [float(x) for x in explained_arr.tolist()]
        ),
        "model_serializer": model_serializer,
        "fes": {
            "bins": int(args.fes_bins),
            "sigma_bins": float(args.fes_sigma),
            "temperature_K": float(args.temperature),
            "levels": int(args.fes_levels),
            "cmap": str(args.fes_cmap),
            "n_minima": int(args.n_minima),
            "minima_mode": minima_mode,
            "minima_separation_bins_requested": (
                None if args.minima_separation_bins is None else int(args.minima_separation_bins)
            ),
            "minima_separation_bins_effective": int(effective_min_sep_bins),
            "manual_m1": None if manual_m1 is None else [float(manual_m1[0]), float(manual_m1[1])],
            "manual_m2": None if manual_m2 is None else [float(manual_m2[0]), float(manual_m2[1])],
        },
    }
    import hashlib
    metadata["fixed_artifact_sha256"] = {
        name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        for name in ("pca_model.pkl", "reference_ca.pdb", "align_ca_positions.npy", "projection_scores.npy", "state_centers.npy")}
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Fit complete ===")
    print(f"Model file:         {model_path}")
    print(f"Reference (full):   {ref_pdb_path}")
    print(f"Reference (CA):     {ref_ca_path}")
    print(f"Feature indices:    {out_dir / 'atom_indices.npy'}")
    print(f"Align CA positions: {out_dir / 'align_ca_positions.npy'}")
    print(f"Score CSV:          {score_csv_path}")
    print(f"Scatter PNG:        {scatter_png}")
    if scatter_pdf is not None:
        print(f"Scatter PDF:        {scatter_pdf}")
    print(f"FES PNG:            {fes_png}")
    if fes_pdf is not None:
        print(f"FES PDF:            {fes_pdf}")
    print(f"FES grid:           {fes_grid_path}")
    print(f"Minima CSV:         {minima_csv_path}")
    for row in minima_rows:
        print(
            f"  {row['label']}: x={row['x']:.4f}, y={row['y']:.4f}, "
            f"G={row['G_kcal_mol']:.4f} kcal/mol"
        )


def _project(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    model_dir = model_path.parent
    model = _load_model(model_path)

    meta_path = model_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to parse metadata.json ({exc}); fallback defaults")
            meta = {}

    reduction_method = str(meta.get("reduction_method", "pca")).lower()
    component_labels = meta.get("component_labels", ["PC1", "PC2"])
    if not isinstance(component_labels, list) or len(component_labels) < 2:
        component_labels = ["PC1", "PC2"]

    reference_ca_path = Path(args.reference_ca).resolve() if args.reference_ca else model_dir / "reference_ca.pdb"
    align_ca_positions_path = (
        Path(args.align_ca_positions).resolve()
        if args.align_ca_positions
        else model_dir / "align_ca_positions.npy"
    )
    exclude_resnames = _comma_list(args.exclude_resnames)

    ref_ca = md.load(str(reference_ca_path))
    align_ca_positions = np.load(align_ca_positions_path, allow_pickle=False)
    if (align_ca_positions.ndim != 1 or len(align_ca_positions) < 3
            or not np.issubdtype(align_ca_positions.dtype, np.integer)
            or align_ca_positions.min() < 0 or align_ca_positions.max() >= ref_ca.n_atoms
            or len(np.unique(align_ca_positions)) != len(align_ca_positions)):
        raise ValueError("Invalid fixed-transform alignment indices")

    rows = []
    for struct_path_str in args.structures:
        struct_path = Path(struct_path_str).resolve()
        topo = Path(args.topology).resolve() if args.topology else None
        t_raw = _load_traj(struct_path, topo, stride=1)

        keep_mask = _build_non_solvent_atom_mask(t_raw.topology, exclude_resnames)
        t = t_raw.atom_slice(np.where(keep_mask)[0])

        ca_idx = t.topology.select("name CA")
        if len(ca_idx) != ref_ca.n_atoms:
            raise ValueError(
                f"{struct_path.name}: CA atom count mismatch. "
                f"input={len(ca_idx)} vs reference_ca={ref_ca.n_atoms}. "
                "Ensure residue order/length matches training system."
            )
        t_ca = t.atom_slice(ca_idx)
        if ([a.residue.name for a in t_ca.topology.atoms]
                != [a.residue.name for a in ref_ca.topology.atoms]):
            raise ValueError("C-alpha residue order differs from the fixed-transform reference")
        if not np.isfinite(t_ca.xyz).all():
            raise ValueError("Nonfinite coordinates cannot be projected")
        t_ca.superpose(
            ref_ca,
            frame=0,
            atom_indices=align_ca_positions,
            ref_atom_indices=align_ca_positions,
        )

        X = t_ca.xyz.reshape(t_ca.n_frames, -1)
        proj = _transform_with_model(model, X)
        if proj.ndim != 2 or proj.shape[1] < 2 or not np.isfinite(proj).all():
            raise ValueError(f"Projected dimension is {proj.shape[1]}, expected >=2")

        for i in range(t_ca.n_frames):
            rows.append((struct_path.name, i, float(proj[i, 0]), float(proj[i, 1])))

    out_csv = Path(args.output_csv).resolve() if args.output_csv else model_dir / "projection_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["structure", "frame", component_labels[0].lower(), component_labels[1].lower()])
        writer.writerows(rows)

    if args.plot:
        fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=args.dpi)
        dim1 = np.array([r[2] for r in rows], dtype=float)
        dim2 = np.array([r[3] for r in rows], dtype=float)
        cval = np.arange(len(rows), dtype=float)
        sc = ax.scatter(dim1, dim2, c=cval, cmap=args.cmap, s=18, alpha=0.9, linewidths=0.0)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Projection order", fontsize=14)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        ax.set_xlabel(component_labels[0], fontsize=14)
        ax.set_ylabel(component_labels[1], fontsize=14)
        ax.set_title(args.plot_title or f"Projected structures ({reduction_method.upper()})", fontsize=15)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        _style_axes(ax)
        fig.tight_layout()
        out_plot = Path(args.output_plot).resolve() if args.output_plot else model_dir / "projection_plot.png"
        fig.savefig(out_plot, dpi=args.dpi)
        plt.close(fig)
        print(f"Projection plot saved: {out_plot}")

    print(f"Projection CSV saved: {out_csv}")
    print(f"Projected points: {len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conformation-space fit/project with TICA/PCA + FES")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------- fit ----------
    p_fit = sub.add_parser("fit", help="Fit projection basis and FES from trajectory/trajectories")
    p_fit.add_argument("--system", choices=["jnj2113", "generic"], default="jnj2113",
                       help="JNJ-2113 enforces receptor-only alignment and four replica boundaries")
    p_fit.add_argument("--trajectory-lengths", nargs="+", type=int,
                       help="Independent replica lengths AFTER stride, needed for a concatenated file")
    p_fit.add_argument("--topology", required=True, help="AMBER topology (.prmtop)")
    p_fit.add_argument("--traj", required=True, nargs="+", help="Trajectory file(s), e.g. .nc")
    p_fit.add_argument(
        "--reference",
        default=None,
        help="Optional reference structure/frame file. If omitted, use joined trajectory frame.",
    )
    p_fit.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Reference frame index in joined trajectory when --reference is omitted (default: 0)",
    )
    p_fit.add_argument(
        "--align-selection",
        default=None,
        help=(
            "MDTraj atom selection for alignment (receptor backbone CA). "
            "If omitted, auto-detect largest protein chain CA."
        ),
    )
    p_fit.add_argument(
        "--receptor-chain",
        type=int,
        default=None,
        help="Optional receptor chain index for alignment selection.",
    )
    p_fit.add_argument(
        "--feature-selection",
        default="name CA",
        help="MDTraj atom selection used as projection features (default: 'name CA').",
    )
    p_fit.add_argument(
        "--exclude-resnames",
        default="Na+,Cl-,NA,CL,K+,K,MG,MG2+,CA,ZN,ZN2+",
        help="Comma-separated residue names to remove in addition to water.",
    )
    p_fit.add_argument("--stride", type=int, default=1, help="Load every Nth frame (default: 1)")
    p_fit.add_argument("--export-training-inputs", action="store_true",
                       help="Export aligned physical coordinates and explicit replica/frame IDs; list independent replicas in numerical order")

    # reduction backend
    p_fit.add_argument("--method", choices=["auto", "tica", "pca"], default="tica")
    p_fit.add_argument("--tica-lag", type=int, default=5, help="TICA lag time in frames (default: 5)")
    p_fit.add_argument("--n-components", type=int, default=2, help="Projection dimensions (default: 2)")
    p_fit.add_argument("--random-state", type=int, default=42, help="PCA random_state")

    # output and plotting
    p_fit.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    p_fit.add_argument("--figure-prefix", default="pca_scatter", help="Figure filename prefix")
    p_fit.add_argument("--title", default=None, help="Scatter plot title")
    p_fit.add_argument("--cmap", default="viridis", help="Scatter plot colormap")
    p_fit.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p_fit.add_argument("--save-pdf", action="store_true", help="Also save PDF figures")

    # FES options
    p_fit.add_argument("--fes-bins", type=int, default=100, help="2D histogram bins for FES (default: 100)")
    p_fit.add_argument(
        "--fes-sigma",
        type=float,
        default=1.8,
        help="Gaussian smoothing sigma in bin units (default: 1.8)",
    )
    p_fit.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (default: 300)")
    p_fit.add_argument("--fes-cmap", default="coolwarm", help="FES contour colormap (default: coolwarm)")
    p_fit.add_argument("--fes-levels", type=int, default=30, help="Number of contour levels (default: 30)")
    p_fit.add_argument("--fes-title", default=None, help="FES plot title")
    p_fit.add_argument("--n-minima", type=int, default=5, help="Number of minima to annotate (default: 5)")
    p_fit.add_argument(
        "--minima-separation-bins",
        type=int,
        default=None,
        help=(
            "Minimum distance between selected minima in bins. "
            "If omitted, uses max(30, int(fes_bins*0.3)) for macro-state separation."
        ),
    )
    p_fit.add_argument(
        "--manual-m1",
        default=None,
        help=(
            "Manual M1 coordinate as 'x,y'. "
            "If both --manual-m1 and --manual-m2 are provided, auto minima search is skipped."
        ),
    )
    p_fit.add_argument(
        "--manual-m2",
        default=None,
        help=(
            "Manual M2 coordinate as 'x,y'. "
            "If both --manual-m1 and --manual-m2 are provided, auto minima search is skipped."
        ),
    )
    p_fit.set_defaults(func=_fit)

    # ---------- project ----------
    p_proj = sub.add_parser("project", help="Project new structures into existing basis")
    p_proj.add_argument("--model", required=True, help="Saved model path (still named pca_model.pkl)")
    p_proj.add_argument(
        "--structures",
        required=True,
        nargs="+",
        help="Structure/trajectory file(s) to project (e.g., generated single-frame PDB)",
    )
    p_proj.add_argument(
        "--topology",
        default=None,
        help="Topology for non-PDB trajectory inputs (optional for PDB).",
    )
    p_proj.add_argument(
        "--reference-ca",
        default=None,
        help="Path to reference_ca.pdb (default: <model_dir>/reference_ca.pdb)",
    )
    p_proj.add_argument(
        "--align-ca-positions",
        default=None,
        help="Path to align_ca_positions.npy (default: <model_dir>/align_ca_positions.npy)",
    )
    p_proj.add_argument(
        "--exclude-resnames",
        default="Na+,Cl-,NA,CL,K+,K,MG,MG2+,CA,ZN,ZN2+",
        help="Comma-separated residue names to remove in addition to water.",
    )
    p_proj.add_argument("--output-csv", default=None, help="CSV output path")
    p_proj.add_argument("--plot", action="store_true", help="Save projection scatter plot")
    p_proj.add_argument("--output-plot", default=None, help="Projection plot output path")
    p_proj.add_argument("--plot-title", default=None, help="Projection plot title")
    p_proj.add_argument("--cmap", default="plasma", help="Matplotlib colormap")
    p_proj.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    p_proj.set_defaults(func=_project)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
