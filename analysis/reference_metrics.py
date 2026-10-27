"""Explicitly defined reference distances; no implicit 'training' subset.

Coordinates must use the same atom order and units. Returned distances use
those units. These corrected routines do not regenerate historical table values.
"""
import numpy as np


def atom_rmsd(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0:
        raise ValueError("RMSD requires matching nonempty [atoms, 3] arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite coordinates")
    return float(np.sqrt(np.square(a - b).sum(axis=-1).mean()))


def _indices(values, size, minimum=1):
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) < minimum
            or not np.issubdtype(values.dtype, np.integer)
            or len(np.unique(values)) != len(values)
            or values.min() < 0 or values.max() >= size):
        raise ValueError("Supply unique valid zero-based indices")
    return values


def nearest_reference(query, references, alignment_indices, metric_indices,
                      reference_rows, chunk_size=1024):
    """Independently align each query/reference pair on receptor atoms.

    Evaluate only metric_indices after this fit; never refit the peptide.
    reference_rows must be explicit (e.g. split_indices['train']); no prefix
    of a concatenated trajectory is silently called the training set.
    """
    query = np.asarray(query, dtype=float)
    references = np.asarray(references)
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError("query must be finite [atoms, 3]")
    if references.ndim != 3 or references.shape[1:] != query.shape or chunk_size < 1:
        raise ValueError("Reference shape/order must match query and chunk_size must be positive")
    align = _indices(alignment_indices, len(query), 3)
    metric = _indices(metric_indices, len(query))
    rows = _indices(reference_rows, len(references))
    x = query[align]
    xc = x - x.mean(axis=0)
    if np.linalg.matrix_rank(xc) < 2:
        raise ValueError("Alignment atoms are collinear")
    best_distance, best_row = np.inf, None
    for start in range(0, len(rows), chunk_size):
        selected_rows = rows[start:start + chunk_size]
        ref = np.asarray(references[selected_rows], dtype=float)
        if not np.isfinite(ref).all():
            raise ValueError("Nonfinite reference coordinates")
        y = ref[:, align]
        yc = y - y.mean(axis=1, keepdims=True)
        if np.any(np.linalg.matrix_rank(yc) < 2):
            raise ValueError("Reference alignment atoms are collinear")
        covariance = np.einsum("ni,bnj->bij", xc, yc)
        u, _, vh = np.linalg.svd(covariance)
        d = np.ones((len(ref), 3))
        d[:, -1] = np.where(np.linalg.det(u @ vh) < 0, -1.0, 1.0)
        rotation = (u * d[:, None, :]) @ vh
        fitted = np.einsum("ni,bij->bnj", query[metric] - x.mean(axis=0), rotation)
        fitted += y.mean(axis=1)[:, None, :]
        distances = np.sqrt(np.square(fitted - ref[:, metric]).sum(axis=-1).mean(axis=1))
        local = int(np.argmin(distances))
        if distances[local] < best_distance:
            best_distance, best_row = float(distances[local]), int(selected_rows[local])
    return best_distance, best_row


def assignment(points, centers):
    points, centers = np.asarray(points), np.asarray(centers)
    if (points.ndim != 2 or points.shape[1] != 2 or centers.ndim != 2
            or centers.shape[1] != 2 or len(centers) == 0
            or not np.isfinite(points).all() or not np.isfinite(centers).all()):
        raise ValueError("Supply finite TIC1/TIC2 arrays and nonempty ordered centers")
    return np.square(points[:, None, :] - centers[None, :, :]).sum(-1).argmin(-1)


def initial_sample_rate(passed, initial_count):
    """Missing/failed outputs stay in the initially requested denominator."""
    if (not isinstance(initial_count, (int, np.integer)) or initial_count <= 0
            or not isinstance(passed, (int, np.integer)) or not 0 <= passed <= initial_count):
        raise ValueError("Invalid initial count or number passed")
    return 100.0 * passed / initial_count
