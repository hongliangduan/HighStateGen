"""Regularized reversible TICA with explicit independent-trajectory boundaries."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TicaModel:
    mean_: np.ndarray
    components_: np.ndarray
    eigenvalues_: np.ndarray
    lagtime_: int

    def transform(self, features):
        return (np.asarray(features, dtype=np.float64) - self.mean_) @ self.components_


def fit_tica(features, lengths, lagtime=5, n_components=2, regularization=1e-6):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite two-dimensional array")
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
            or lengths.sum() != len(features) or lagtime < 1 or np.any(lengths <= lagtime)):
        raise ValueError("Supply a length > lagtime for every independent trajectory")
    if not 1 <= n_components <= features.shape[1] or regularization <= 0:
        raise ValueError("Invalid component count or regularization")
    mean = features.mean(axis=0)
    dimension = features.shape[1]
    c00 = np.zeros((dimension, dimension), dtype=np.float64)
    ctau = np.zeros_like(c00)
    offset = n_pairs = 0
    for length in lengths:
        segment = features[offset:offset + length] - mean
        x0, xt = segment[:-lagtime], segment[lagtime:]
        c00 += x0.T @ x0 + xt.T @ xt
        ctau += x0.T @ xt + xt.T @ x0
        n_pairs += len(x0)
        offset += length
    c00 /= 2 * n_pairs
    ctau /= 2 * n_pairs
    epsilon = regularization * np.trace(c00) / dimension
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("TICA cannot be fitted to zero-variance features")
    regularized = c00 + epsilon * np.eye(dimension)
    values, vectors = np.linalg.eigh(regularized)
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    symmetric = whitening @ ctau @ whitening
    eigenvalues, directions = np.linalg.eigh((symmetric + symmetric.T) / 2)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    components = whitening @ directions[:, order]
    # Resolve eigenvector sign deterministically; imported fixed models are not refitted.
    for column in range(n_components):
        pivot = np.argmax(np.abs(components[:, column]))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    model = TicaModel(mean, components, eigenvalues[order], int(lagtime))
    info = {"reduction_method": "tica", "backend": "numpy_replica_tica",
            "lagtime": int(lagtime), "regularization": float(regularization),
            "diagonal_regularization": float(epsilon), "lagged_pairs": int(n_pairs),
            "trajectory_lengths": lengths.tolist(), "cross_replica_pairs": 0,
            "explained_variance_ratio": None,
            "tica_eigenvalues": model.eigenvalues_.tolist()}
    return model, model.transform(features), info
