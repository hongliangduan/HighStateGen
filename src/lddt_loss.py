"""Differentiable local-distance surrogate in nm (not a reported hard lDDT score)."""
import torch
import torch.nn as nn


def compute_lddt_loss(pred_coords, target_coords, mask=None,
                      distance_threshold=1.5, eps=1e-6, reduction="mean"):
    """Use each sample's reference contacts and masks; average per sample.

    The archived sigmoid smoothing and four distance tolerances are retained.
    A sample with no evaluable pairs contributes zero, with a connected gradient.
    """
    if pred_coords.shape != target_coords.shape or pred_coords.ndim != 3 or pred_coords.shape[-1] != 3:
        raise ValueError("Coordinates must be matching [batch, atoms, 3] tensors")
    batch, count = pred_coords.shape[:2]
    if mask is None:
        mask = torch.ones(batch, count, device=pred_coords.device, dtype=torch.bool)
    if mask.shape != (batch, count):
        raise ValueError("mask must have shape [batch, atoms]")
    i, j = torch.triu_indices(count, count, offset=1, device=pred_coords.device)
    target_dist = torch.linalg.vector_norm(target_coords[:, i] - target_coords[:, j], dim=-1)
    pred_dist = torch.linalg.vector_norm(pred_coords[:, i] - pred_coords[:, j], dim=-1)
    valid = ((target_dist > 0) & (target_dist < distance_threshold)
             & mask[:, i].bool() & mask[:, j].bool())
    delta = torch.abs(pred_dist - target_dist)
    score = sum(torch.sigmoid(tolerance - delta) for tolerance in (0.05, 0.10, 0.20, 0.40)) / 4.0
    valid_count = valid.sum(dim=1)
    losses = ((1.0 - score) * valid).sum(dim=1) / valid_count.clamp_min(1)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none' or 'mean'")


class LDDTLoss(nn.Module):
    def __init__(self, distance_threshold=1.5, weight=1.0):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.weight = weight

    def forward(self, pred_coords, target_coords, mask=None):
        return self.weight * compute_lddt_loss(
            pred_coords, target_coords, mask, self.distance_threshold)
