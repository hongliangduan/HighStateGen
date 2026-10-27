"""
SE(3) flow matching: interpolation, velocity targets, and losses.

Translation: linear interpolation (same as Cartesian flow matching)
Rotation: geodesic interpolation on SO(3)
"""

import torch
import numpy as np
from frame_utils import so3_exp_map, so3_log_map, geodesic_so3, random_so3


def frame_flow_interpolate(R0, t0, R1, t1, s, sigma_min=1e-4):
    """SE(3) flow matching interpolation.

    Translation: t_s = (1 - (1-σ)s) * t0 + s * t1
    Rotation:    R_s = R0 @ exp(s * log(R0^T @ R1))

    Args:
        R0: (B, N_res, 3, 3) data rotations
        t0: (B, N_res, 3) data translations (standardized)
        R1: (B, N_res, 3, 3) noise rotations (random SO(3))
        t1: (B, N_res, 3) noise translations (Gaussian)
        s: (B,) time in [0, 1]
        sigma_min: minimum noise level
    Returns:
        R_s, t_s: interpolated frames
    """
    s_ = s.reshape(-1, 1, 1)  # (B, 1, 1)
    t_s = (1.0 - (1.0 - sigma_min) * s_) * t0 + s_ * t1
    R_s = geodesic_so3(R0, R1, s.reshape(-1, 1))  # (B, N_res, 3, 3)
    return R_s, t_s


def frame_velocity_target(R0, t0, R1, t1, sigma_min=1e-4):
    """Compute velocity targets for SE(3) flow matching.

    Translation: v_trans = t1 - (1-σ) * t0
    Rotation:    omega = log(R0^T @ R1)

    Args:
        R0: (B, N_res, 3, 3) data rotations
        t0: (B, N_res, 3) data translations
        R1: (B, N_res, 3, 3) noise rotations
        t1: (B, N_res, 3) noise translations
    Returns:
        v_trans: (B, N_res, 3)
        omega: (B, N_res, 3) axis-angle rotation velocity
    """
    v_trans = t1 - (1.0 - sigma_min) * t0
    R_rel = R0.transpose(-1, -2) @ R1
    omega = so3_log_map(R_rel)  # (B, N_res, 3)
    return v_trans, omega


def frame_flow_loss(v_trans_pred, omega_pred, v_trans_target, omega_target,
                    lambda_rot=1.0, reduction="mean"):
    """Combined SE(3) flow matching loss.

    L = MSE(v_trans) + lambda_rot * MSE(omega)

    Returns:
        loss, loss_trans, loss_rot: scalars
    """
    loss_trans = ((v_trans_pred - v_trans_target) ** 2).mean(dim=(1, 2))
    loss_rot = ((omega_pred - omega_target) ** 2).mean(dim=(1, 2))
    if reduction == "mean":
        loss_trans, loss_rot = loss_trans.mean(), loss_rot.mean()
    elif reduction != "none":
        raise ValueError("reduction must be 'none' or 'mean'")
    loss = loss_trans + lambda_rot * loss_rot
    return loss, loss_trans, loss_rot


def sample_time_logit_normal(B, device, mean=0.0, std=1.0, sigma_min=1e-4):
    """Logit-normal time sampling (focuses on mid-range t)."""
    z = torch.randn(B, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(1e-3, 1.0 - 1e-3)  # stricter clamp to avoid boundary singularities


def sample_time_uniform(B, device, sigma_min=1e-4):
    """Uniform time sampling."""
    return torch.rand(B, device=device) * (1.0 - sigma_min) + sigma_min


# ─── Predicted clean frames from velocity ─────────────────────────

def pred_frames_from_velocity(R_s, t_s, v_pred, omega_pred, s, sigma_min=1e-4):
    """Recover predicted clean frames (R0, t0) from noisy frames + predicted velocity.

    This inversion is exact for the flow parameterization used in this file.

    Translation flow:
        t_s = (1 - (1 - sigma_min) s) t0 + s t1
        v   = t1 - (1 - sigma_min) t0

    Substituting v into t_s gives:
        t_s = t0 + s v

    therefore:
        t0 = t_s - s v

    Rotation flow:
        R_s = R0 @ exp(s * omega)

    therefore:
        R0 = R_s @ exp(-s * omega)

    Args:
        R_s: (B, N, 3, 3) noisy rotations
        t_s: (B, N, 3) noisy translations (standardized)
        v_pred: (B, N, 3) predicted translation velocity
        omega_pred: (B, N, 3) predicted rotation velocity
        s: (B,) time
        sigma_min: retained for API consistency with the forward process
    Returns:
        pred_R0: (B, N, 3, 3), pred_t0: (B, N, 3)
    """
    s_ = s.reshape(-1, 1, 1).to(t_s.dtype)  # (B, 1, 1)
    pred_t0 = t_s - s_ * v_pred
    pred_R0 = R_s @ so3_exp_map(-s_ * omega_pred)
    return pred_R0, pred_t0
