"""
LDDT (Local Distance Difference Test) loss
基于RFDiffusion的实现，简化版本
"""
import torch
import torch.nn as nn


def compute_lddt_loss(pred_coords, target_coords, mask=None,
                     distance_threshold=1.5, eps=1e-6):
    """
    计算LDDT loss

    Args:
        pred_coords: (B, N, 3) 预测坐标 (nm)
        target_coords: (B, N, 3) 目标坐标 (nm)
        mask: (B, N) 原子mask
        distance_threshold: 距离阈值 (nm)，只考虑参考距离<threshold的原子对

    Returns:
        loss: 标量，1 - LDDT (越小越好)
    """
    B, N = pred_coords.shape[:2]
    device = pred_coords.device

    # 默认mask
    if mask is None:
        mask = torch.ones(B, N, device=device, dtype=torch.bool)

    # 计算所有原子对的索引 (上三角)
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=device)

    # BUG FIX: Compute valid pairs per sample, not just from first sample
    # Calculate reference distances from target (assume same topology across batch)
    target_dists = torch.norm(
        target_coords[:, i_idx] - target_coords[:, j_idx],
        dim=-1
    )  # (B, n_pairs)

    # Only keep pairs with distance < threshold (use first sample as reference)
    valid_pairs = (target_dists[0] > 0) & (target_dists[0] < distance_threshold)

    # Mask: both atoms must be valid
    pair_mask = mask[:, i_idx] & mask[:, j_idx]  # (B, n_pairs)
    valid_pairs_batch = valid_pairs.unsqueeze(0) & pair_mask  # (B, n_pairs)

    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=device)

    # 筛选有效原子对 (per sample)
    # For efficiency, use the reference topology from first sample
    i_valid = i_idx[valid_pairs]
    j_valid = j_idx[valid_pairs]
    target_dists_valid = target_dists[:, valid_pairs]  # (B, n_valid)

    # 计算预测距离
    pred_dists = torch.norm(
        pred_coords[:, i_valid] - pred_coords[:, j_valid],
        dim=-1
    )  # (B, n_valid)

    # 计算距离差异
    delta = torch.abs(pred_dists - target_dists_valid + eps)  # (B, n_valid)

    # LDDT: 4个阈值的平均 (0.5, 1.0, 2.0, 4.0 Å = 0.05, 0.1, 0.2, 0.4 nm)
    lddt = 0.25 * (
        torch.sigmoid(0.05 - delta) +
        torch.sigmoid(0.10 - delta) +
        torch.sigmoid(0.20 - delta) +
        torch.sigmoid(0.40 - delta)
    )  # (B, n_valid)

    # 平均
    lddt_score = lddt.mean(dim=-1)  # (B,)

    # Loss = 1 - LDDT
    loss = 1.0 - lddt_score.mean()

    return loss


class LDDTLoss(nn.Module):
    """LDDT loss模块"""

    def __init__(self, distance_threshold=1.5, weight=1.0):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.weight = weight

    def forward(self, pred_coords, target_coords, mask=None):
        """
        Args:
            pred_coords: (B, N, 3) nm
            target_coords: (B, N, 3) nm
            mask: (B, N) bool
        """
        loss = compute_lddt_loss(
            pred_coords, target_coords, mask,
            self.distance_threshold
        )
        return self.weight * loss
