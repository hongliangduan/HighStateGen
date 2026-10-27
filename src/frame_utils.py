"""
Frame utilities for SE(3) flow matching.

Frame = (R ∈ SO(3), t ∈ R³) per residue.
  - Origin at CA
  - x-axis: CA → C (normalized)
  - z-axis: normal to N-CA-C plane
  - y-axis: z × x (right-handed)

All SO(3) operations forced to float32 for numerical stability under bf16.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─── Reference bond geometry (nm) ───────────────────────────────
# These are idealized values; actual values computed from data in dataset.
BOND_CA_C = 0.1529   # CA-C bond length (nm)
BOND_CA_N = 0.1465   # N-CA bond length (nm)
BOND_C_O  = 0.1231   # C=O bond length (nm)
ANGLE_N_CA_C = np.radians(110.6)  # N-CA-C angle


def _ensure_f32(fn):
    """Decorator: cast inputs to float32, cast output back to input dtype."""
    def wrapper(*args, **kwargs):
        orig_dtype = args[0].dtype
        args_f32 = [a.float() if torch.is_tensor(a) else a for a in args]
        kwargs_f32 = {k: v.float() if torch.is_tensor(v) else v
                      for k, v in kwargs.items()}
        out = fn(*args_f32, **kwargs_f32)
        if isinstance(out, tuple):
            return tuple(o.to(orig_dtype) if torch.is_tensor(o) else o for o in out)
        return out.to(orig_dtype) if torch.is_tensor(out) else out
    return wrapper


# ─── SO(3) operations ────────────────────────────────────────────

def hat(omega):
    """Convert axis-angle vector to skew-symmetric matrix.
    omega: (*, 3) -> (*, 3, 3)
    """
    x, y, z = omega.unbind(-1)
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)


@_ensure_f32
def so3_exp_map(omega):
    """Exponential map so(3) → SO(3) via Rodrigues' formula.

    Args:
        omega: (*, 3) axis-angle vectors
    Returns:
        R: (*, 3, 3) rotation matrices
    """
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)  # (*, 1)
    axis = omega / theta  # (*, 3)
    K = hat(axis)  # (*, 3, 3)

    # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
    # For small θ: sin(θ)/θ ≈ 1, (1-cos(θ))/θ² ≈ 0.5
    sin_t = torch.sin(theta).unsqueeze(-1)   # (*, 1, 1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


@_ensure_f32
def so3_log_map(R):
    """Logarithmic map SO(3) → so(3).

    Handles small angles (θ≈0) and near-π angles with special cases.

    Args:
        R: (*, 3, 3) rotation matrices
    Returns:
        omega: (*, 3) axis-angle vectors
    """
    # cos(θ) = (tr(R) - 1) / 2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (*)

    # Skew-symmetric part: (R - R^T) / 2
    skew = 0.5 * (R - R.transpose(-1, -2))
    axis_unnorm = torch.stack([
        skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]
    ], dim=-1)  # (*, 3)

    safe_sin = torch.sin(angle).clamp(min=1e-8)
    scale = angle / safe_sin

    small = angle.abs() < 1e-4
    scale = torch.where(small, torch.ones_like(scale), scale)

    # near-pi: threshold must be < acos(-1+1e-7) ≈ 3.1396, so use 2e-3 margin
    near_pi = angle.abs() > (torch.pi - 2e-3)
    if near_pi.any():
        # Extract axis from (R + I)/2 diagonal, then use the column with
        # the largest diagonal entry to determine signs (standard method).
        diag = torch.stack([
            R[..., 0, 0] + 1.0,
            R[..., 1, 1] + 1.0,
            R[..., 2, 2] + 1.0,
        ], dim=-1).clamp(min=0.0)  # (*, 3)
        # Pick the axis component with largest magnitude from diagonal
        max_idx = diag.argmax(dim=-1, keepdim=True)  # (*, 1)
        axis_pi = torch.sqrt(diag / 2.0 + 1e-8)
        # Fix signs using off-diagonal elements of (R + I)/2
        # For the dominant axis k, sign of other axes j is sign(R[j,k] + R[k,j])
        Rsym = R + R.transpose(-1, -2)  # symmetric part * 2
        # Use sign of Rsym[j, k_max] to fix axis_pi[j] relative to axis_pi[k_max]
        # Gather the k_max column from Rsym: (*, 3)
        max_idx_3 = max_idx.unsqueeze(-1).expand_as(Rsym[..., :1, :])
        ref_col = Rsym.gather(-2, max_idx_3).squeeze(-2)  # (*, 3)
        sign = torch.sign(ref_col + 1e-12)
        axis_pi = axis_pi * sign
        axis_pi = F.normalize(axis_pi, dim=-1)
        omega_pi = axis_pi * angle.unsqueeze(-1)
        omega = axis_unnorm * scale.unsqueeze(-1)
        near_pi_expanded = near_pi.unsqueeze(-1).expand_as(omega)
        omega = torch.where(near_pi_expanded, omega_pi, omega)
    else:
        omega = axis_unnorm * scale.unsqueeze(-1)

    # Final clamp to prevent extreme values
    omega = omega.clamp(-20.0, 20.0)
    return omega


@_ensure_f32
def geodesic_so3(R0, R1, t):
    """Geodesic interpolation on SO(3): R_t = R0 @ exp(t * log(R0^T @ R1)).

    Args:
        R0: (*, 3, 3) start rotations
        R1: (*, 3, 3) end rotations
        t: scalar or broadcastable tensor
    Returns:
        R_t: (*, 3, 3) interpolated rotations
    """
    # Relative rotation in local frame of R0
    R_rel = R0.transpose(-1, -2) @ R1  # R0^T @ R1
    omega = so3_log_map(R_rel)  # (*, 3)

    if not isinstance(t, (int, float)):
        while t.dim() < omega.dim():
            t = t.unsqueeze(-1)

    R_t = R0 @ so3_exp_map(t * omega)
    return R_t


def random_so3(shape, device, dtype=torch.float32):
    """Sample uniform random rotations on SO(3) via QR decomposition.

    Args:
        shape: batch shape, e.g. (B, N_res)
        device: torch device
    Returns:
        R: (*shape, 3, 3) random rotation matrices
    """
    flat = int(np.prod(shape)) if isinstance(shape, (tuple, list)) else shape
    # Random Gaussian matrix → QR → orthogonal
    M = torch.randn(flat, 3, 3, device=device, dtype=torch.float32)
    Q, Rmat = torch.linalg.qr(M)
    # Ensure det(Q) = +1 (proper rotation)
    sign = torch.sign(torch.diagonal(Rmat, dim1=-2, dim2=-1))  # (flat, 3)
    Q = Q * sign.unsqueeze(-2)
    det = torch.det(Q)
    # Flip first column if det < 0
    Q[det < 0, :, 0] *= -1.0
    if isinstance(shape, (tuple, list)):
        Q = Q.reshape(*shape, 3, 3)
    return Q.to(dtype)


# ─── Frame ↔ Coordinate conversion ──────────────────────────────

def coords_to_frames(coords, residue_bb_indices):
    """Convert backbone atom coordinates to per-residue frames.

    Args:
        coords: (B, N_atoms, 3) or (N_atoms, 3) in nm (physical space)
        residue_bb_indices: (N_res, 4) int, columns = [N, CA, C, O] local indices
    Returns:
        R: (B, N_res, 3, 3) rotation matrices
        t: (B, N_res, 3) translations (CA positions)
        O_local: (B, N_res, 3) O atom in local frame
    """
    squeeze = coords.dim() == 2
    if squeeze:
        coords = coords.unsqueeze(0)

    coords = coords.float()  # force f32
    B = coords.shape[0]
    idx_N  = residue_bb_indices[:, 0]  # (N_res,)
    idx_CA = residue_bb_indices[:, 1]
    idx_C  = residue_bb_indices[:, 2]
    idx_O  = residue_bb_indices[:, 3]

    N_pos  = coords[:, idx_N]   # (B, N_res, 3)
    CA_pos = coords[:, idx_CA]
    C_pos  = coords[:, idx_C]
    O_pos  = coords[:, idx_O]

    # Translation = CA position
    t = CA_pos  # (B, N_res, 3)

    # x-axis: CA → C
    x_axis = F.normalize(C_pos - CA_pos, dim=-1)
    # v = N - CA (in the N-CA-C plane)
    v = N_pos - CA_pos
    # z-axis: normal to N-CA-C plane = x × v (then normalize)
    z_axis = F.normalize(torch.cross(x_axis, v, dim=-1), dim=-1)
    # y-axis: z × x
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    # R = [x | y | z] as columns → R[:,:,i] = axis_i
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, N_res, 3, 3)

    # O in local frame: O_local = R^T @ (O - CA)
    O_local = torch.einsum('...ij,...j->...i', R.transpose(-1, -2), O_pos - CA_pos)

    if squeeze:
        R, t, O_local = R.squeeze(0), t.squeeze(0), O_local.squeeze(0)
    return R, t, O_local


def frames_to_coords(R, t, O_local, N_local, residue_bb_indices, n_atoms):
    """Reconstruct backbone atom coordinates from per-residue frames.

    Args:
        R: (B, N_res, 3, 3) rotation matrices
        t: (B, N_res, 3) CA positions in nm
        O_local: (B, N_res, 3) or (N_res, 3) O position in local frame
        N_local: (N_res, 3) N position in local frame (fixed per residue)
        residue_bb_indices: (N_res, 4) [N, CA, C, O] local atom indices
        n_atoms: total number of atoms in the coordinate array
    Returns:
        coords: (B, n_atoms, 3) reconstructed coordinates in nm
    """
    squeeze = R.dim() == 2  # (N_res, 3, 3) unbatched
    if squeeze:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if O_local.dim() == 2:
        O_local = O_local.unsqueeze(0).expand(R.shape[0], -1, -1)

    R = R.float()
    t = t.float()
    B, N_res = R.shape[:2]

    # C_local is always [d_CA_C, 0, 0] by frame definition (x-axis = CA→C)
    C_local = torch.zeros(N_res, 3, device=R.device, dtype=R.dtype)
    # Compute per-residue CA-C distance from reference (or use constant)
    C_local[:, 0] = BOND_CA_C

    # Global positions: P_global = R @ P_local + t
    # CA = t (by definition)
    CA_global = t  # (B, N_res, 3)
    C_global = torch.einsum('...ij,...j->...i', R, C_local) + t
    N_global = torch.einsum('...ij,...j->...i', R, N_local.to(R.device).float()) + t
    O_global = torch.einsum('...ij,...j->...i', R, O_local.float()) + t

    # Scatter into full coordinate array
    coords = torch.zeros(B, n_atoms, 3, device=R.device, dtype=R.dtype)
    idx_N  = residue_bb_indices[:, 0]
    idx_CA = residue_bb_indices[:, 1]
    idx_C  = residue_bb_indices[:, 2]
    idx_O  = residue_bb_indices[:, 3]

    coords[:, idx_N]  = N_global
    coords[:, idx_CA] = CA_global
    coords[:, idx_C]  = C_global
    coords[:, idx_O]  = O_global

    if squeeze:
        coords = coords.squeeze(0)
    return coords


def build_residue_bb_indices(topology, selected_atom_indices=None,
                             skip_residues=None):
    """Build residue → backbone atom index mapping from topology.

    Args:
        topology: mdtraj.Topology
        selected_atom_indices: optional array of subset atom indices (global)
        skip_residues: list of residue indices to skip (e.g. ACE caps)
    Returns:
        residue_bb_indices: (N_res, 4) np.int64, columns = [N, CA, C, O]
            indices are LOCAL (into the subset array if selected_atom_indices given)
        valid_residue_ids: (N_res,) residue indices that have all 4 backbone atoms
    """
    if skip_residues is None:
        skip_residues = []

    # Build global → local index map
    g2l = None
    if selected_atom_indices is not None:
        g2l = {int(g): l for l, g in enumerate(selected_atom_indices)}

    bb_names = {'N', 'CA', 'C', 'O'}
    residue_bb = []
    valid_ids = []

    for res in topology.residues:
        if res.index in skip_residues:
            continue
        atoms = {}
        for atom in res.atoms:
            if atom.name in bb_names:
                gidx = atom.index
                if g2l is not None:
                    if gidx not in g2l:
                        continue
                    lidx = g2l[gidx]
                else:
                    lidx = gidx
                atoms[atom.name] = lidx

        if len(atoms) == 4:
            residue_bb.append([atoms['N'], atoms['CA'], atoms['C'], atoms['O']])
            valid_ids.append(res.index)

    return np.array(residue_bb, dtype=np.int64), np.array(valid_ids, dtype=np.int64)


def compute_reference_locals(coords_nm, residue_bb_indices):
    """Compute reference N_local and O_local from training data.

    Args:
        coords_nm: (N_samples, N_atoms, 3) physical coordinates in nm
        residue_bb_indices: (N_res, 4) [N, CA, C, O] indices
    Returns:
        N_local_mean: (N_res, 3) mean N position in local frame
        O_local_mean: (N_res, 3) mean O position in local frame
    """
    coords_t = torch.from_numpy(coords_nm).float()
    R, t, O_local = coords_to_frames(coords_t, torch.from_numpy(residue_bb_indices))
    # R: (N_samples, N_res, 3, 3), t: (N_samples, N_res, 3)

    # N in local frame
    idx_N = residue_bb_indices[:, 0]
    idx_CA = residue_bb_indices[:, 1]
    N_pos = coords_t[:, idx_N]
    CA_pos = coords_t[:, idx_CA]
    N_local = torch.einsum('...ij,...j->...i', R.transpose(-1, -2), N_pos - CA_pos)

    N_local_mean = N_local.mean(dim=0)  # (N_res, 3)
    O_local_mean = O_local.mean(dim=0)  # (N_res, 3)
    return N_local_mean, O_local_mean


def reconstruct_pseudo_cb(N_pos, CA_pos, C_pos):
    """从 backbone 原子重建理想 pseudo-Cβ 位置。

    使用标准几何：CA-Cβ = 0.153 nm, angle(N-CA-Cβ) = 110.5°, dihedral(C-CA-N-Cβ) = 122.5°

    Args:
        N_pos:  (..., 3) N coordinates in nm
        CA_pos: (..., 3) CA coordinates in nm
        C_pos:  (..., 3) C coordinates in nm

    Returns:
        CB_pos: (..., 3) pseudo-Cβ coordinates in nm
    """
    # Standard geometry parameters
    CA_CB_LENGTH = 0.153  # nm
    N_CA_CB_ANGLE = 1.9286  # radians (~110.5°)
    C_CA_N_CB_DIHEDRAL = 2.1380  # radians (~122.5°)

    # Build local coordinate system at CA
    v1 = N_pos - CA_pos  # CA -> N
    v2 = C_pos - CA_pos  # CA -> C

    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)

    # Normal to N-CA-C plane
    n = torch.cross(v1, v2, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)

    # Cβ direction in the plane defined by angle and dihedral
    cos_angle = torch.cos(torch.tensor(N_CA_CB_ANGLE, device=CA_pos.device))
    sin_angle = torch.sin(torch.tensor(N_CA_CB_ANGLE, device=CA_pos.device))
    cos_dihed = torch.cos(torch.tensor(C_CA_N_CB_DIHEDRAL, device=CA_pos.device))
    sin_dihed = torch.sin(torch.tensor(C_CA_N_CB_DIHEDRAL, device=CA_pos.device))

    # Cβ direction: rotate from N direction by angle around normal
    cb_dir = cos_angle * v1 + sin_angle * (cos_dihed * n + sin_dihed * torch.cross(n, v1, dim=-1))
    cb_dir = cb_dir / (cb_dir.norm(dim=-1, keepdim=True) + 1e-8)

    CB_pos = CA_pos + CA_CB_LENGTH * cb_dir
    return CB_pos
