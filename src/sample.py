#!/usr/bin/env python3
"""
HighStateGen sampling — Dual Frame Flow with Heun ODE integrator.

Sampling design:
  - Heun (2nd-order predictor-corrector) replaces Euler integrator
  - Removed Langevin corrector (injected noise near t=0, destroying structure)
  - CFG uses batch-merged single forward pass (2x GPU efficiency)
  - Physics guidance retained as deterministic gradient correction only
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, 'analysis'))

from frame_utils import random_so3, so3_exp_map, frames_to_coords, so3_log_map
from ipa_model_dual import ProteinPeptideFrameDenoiser
from physics_losses import TopologyHarmonicPenalty, build_disulfide_loss
from frame_flow_matching import pred_frames_from_velocity


# ─── SO(3) projection ────────────────────────────────────────────
def project_so3(R):
    U, _, Vt = torch.linalg.svd(R)
    R_proj = U @ Vt
    det = torch.det(R_proj)
    U_mod = U.clone()
    U_mod[..., :, 2] = U_mod[..., :, 2] * det[..., None]
    return U_mod @ Vt


def ca_ca_guidance_loss(t_norm, global_scale, target_nm=0.385):
    """Harmonic penalty on sequential CA-CA distances (in normalized coords).
    target_nm: target CA-CA distance in nm (0.385 nm = 3.85 A, mean of training data).
    """
    target_norm = target_nm / global_scale
    diffs = t_norm[:, 1:] - t_norm[:, :-1]          # (B, N-1, 3)
    dists = torch.norm(diffs, dim=-1)                 # (B, N-1)
    return ((dists - target_norm) ** 2).mean()


def tica_project(
    protein_t_norm,
    peptide_t_norm,
    tica_mean,
    tica_components,
    global_scale,
    complex_com,
):
    """Differentiable linear TICA/PCA projection on flattened CA coordinates.

    This intentionally uses the saved linear model directly. It assumes the
    generated coordinates are already in the same aligned coordinate frame as
    the training set, which is true for the v14 frame dataset used here.
    """
    ca_norm = torch.cat([protein_t_norm, peptide_t_norm], dim=1)
    com = torch.as_tensor(complex_com, device=ca_norm.device, dtype=ca_norm.dtype)
    ca_angstrom = (ca_norm * float(global_scale) + com[None, None, :]) * 10.0
    flat = ca_angstrom.reshape(ca_angstrom.shape[0], -1)
    return (flat - tica_mean[None, :]) @ tica_components


def tica_guidance_loss_from_tic(
    tic,
    tica_target,
    flat_bottom_threshold=0.0,
):
    """Flat-bottom TICA guidance.

    If flat_bottom_threshold > 0, guidance activates only when distance to
    target exceeds threshold: max(0, ||tic-target|| - threshold)^2.
    """
    delta = tic - tica_target[None, :]
    dist = torch.norm(delta, dim=-1)
    if flat_bottom_threshold is not None and float(flat_bottom_threshold) > 0.0:
        excess = torch.clamp(dist - float(flat_bottom_threshold), min=0.0)
        loss = (excess ** 2).mean()
        active_frac = (dist > float(flat_bottom_threshold)).float().mean()
    else:
        loss = (delta ** 2).sum(dim=-1).mean()
        active_frac = torch.ones((), device=tic.device, dtype=tic.dtype)
    return loss, dist, active_frac


def anti_m2_repulsion_loss_from_tic(
    tic,
    anti_m2_center,
    repel_radius=0.9,
):
    """Repulsion loss from M2 center in TIC space.

    Active only inside `repel_radius`:
      loss = max(0, repel_radius - ||tic-center||)^2
    """
    delta = tic - anti_m2_center[None, :]
    dist = torch.norm(delta, dim=-1)
    if repel_radius is None or float(repel_radius) <= 0.0:
        loss = torch.zeros((), device=tic.device, dtype=tic.dtype)
        active_frac = torch.zeros((), device=tic.device, dtype=tic.dtype)
    else:
        gap = torch.clamp(float(repel_radius) - dist, min=0.0)
        loss = (gap ** 2).mean()
        active_frac = (dist < float(repel_radius)).float().mean()
    return loss, dist, active_frac


def annealed_window_weight(step_frac, start_ratio, end_ratio, mode="cosine"):
    """Guidance weight gate on [start_ratio, end_ratio], zero elsewhere."""
    if end_ratio <= start_ratio:
        return 0.0
    if step_frac < start_ratio or step_frac > end_ratio:
        return 0.0
    s = (step_frac - start_ratio) / max(end_ratio - start_ratio, 1e-8)
    s = min(max(float(s), 0.0), 1.0)
    if mode == "linear":
        return max(0.0, 1.0 - s)
    # default cosine: smooth 1 -> 0 anneal within active window
    return 0.5 * (1.0 + math.cos(math.pi * s))


def clip_rotation_update(delta_xi, max_norm):
    """Clip per-residue SO(3) tangent updates to a maximum norm (radians)."""
    if max_norm is None or max_norm <= 0:
        return delta_xi
    norms = torch.norm(delta_xi, dim=-1, keepdim=True)
    safe_norms = norms.clamp(min=1e-8)
    scale = torch.clamp(max_norm / safe_norms, max=1.0)
    return delta_xi * scale


def _bond_angle(a, b, c):
    """Angle ABC in radians for batched points with shape (..., 3)."""
    v1 = a - b
    v2 = c - b
    n1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    n2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    cosv = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosv)


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle in radians [-pi, pi] for batched points."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-8)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _periodic_sqdiff(x, x0):
    """Periodic squared difference for angles in radians."""
    d = x - x0
    return torch.atan2(torch.sin(d), torch.cos(d)) ** 2


def infer_peptide_sulfur_templates(dataset, disulfide_pairs_local=None, max_ref_frames=4096):
    """Infer per-residue sulfur local coordinates from reference data.

    Returns:
        sulfur_local_nm: (N_pep, 3) tensor, local sulfur coordinates in nm
        sulfur_mask: (N_pep,) bool tensor, True where sulfur template exists
        cb_local_nm: (N_pep, 3) tensor, local CB coordinates in nm (where available)
        cb_mask: (N_pep,) bool tensor, True where CB template exists
        sulfur_atom_names: dict {pep_res_idx: atom_name}
        disulfide_sg_target_nm: dict {(i,j): target_sg_sg_distance_nm}
        disulfide_sg_angle_target_rad: dict {(i,j): (angle_i, angle_j)}
        disulfide_sg_dihedral_target_rad: dict {(i,j): dihedral}
    """
    from frame_utils import coords_to_frames

    info = getattr(dataset, "subset_info", None)
    if info is None:
        print("[Sulfur] subset_info missing; sulfur templates disabled")
        return None, None, None, None, {}, {}, {}, {}

    atom_names = list(info.get("atom_names", []))
    residue_ids = np.asarray(info.get("residue_ids", []), dtype=np.int64)
    categories = list(info.get("atom_categories", [""] * len(atom_names)))
    if len(atom_names) == 0:
        print("[Sulfur] No subset atom metadata; sulfur templates disabled")
        return None, None, None, None, {}, {}, {}, {}

    pep_res_ids = [int(x) for x in np.asarray(dataset.peptide_residue_ids)]
    rid_to_pep = {rid: i for i, rid in enumerate(pep_res_ids)}

    candidates = {}
    for local_idx, (aname, rid, cat) in enumerate(zip(atom_names, residue_ids, categories)):
        rid = int(rid)
        if rid not in rid_to_pep:
            continue
        is_sulfur = (cat == "sulfur") or aname.upper().startswith("S")
        if not is_sulfur:
            continue
        pep_idx = rid_to_pep[rid]
        candidates.setdefault(pep_idx, []).append((aname, local_idx))

    if not candidates:
        print("[Sulfur] No peptide sulfur atoms found in enhanced subset")
        return None, None, None, None, {}, {}, {}, {}

    def _pick_atom(cands):
        def score(item):
            name = item[0].upper()
            if name == "SG":
                return (0, name)
            if name.startswith("S"):
                return (1, name)
            return (2, name)
        return sorted(cands, key=score)[0]

    selected = {pep_idx: _pick_atom(cands) for pep_idx, cands in candidates.items()}

    n_ref = min(int(max_ref_frames), int(dataset.coords_nm.shape[0]))
    coords_ref = torch.from_numpy(dataset.coords_nm[:n_ref]).float()
    pep_bb_idx = torch.from_numpy(dataset.peptide_bb_indices).long()
    pep_R_ref, pep_t_ref, _ = coords_to_frames(coords_ref, pep_bb_idx)  # physical nm

    sulfur_local_nm = torch.zeros(dataset.n_peptide, 3, dtype=torch.float32)
    sulfur_mask = torch.zeros(dataset.n_peptide, dtype=torch.bool)
    cb_local_nm = torch.zeros(dataset.n_peptide, 3, dtype=torch.float32)
    cb_mask = torch.zeros(dataset.n_peptide, dtype=torch.bool)
    sulfur_atom_names = {}
    sulfur_global_idx = {}
    cb_global_idx = {}

    # Find CB indices from subset metadata (needed for S-S angle/dihedral guidance)
    for local_idx, (aname, rid) in enumerate(zip(atom_names, residue_ids)):
        rid = int(rid)
        if rid not in rid_to_pep:
            continue
        if aname.upper() != "CB":
            continue
        pep_idx = rid_to_pep[rid]
        cb_global_idx[int(pep_idx)] = int(local_idx)

    for pep_idx, (atom_name, sulfur_local_idx) in selected.items():
        S_pos = coords_ref[:, sulfur_local_idx, :]        # (B_ref, 3)
        CA_pos = pep_t_ref[:, pep_idx, :]                 # (B_ref, 3)
        rel = S_pos - CA_pos
        local = torch.einsum('bij,bj->bi', pep_R_ref[:, pep_idx].transpose(-1, -2), rel)
        sulfur_local_nm[pep_idx] = local.mean(dim=0)
        sulfur_mask[pep_idx] = True
        sulfur_atom_names[int(pep_idx)] = str(atom_name)
        sulfur_global_idx[int(pep_idx)] = int(sulfur_local_idx)
        if int(pep_idx) in cb_global_idx:
            cb_idx = cb_global_idx[int(pep_idx)]
            CB_pos = coords_ref[:, cb_idx, :]
            rel_cb = CB_pos - CA_pos
            local_cb = torch.einsum('bij,bj->bi', pep_R_ref[:, pep_idx].transpose(-1, -2), rel_cb)
            cb_local_nm[pep_idx] = local_cb.mean(dim=0)
            cb_mask[pep_idx] = True

    disulfide_sg_target_nm = {}
    disulfide_sg_angle_target_rad = {}
    disulfide_sg_dihedral_target_rad = {}
    if disulfide_pairs_local:
        for i, j in disulfide_pairs_local:
            i, j = int(i), int(j)
            if i not in sulfur_global_idx or j not in sulfur_global_idx:
                continue
            si = sulfur_global_idx[i]
            sj = sulfur_global_idx[j]
            d_ij = (coords_ref[:, si, :] - coords_ref[:, sj, :]).norm(dim=-1).mean().item()
            key = (min(i, j), max(i, j))
            disulfide_sg_target_nm[key] = float(d_ij)
            # Angle/dihedral references require CB on both residues
            if i in cb_global_idx and j in cb_global_idx:
                cbi = coords_ref[:, cb_global_idx[i], :]
                cbj = coords_ref[:, cb_global_idx[j], :]
                sgi = coords_ref[:, si, :]
                sgj = coords_ref[:, sj, :]
                a_i = _bond_angle(cbi, sgi, sgj).mean().item()
                a_j = _bond_angle(cbj, sgj, sgi).mean().item()
                dih = _dihedral(cbi, sgi, sgj, cbj)
                dih_mean = torch.atan2(torch.sin(dih).mean(), torch.cos(dih).mean()).item()
                disulfide_sg_angle_target_rad[key] = (float(a_i), float(a_j))
                disulfide_sg_dihedral_target_rad[key] = float(dih_mean)

    n_s = int(sulfur_mask.sum().item())
    print(f"[Sulfur] Built sulfur templates for {n_s}/{dataset.n_peptide} peptide residues")
    if disulfide_sg_target_nm:
        for (i, j), d in disulfide_sg_target_nm.items():
            print(f"[Sulfur] Disulfide SG target ({i},{j}) = {d:.4f} nm")
    if disulfide_sg_angle_target_rad:
        for (i, j), (a_i, a_j) in disulfide_sg_angle_target_rad.items():
            print(f"[Sulfur] Disulfide SG angles ({i},{j}) = ({np.degrees(a_i):.2f}°, {np.degrees(a_j):.2f}°)")
    if disulfide_sg_dihedral_target_rad:
        for (i, j), dih in disulfide_sg_dihedral_target_rad.items():
            print(f"[Sulfur] Disulfide SG dihedral ({i},{j}) = {np.degrees(dih):.2f}°")

    return (
        sulfur_local_nm,
        sulfur_mask,
        cb_local_nm,
        cb_mask,
        sulfur_atom_names,
        disulfide_sg_target_nm,
        disulfide_sg_angle_target_rad,
        disulfide_sg_dihedral_target_rad,
    )


def reconstruct_peptide_sulfur_coords(peptide_R, peptide_t_nm, sulfur_local_nm):
    """Reconstruct peptide sulfur coordinates in physical space (nm)."""
    if sulfur_local_nm is None:
        return None
    return (
        torch.einsum('bnij,nj->bni', peptide_R.float(), sulfur_local_nm.to(peptide_R.device).float())
        + peptide_t_nm.float()
    )


def reconstruct_peptide_cb_coords(peptide_coords_nm):
    """Reconstruct pseudo-Cβ coordinates from peptide backbone coords (nm)."""
    from frame_utils import reconstruct_pseudo_cb

    coords_t = torch.as_tensor(peptide_coords_nm, dtype=torch.float32)
    if coords_t.dim() == 2:
        coords_t = coords_t.unsqueeze(0)
    if coords_t.shape[1] % 4 != 0:
        raise ValueError("Peptide backbone coords must contain [N,CA,C,O] per residue")

    N_pos = coords_t[:, 0::4, :]
    CA_pos = coords_t[:, 1::4, :]
    C_pos = coords_t[:, 2::4, :]
    return reconstruct_pseudo_cb(N_pos, CA_pos, C_pos)


# ─── Velocity computation with CFG (batch-merged single forward pass) ──
def compute_velocity(model, protein_R, protein_t, peptide_R, peptide_t,
                     time_scalar, n_samples, device, cfg_scale=1.0, slow_feat=None,
                     protein_aa=None, peptide_aa=None):
    """CFG with batch merging: cond + uncond in one forward pass."""
    if cfg_scale <= 1.0:
        time = torch.full((n_samples,), time_scalar, device=device)
        return model(protein_R, protein_t, peptide_R, peptide_t,
                     time, slow_feat=slow_feat, cond_mask=None,
                     protein_aa=protein_aa, peptide_aa=peptide_aa)

    # Batch-merge: [cond; uncond] in a single forward pass
    B2 = n_samples * 2
    pR2 = torch.cat([protein_R, protein_R], dim=0)
    pt2 = torch.cat([protein_t, protein_t], dim=0)
    eR2 = torch.cat([peptide_R, peptide_R], dim=0)
    et2 = torch.cat([peptide_t, peptide_t], dim=0)
    time2 = torch.full((B2,), time_scalar, device=device)
    sf2 = torch.cat([slow_feat, slow_feat], dim=0) if slow_feat is not None else None
    cond_mask2 = torch.cat([torch.zeros(n_samples, device=device, dtype=torch.bool),
                            torch.ones(n_samples, device=device, dtype=torch.bool)])
    paa2 = torch.cat([protein_aa, protein_aa], dim=0) if protein_aa is not None and protein_aa.dim() > 1 else protein_aa
    eaa2 = torch.cat([peptide_aa, peptide_aa], dim=0) if peptide_aa is not None and peptide_aa.dim() > 1 else peptide_aa

    pv2, po2, ev2, eo2 = model(
        pR2, pt2, eR2, et2, time2,
        slow_feat=sf2, cond_mask=cond_mask2,
        protein_aa=paa2, peptide_aa=eaa2)

    pv_cond, pv_unc = pv2.chunk(2, dim=0)
    po_cond, po_unc = po2.chunk(2, dim=0)
    ev_cond, ev_unc = ev2.chunk(2, dim=0)
    eo_cond, eo_unc = eo2.chunk(2, dim=0)

    pv = pv_unc + cfg_scale * (pv_cond - pv_unc)
    po = po_unc + cfg_scale * (po_cond - po_unc)
    ev = ev_unc + cfg_scale * (ev_cond - ev_unc)
    eo = eo_unc + cfg_scale * (eo_cond - eo_unc)
    return pv, po, ev, eo


# ─── Heun (2nd-order predictor-corrector) ODE sampler ────────────
def ode_sample_dual(model, n_samples, n_protein, n_peptide, device,
                    sigma_min=1e-4, num_steps=100,
                    cfg_scale=2.0, cfg_warmup_ratio=0.2, slow_feat=None,
                    topo_prot=None, topo_pep=None,
                    topo_weight_protein=1.0, topo_weight_peptide=1.0,
                    disulfide_loss=None, disulfide_weight=0.01,
                    disulfide_sg_pairs=None, disulfide_sg_local_norm=None,
                    disulfide_sg_target_norm=None, disulfide_sg_weight=0.0,
                    disulfide_cb_local_norm=None,
                    disulfide_sg_angle_target_rad=None,
                    disulfide_sg_dihedral_target_rad=None,
                    disulfide_sg_angle_weight=0.0,
                    disulfide_sg_dihedral_weight=0.0,
                    disulfide_sg_rot_weight=0.0,
                    topo_rot_weight=0.5, disulfide_rot_weight=0.1,
                    rot_step_clip_rad=0.05,
                    protein_aa=None, peptide_aa=None,
                    ca_ca_weight=10.0, global_scale=1.0, complex_com=None,
                    tica_guidance_weight=0.0, tica_guidance_clip=0.25,
                    tica_guidance_start_ratio=0.0,
                    tica_guidance_end_ratio=0.75,
                    tica_guidance_anneal_mode="cosine",
                    tica_guidance_flat_bottom=0.50,
                    tica_mean=None, tica_components=None, tica_target=None,
                    anti_m2_weight=0.0, anti_m2_center=None,
                    anti_m2_radius=0.9, anti_m2_clip=0.25,
                    anti_m2_start_ratio=0.0, anti_m2_end_ratio=0.75,
                    anti_m2_anneal_mode="cosine"):
    """Heun ODE sampler: 2nd-order accuracy, no Langevin noise injection.

    Integrates from t=1 (noise) to t≈0 (data) using the predictor-corrector
    scheme:  x_{n+1} = x_n + dt/2 * (v(x_n, t_n) + v(x_pred, t_{n+1}))
    where x_pred = x_n + dt * v(x_n, t_n) is the Euler predictor.

    Physics guidance (topo + disulfide) applied as gradient correction on the
    integrated state, with time-decaying weight (strongest near t=0).
    """
    # Initialize from pure noise
    protein_R = random_so3((n_samples, n_protein), device)
    protein_t = torch.randn(n_samples, n_protein, 3, device=device)
    peptide_R = random_so3((n_samples, n_peptide), device)
    peptide_t = torch.randn(n_samples, n_peptide, 3, device=device)

    dt = -(1.0 - sigma_min) / num_steps
    cfg_warmup_steps = int(cfg_warmup_ratio * num_steps)
    use_topo = (
        topo_prot is not None and topo_pep is not None and
        (topo_weight_protein > 0.0 or topo_weight_peptide > 0.0 or topo_rot_weight > 0.0)
    )
    use_ss = (
        disulfide_loss is not None and
        (disulfide_weight > 0.0 or disulfide_rot_weight > 0.0)
    )
    use_ss_sg = (
        disulfide_sg_pairs is not None and len(disulfide_sg_pairs) > 0 and
        disulfide_sg_local_norm is not None and
        (
            disulfide_sg_target_norm is not None or
            disulfide_sg_angle_target_rad is not None or
            disulfide_sg_dihedral_target_rad is not None
        ) and
        (
            disulfide_sg_weight > 0.0 or
            disulfide_sg_angle_weight > 0.0 or
            disulfide_sg_dihedral_weight > 0.0 or
            disulfide_sg_rot_weight > 0.0
        )
    )
    use_ca_ca = ca_ca_weight > 0.0
    use_tica = (
        tica_guidance_weight > 0.0 and
        tica_mean is not None and
        tica_components is not None and
        tica_target is not None and
        complex_com is not None
    )
    use_anti_m2 = (
        anti_m2_weight > 0.0 and
        tica_mean is not None and
        tica_components is not None and
        anti_m2_center is not None and
        complex_com is not None
    )
    print(f"Heun sampling: {num_steps} steps, CFG warmup: {cfg_warmup_steps} steps -> {cfg_scale}")
    if use_topo:
        print("Topology guidance enabled "
              f"(prot_trans={topo_weight_protein}, pep_trans={topo_weight_peptide}, "
              f"pep_rot={topo_rot_weight})")
    if use_ss:
        print(f"Disulfide geometry guidance enabled (trans={disulfide_weight}, rot={disulfide_rot_weight})")
    if use_ss_sg:
        print("Disulfide SG guidance enabled "
              f"(dist={disulfide_sg_weight}, angle={disulfide_sg_angle_weight}, "
              f"dihedral={disulfide_sg_dihedral_weight}, rot={disulfide_sg_rot_weight})")
    if use_ca_ca:
        print(f"CA-CA distance guidance enabled (weight={ca_ca_weight})")
    if use_tica:
        print(
            "TICA guidance enabled "
            f"(weight={tica_guidance_weight}, clip={tica_guidance_clip}, "
            f"window=[{tica_guidance_start_ratio:.2f}, {tica_guidance_end_ratio:.2f}], "
            f"anneal={tica_guidance_anneal_mode}, flat_bottom={tica_guidance_flat_bottom})"
        )
    if use_anti_m2:
        print(
            "Anti-M2 repulsion enabled "
            f"(weight={anti_m2_weight}, clip={anti_m2_clip}, radius={anti_m2_radius}, "
            f"window=[{anti_m2_start_ratio:.2f}, {anti_m2_end_ratio:.2f}], "
            f"anneal={anti_m2_anneal_mode})"
        )

    for step in tqdm(range(num_steps), desc="ODE Sampling (Heun)"):
        t_cur = 1.0 + step * dt
        t_next = t_cur + dt
        step_frac = float(step) / max(float(num_steps - 1), 1.0)
        tica_gate = 0.0
        if use_tica:
            tica_gate = annealed_window_weight(
                step_frac=step_frac,
                start_ratio=float(tica_guidance_start_ratio),
                end_ratio=float(tica_guidance_end_ratio),
                mode=str(tica_guidance_anneal_mode),
            )
        anti_m2_gate = 0.0
        if use_anti_m2:
            anti_m2_gate = annealed_window_weight(
                step_frac=step_frac,
                start_ratio=float(anti_m2_start_ratio),
                end_ratio=float(anti_m2_end_ratio),
                mode=str(anti_m2_anneal_mode),
            )
        if step < cfg_warmup_steps:
            current_cfg = 1.0 + (cfg_scale - 1.0) * step / max(1, cfg_warmup_steps)
        else:
            current_cfg = cfg_scale

        with torch.no_grad():
            # ── Predictor (Euler step) ──
            pv1, po1, ev1, eo1 = compute_velocity(
                model, protein_R, protein_t, peptide_R, peptide_t,
                t_cur, n_samples, device, current_cfg, slow_feat,
                protein_aa, peptide_aa)

            protein_t_pred = protein_t + dt * pv1
            protein_R_pred = project_so3(protein_R @ so3_exp_map(dt * po1))
            peptide_t_pred = peptide_t + dt * ev1
            peptide_R_pred = project_so3(peptide_R @ so3_exp_map(dt * eo1))

            # ── Corrector (evaluate velocity at predicted state) ──
            pv2, po2, ev2, eo2 = compute_velocity(
                model, protein_R_pred, protein_t_pred,
                peptide_R_pred, peptide_t_pred,
                t_next, n_samples, device, current_cfg, slow_feat,
                protein_aa, peptide_aa)

            # ── Heun update: average of two velocities ──
            protein_t_next = protein_t + dt * 0.5 * (pv1 + pv2)
            peptide_t_next = peptide_t + dt * 0.5 * (ev1 + ev2)
            # SO(3): average the axis-angle increments, then apply
            protein_R_next = project_so3(protein_R @ so3_exp_map(dt * 0.5 * (po1 + po2)))
            peptide_R_next = project_so3(peptide_R @ so3_exp_map(dt * 0.5 * (eo1 + eo2)))

        # ── Physics guidance (gradient correction, no noise) ──
        # Active throughout entire sampling trajectory (not just t<0.5).
        # Gradient clipping prevents harmonic spring forces from exploding.
        if use_topo or use_ss or use_ss_sg or use_ca_ca or use_tica or use_anti_m2:
            with torch.enable_grad():
                protein_t_grad = protein_t_next.detach().requires_grad_(True)
                peptide_t_grad = peptide_t_next.detach().requires_grad_(True)
                peptide_xi = torch.zeros_like(peptide_t_next, dtype=torch.float32,
                                              device=device, requires_grad=True)
                peptide_R_guided = peptide_R_next.detach().float() @ so3_exp_map(peptide_xi)

                topo_loss_prot = torch.tensor(0.0, device=device)
                topo_loss_pep = torch.tensor(0.0, device=device)
                ss_loss = torch.tensor(0.0, device=device)
                sg_loss = torch.tensor(0.0, device=device)
                ca_loss = torch.tensor(0.0, device=device)
                tica_loss = torch.tensor(0.0, device=device)
                tica_dist_mean = torch.tensor(0.0, device=device)
                tica_active_frac = torch.tensor(0.0, device=device)
                anti_m2_loss = torch.tensor(0.0, device=device)
                anti_m2_dist_mean = torch.tensor(0.0, device=device)
                anti_m2_active_frac = torch.tensor(0.0, device=device)
                if use_topo:
                    lb_prot, la_prot = topo_prot(protein_R_next.detach(), protein_t_grad)
                    lb_pep, la_pep = topo_pep(peptide_R_guided, peptide_t_grad)
                    topo_loss_prot = lb_prot + la_prot
                    topo_loss_pep = lb_pep + la_pep
                if use_ss:
                    ss_loss = disulfide_loss(peptide_R_guided, peptide_t_grad)
                if use_ss_sg:
                    sulfur_pos = (
                        torch.einsum('bnij,nj->bni',
                                     peptide_R_guided,
                                     disulfide_sg_local_norm.to(device=device, dtype=torch.float32))
                        + peptide_t_grad
                    )
                    cb_pos = None
                    if disulfide_cb_local_norm is not None:
                        cb_pos = (
                            torch.einsum('bnij,nj->bni',
                                         peptide_R_guided,
                                         disulfide_cb_local_norm.to(device=device, dtype=torch.float32))
                            + peptide_t_grad
                        )
                    sg_terms = []
                    for i, j in disulfide_sg_pairs:
                        key = (min(int(i), int(j)), max(int(i), int(j)))
                        if disulfide_sg_target_norm is not None and key in disulfide_sg_target_norm:
                            d0 = float(disulfide_sg_target_norm[key])
                            dij = (sulfur_pos[:, int(i)] - sulfur_pos[:, int(j)]).norm(dim=-1)
                            sg_terms.append(disulfide_sg_weight * (dij - d0) ** 2)

                        if (
                            cb_pos is not None and
                            disulfide_sg_angle_target_rad is not None and
                            key in disulfide_sg_angle_target_rad and
                            disulfide_sg_angle_weight > 0.0
                        ):
                            a0_i, a0_j = disulfide_sg_angle_target_rad[key]
                            a_i = _bond_angle(cb_pos[:, int(i)], sulfur_pos[:, int(i)], sulfur_pos[:, int(j)])
                            a_j = _bond_angle(cb_pos[:, int(j)], sulfur_pos[:, int(j)], sulfur_pos[:, int(i)])
                            sg_terms.append(disulfide_sg_angle_weight * (a_i - float(a0_i)) ** 2)
                            sg_terms.append(disulfide_sg_angle_weight * (a_j - float(a0_j)) ** 2)

                        if (
                            cb_pos is not None and
                            disulfide_sg_dihedral_target_rad is not None and
                            key in disulfide_sg_dihedral_target_rad and
                            disulfide_sg_dihedral_weight > 0.0
                        ):
                            dih0 = float(disulfide_sg_dihedral_target_rad[key])
                            dih = _dihedral(
                                cb_pos[:, int(i)],
                                sulfur_pos[:, int(i)],
                                sulfur_pos[:, int(j)],
                                cb_pos[:, int(j)],
                            )
                            sg_terms.append(disulfide_sg_dihedral_weight * _periodic_sqdiff(dih, dih0))
                    if sg_terms:
                        sg_loss = torch.stack([x.mean() for x in sg_terms]).mean()
                if use_ca_ca:
                    ca_loss = (
                        ca_ca_guidance_loss(protein_t_grad, global_scale) +
                        ca_ca_guidance_loss(peptide_t_grad, global_scale)
                    )
                if use_tica or use_anti_m2:
                    tic = tica_project(
                        protein_t_grad,
                        peptide_t_grad,
                        tica_mean,
                        tica_components,
                        global_scale,
                        complex_com,
                    )
                if use_tica:
                    tica_loss, tica_dist, tica_active_frac = tica_guidance_loss_from_tic(
                        tic=tic,
                        tica_target=tica_target,
                        flat_bottom_threshold=tica_guidance_flat_bottom,
                    )
                    tica_dist_mean = tica_dist.mean()
                if use_anti_m2:
                    anti_m2_loss, anti_m2_dist, anti_m2_active_frac = anti_m2_repulsion_loss_from_tic(
                        tic=tic,
                        anti_m2_center=anti_m2_center,
                        repel_radius=anti_m2_radius,
                    )
                    anti_m2_dist_mean = anti_m2_dist.mean()

                # Time weight: strongest near t=0 (data end), active over full trajectory
                lambda_t = 1.0 - t_cur

                # Apply topology gradient with strict clipping.
                # Protein and peptide use separate translation weights to avoid
                # peptide over-correction while preserving protein bond integrity.
                if use_topo and topo_weight_protein > 0.0 and topo_loss_prot.requires_grad:
                    need_graph_after_prot_topo = (
                        (topo_weight_peptide > 0.0 or topo_rot_weight > 0.0)
                        or use_ss or use_ss_sg or use_ca_ca or use_tica
                    )
                    grad_pt_topo = torch.autograd.grad(
                        topo_loss_prot, protein_t_grad,
                        retain_graph=need_graph_after_prot_topo,
                        allow_unused=True
                    )[0]
                    if grad_pt_topo is not None and torch.isfinite(grad_pt_topo).all():
                        grad_pt_topo = torch.clamp(grad_pt_topo.detach(), min=-0.5, max=0.5)
                        protein_t_next = protein_t_next - lambda_t * topo_weight_protein * grad_pt_topo

                if use_topo and (topo_weight_peptide > 0.0 or topo_rot_weight > 0.0) and topo_loss_pep.requires_grad:
                    grad_et_topo, grad_exi_topo = torch.autograd.grad(
                        topo_loss_pep, [peptide_t_grad, peptide_xi],
                        retain_graph=(use_ss or use_ss_sg or use_ca_ca or use_tica), allow_unused=True
                    )
                    if grad_et_topo is not None and torch.isfinite(grad_et_topo).all():
                        grad_et_topo = torch.clamp(grad_et_topo.detach(), min=-0.5, max=0.5)
                        peptide_t_next = peptide_t_next - lambda_t * topo_weight_peptide * grad_et_topo
                    if grad_exi_topo is not None and torch.isfinite(grad_exi_topo).all():
                        delta_xi_topo = -lambda_t * topo_rot_weight * grad_exi_topo.detach()
                        delta_xi_topo = clip_rotation_update(delta_xi_topo, rot_step_clip_rad)
                        peptide_R_next = project_so3(
                            peptide_R_next.float() @ so3_exp_map(delta_xi_topo))

                # Apply disulfide gradient with strict clipping (peptide only)
                if use_ss and (disulfide_weight > 0.0 or disulfide_rot_weight > 0.0) and ss_loss.requires_grad:
                    grad_et_ss, grad_exi_ss = torch.autograd.grad(
                        ss_loss, [peptide_t_grad, peptide_xi],
                        retain_graph=(use_ss_sg or use_ca_ca or use_tica), allow_unused=True)
                    if grad_et_ss is not None and torch.isfinite(grad_et_ss).all():
                        grad_et_ss = torch.clamp(grad_et_ss.detach(), min=-0.5, max=0.5)
                        peptide_t_next = peptide_t_next - lambda_t * disulfide_weight * grad_et_ss
                    if grad_exi_ss is not None and torch.isfinite(grad_exi_ss).all():
                        delta_xi_ss = -lambda_t * disulfide_rot_weight * grad_exi_ss.detach()
                        delta_xi_ss = clip_rotation_update(delta_xi_ss, rot_step_clip_rad)
                        peptide_R_next = project_so3(
                            peptide_R_next.float() @ so3_exp_map(delta_xi_ss))

                # Apply explicit sulfur-sulfur guidance (peptide only)
                if use_ss_sg and (
                    disulfide_sg_weight > 0.0 or
                    disulfide_sg_angle_weight > 0.0 or
                    disulfide_sg_dihedral_weight > 0.0 or
                    disulfide_sg_rot_weight > 0.0
                ) and sg_loss.requires_grad:
                    grad_et_sg, grad_exi_sg = torch.autograd.grad(
                        sg_loss, [peptide_t_grad, peptide_xi],
                        retain_graph=(use_ca_ca or use_tica), allow_unused=True
                    )
                    if grad_et_sg is not None and torch.isfinite(grad_et_sg).all():
                        grad_et_sg = torch.clamp(grad_et_sg.detach(), min=-0.5, max=0.5)
                        peptide_t_next = peptide_t_next - lambda_t * grad_et_sg
                    if (
                        disulfide_sg_rot_weight > 0.0 and
                        grad_exi_sg is not None and torch.isfinite(grad_exi_sg).all()
                    ):
                        delta_xi_sg = -lambda_t * disulfide_sg_rot_weight * grad_exi_sg.detach()
                        delta_xi_sg = clip_rotation_update(delta_xi_sg, rot_step_clip_rad)
                        peptide_R_next = project_so3(
                            peptide_R_next.float() @ so3_exp_map(delta_xi_sg))

                # Apply CA-CA distance guidance (protein + peptide)
                if use_ca_ca and ca_loss.requires_grad:
                    grad_pt_ca, grad_et_ca = torch.autograd.grad(
                        ca_loss, [protein_t_grad, peptide_t_grad],
                        retain_graph=use_tica, allow_unused=True)
                    if grad_pt_ca is not None and torch.isfinite(grad_pt_ca).all():
                        grad_pt_ca = torch.clamp(grad_pt_ca.detach(), min=-0.5, max=0.5)
                        protein_t_next = protein_t_next - lambda_t * ca_ca_weight * grad_pt_ca
                    if grad_et_ca is not None and torch.isfinite(grad_et_ca).all():
                        grad_et_ca = torch.clamp(grad_et_ca.detach(), min=-0.5, max=0.5)
                        peptide_t_next = peptide_t_next - lambda_t * ca_ca_weight * grad_et_ca

                need_tica_grad = use_tica and tica_gate > 0.0 and tica_loss.requires_grad
                need_anti_m2_grad = use_anti_m2 and anti_m2_gate > 0.0 and anti_m2_loss.requires_grad
                if need_tica_grad:
                    grad_pt_tica, grad_et_tica = torch.autograd.grad(
                        tica_loss,
                        [protein_t_grad, peptide_t_grad],
                        retain_graph=need_anti_m2_grad,
                        allow_unused=True,
                    )
                    clip = float(tica_guidance_clip)
                    if grad_pt_tica is not None and torch.isfinite(grad_pt_tica).all():
                        grad_pt_tica = torch.clamp(grad_pt_tica.detach(), min=-clip, max=clip)
                        protein_t_next = protein_t_next - tica_gate * tica_guidance_weight * grad_pt_tica
                    if grad_et_tica is not None and torch.isfinite(grad_et_tica).all():
                        grad_et_tica = torch.clamp(grad_et_tica.detach(), min=-clip, max=clip)
                        peptide_t_next = peptide_t_next - tica_gate * tica_guidance_weight * grad_et_tica
                if need_anti_m2_grad:
                    grad_pt_am2, grad_et_am2 = torch.autograd.grad(
                        anti_m2_loss, [protein_t_grad, peptide_t_grad], allow_unused=True
                    )
                    clip = float(anti_m2_clip)
                    if grad_pt_am2 is not None and torch.isfinite(grad_pt_am2).all():
                        grad_pt_am2 = torch.clamp(grad_pt_am2.detach(), min=-clip, max=clip)
                        protein_t_next = protein_t_next - lambda_t * anti_m2_gate * anti_m2_weight * grad_pt_am2
                    if grad_et_am2 is not None and torch.isfinite(grad_et_am2).all():
                        grad_et_am2 = torch.clamp(grad_et_am2.detach(), min=-clip, max=clip)
                        peptide_t_next = peptide_t_next - lambda_t * anti_m2_gate * anti_m2_weight * grad_et_am2

        # ── Update state ──
        protein_R, protein_t = protein_R_next, protein_t_next
        peptide_R, peptide_t = peptide_R_next, peptide_t_next
        protein_t = torch.clamp(protein_t, -3.5, 3.5)
        peptide_t = torch.clamp(peptide_t, -3.5, 3.5)

        if step % 20 == 0:
            msg = (
                f"  step {step}: t={t_cur:.3f}, pv_norm={pv1.norm(dim=-1).mean():.3f}, "
                f"ev_norm={ev1.norm(dim=-1).mean():.3f}"
            )
            if use_tica:
                msg += (
                    f", tica_gate={tica_gate:.3f}, "
                    f"tica_dist={float(tica_dist_mean.detach().cpu()):.3f}, "
                    f"tica_active={float(tica_active_frac.detach().cpu()):.3f}"
                )
            if use_anti_m2:
                msg += (
                    f", anti_m2_gate={anti_m2_gate:.3f}, "
                    f"anti_m2_dist={float(anti_m2_dist_mean.detach().cpu()):.3f}, "
                    f"anti_m2_active={float(anti_m2_active_frac.detach().cpu()):.3f}"
                )
            print(msg)

    print(f"Final normalized range: protein [{protein_t.min():.3f}, {protein_t.max():.3f}], "
          f"peptide [{peptide_t.min():.3f}, {peptide_t.max():.3f}]")
    return protein_R, protein_t, peptide_R, peptide_t


# ─── PDB writer ───────────────────────────────────────────────────
def save_pdb(protein_coords, peptide_coords,
             protein_residue_names, peptide_residue_names, output_path,
             protein_residue_ids=None, peptide_residue_ids=None,
             peptide_cb_coords=None, peptide_cb_mask=None,
             peptide_sulfur_coords=None, peptide_sulfur_mask=None,
             peptide_sulfur_atom_names=None, disulfide_pairs=None,
             disulfide_conect_max_dist_angstrom=2.5):
    """Save multi-model PDB with optional peptide sulfur atoms + CONECT bonds.

    Notes:
      - Input coordinates are in nm and converted to Å in output PDB.
      - Backbone atoms are always emitted as N/CA/C/O.
      - Optional peptide Cβ atoms are emitted where cb_mask=True.
      - Optional sulfur atoms are emitted on peptide residues where mask=True.
      - Optional CONECT is written for CA-CB, CB-SG, and SG-SG disulfide bonds.
    """
    protein_coords = np.asarray(protein_coords)
    peptide_coords = np.asarray(peptide_coords)
    n_models = int(protein_coords.shape[0])
    n_protein = len(protein_residue_names)
    n_peptide = len(peptide_residue_names)

    if protein_residue_ids is None:
        protein_residue_ids = list(range(n_protein))
    if peptide_residue_ids is None:
        peptide_residue_ids = list(range(n_peptide))
    if peptide_cb_mask is None:
        peptide_cb_mask = np.zeros(n_peptide, dtype=bool)
    else:
        peptide_cb_mask = np.asarray(peptide_cb_mask, dtype=bool)
    if peptide_sulfur_mask is None:
        peptide_sulfur_mask = np.zeros(n_peptide, dtype=bool)
    else:
        peptide_sulfur_mask = np.asarray(peptide_sulfur_mask, dtype=bool)
    if peptide_sulfur_atom_names is None:
        peptide_sulfur_atom_names = {}

    def atom_line(serial, atom_name, res_name, chain_id, res_seq, xyz_nm, element):
        x, y, z = (float(xyz_nm[0]) * 10.0, float(xyz_nm[1]) * 10.0, float(xyz_nm[2]) * 10.0)
        return (
            f"ATOM  {serial:5d} {atom_name:>4s} {res_name:>3s} {chain_id:1s}"
            f"{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
        )

    bb_atoms = ('N', 'CA', 'C', 'O')

    def write_bond(fh, a, b):
        if a is None or b is None:
            return
        fh.write(f"CONECT{a:5d}{b:5d}\n")
        fh.write(f"CONECT{b:5d}{a:5d}\n")

    with open(output_path, 'w') as f:
        for m in range(n_models):
            if n_models > 1:
                f.write(f"MODEL     {m + 1:4d}\n")

            serial = 1
            ca_serial_by_pep = {}
            cb_serial_by_pep = {}
            sulfur_serial_by_pep = {}

            # Chain A: protein backbone
            for i, res_name in enumerate(protein_residue_names):
                base = i * 4
                res_seq = int(protein_residue_ids[i]) + 1
                for k, atom_name in enumerate(bb_atoms):
                    f.write(atom_line(
                        serial, atom_name, str(res_name), 'A', res_seq,
                        protein_coords[m, base + k], atom_name[0]
                    ))
                    serial += 1
            f.write("TER\n")

            # Chain B: peptide backbone (+ optional sulfur)
            for i, res_name in enumerate(peptide_residue_names):
                base = i * 4
                res_seq = int(peptide_residue_ids[i]) + 1
                for k, atom_name in enumerate(bb_atoms):
                    f.write(atom_line(
                        serial, atom_name, str(res_name), 'B', res_seq,
                        peptide_coords[m, base + k], atom_name[0]
                    ))
                    if atom_name == 'CA':
                        ca_serial_by_pep[int(i)] = serial
                    serial += 1

                if peptide_cb_coords is not None and peptide_cb_mask[i]:
                    f.write(atom_line(
                        serial, 'CB', str(res_name), 'B', res_seq,
                        peptide_cb_coords[m, i], 'C'
                    ))
                    cb_serial_by_pep[int(i)] = serial
                    serial += 1

                if peptide_sulfur_coords is not None and peptide_sulfur_mask[i]:
                    s_name = str(peptide_sulfur_atom_names.get(int(i), "SG"))[:4]
                    f.write(atom_line(
                        serial, s_name, str(res_name), 'B', res_seq,
                        peptide_sulfur_coords[m, i], 'S'
                    ))
                    sulfur_serial_by_pep[int(i)] = serial
                    serial += 1
            f.write("TER\n")

            # Optional explicit sidechain/disulfide bonds
            for i, cb_serial in cb_serial_by_pep.items():
                write_bond(f, ca_serial_by_pep.get(int(i)), cb_serial)
            for i, sg_serial in sulfur_serial_by_pep.items():
                cb_serial = cb_serial_by_pep.get(int(i))
                if cb_serial is not None:
                    write_bond(f, cb_serial, sg_serial)
                else:
                    write_bond(f, ca_serial_by_pep.get(int(i)), sg_serial)

            if disulfide_pairs:
                for i, j in disulfide_pairs:
                    i, j = int(i), int(j)
                    if i in sulfur_serial_by_pep and j in sulfur_serial_by_pep:
                        if peptide_sulfur_coords is not None:
                            d_nm = np.linalg.norm(peptide_sulfur_coords[m, i] - peptide_sulfur_coords[m, j])
                            if (float(d_nm) * 10.0) <= float(disulfide_conect_max_dist_angstrom):
                                write_bond(f, sulfur_serial_by_pep[i], sulfur_serial_by_pep[j])
                        else:
                            write_bond(f, sulfur_serial_by_pep[i], sulfur_serial_by_pep[j])

            if n_models > 1:
                f.write("ENDMDL\n")

        f.write("END\n")

    print(f"Saved {n_models} samples to {output_path}")


def sanitize_disulfide_conect(pdb_path, max_dist_angstrom=2.5):
    """Remove SG-SG CONECT records if SG-SG distance is unrealistically long."""
    if not os.path.exists(pdb_path):
        return

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    has_model = any(ln.startswith("MODEL") for ln in lines)

    def _process_block(block_lines):
        atom_info = {}
        conect_pairs = []
        kept_non_conect = []

        for ln in block_lines:
            rec = ln[:6].strip()
            if rec in ("ATOM", "HETATM"):
                serial = int(ln[6:11])
                name = ln[12:16].strip().upper()
                x = float(ln[30:38])
                y = float(ln[38:46])
                z = float(ln[46:54])
                atom_info[serial] = (name, np.array([x, y, z], dtype=np.float64))
                kept_non_conect.append(ln)
            elif rec == "CONECT":
                a = int(ln[6:11])
                for k in (11, 16, 21, 26):
                    s = ln[k:k+5].strip()
                    if s:
                        conect_pairs.append((a, int(s)))
            else:
                kept_non_conect.append(ln)

        kept_pairs = []
        for a, b in conect_pairs:
            if a in atom_info and b in atom_info:
                na, pa = atom_info[a]
                nb, pb = atom_info[b]
                if na == "SG" and nb == "SG":
                    d = np.linalg.norm(pa - pb)
                    if float(d) > float(max_dist_angstrom):
                        continue
            kept_pairs.append((a, b))

        insert_idx = len(kept_non_conect)
        for i, ln in enumerate(kept_non_conect):
            rec = ln[:6].strip()
            if rec in ("ENDMDL", "END"):
                insert_idx = i
                break

        conect_lines = [f"CONECT{a:5d}{b:5d}\n" for a, b in kept_pairs]
        return kept_non_conect[:insert_idx] + conect_lines + kept_non_conect[insert_idx:]

    out_lines = []
    if has_model:
        block = []
        for ln in lines:
            block.append(ln)
            if ln[:6].strip() == "ENDMDL":
                out_lines.extend(_process_block(block))
                block = []
        if block:
            out_lines.extend(_process_block(block))
    else:
        out_lines = _process_block(lines)

    with open(pdb_path, "w") as f:
        f.writelines(out_lines)


# ─── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(__file__), "..", "configs", "jnj2113_manuscript.json")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--allow_legacy_checkpoint", action="store_true",
                        help="Explicitly allow archived weights with their original semantics; not manuscript reproduction")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--use_active",       action="store_true")
    parser.add_argument("--inference_context", type=str, default=None,
                        help="Exported fixed metadata; sampling does not read the training NPZ")
    parser.add_argument("--data",             type=str, default=None)
    parser.add_argument("--topology",         type=str, default=None)
    parser.add_argument("--n_samples",        type=int, required=True)
    parser.add_argument("--num_steps",        type=int, default=150)
    parser.add_argument("--cfg_scale",        type=float, default=3.0)
    parser.add_argument("--seed",             type=int, required=True)
    parser.add_argument("--cfg_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--slow_feat",        type=float, nargs="+", default=None)
    parser.add_argument("--slow_feat_stats",  type=str, default=None)
    parser.add_argument("--output",           type=str, default="samples_v12.pdb")
    parser.add_argument("--device",           type=str, default="cuda")
    # ── Optional post-sampling minimization ───────────────────────
    parser.add_argument("--minimize",         action="store_true",
                        help="Deprecated: use graft_and_minimize.py for donor graft and all-atom minimization")
    parser.add_argument("--disulfide_weight", type=float, default=1.0,
                        help="Guidance weight for disulfide Cβ-Cβ proxy loss during sampling (default 1.0)")
    parser.add_argument("--disulfide_rot_weight", type=float, default=0.1,
                        help="Rotation guidance weight for disulfide geometry during sampling (default 0.1)")
    parser.add_argument("--disulfide_sg_weight", type=float, default=0.0,
                        help="Guidance weight for explicit sulfur-sulfur distance loss during sampling (default 0.0)")
    parser.add_argument("--disulfide_sg_angle_weight", type=float, default=0.0,
                        help="Guidance weight for Cβ-SG-SG angle loss during sampling (default 0.0)")
    parser.add_argument("--disulfide_sg_dihedral_weight", type=float, default=0.0,
                        help="Guidance weight for Cβ-SG-SG-Cβ dihedral loss during sampling (default 0.0)")
    parser.add_argument("--disulfide_sg_rot_weight", type=float, default=0.0,
                        help="Rotation guidance weight for explicit sulfur-sulfur loss during sampling (default 0.0)")
    parser.add_argument("--topo_weight",      type=float, default=1.0,
                        help="Guidance weight for topology harmonic penalty during sampling (default 1.0)")
    parser.add_argument("--topo_weight_protein", type=float, default=None,
                        help="Protein translation guidance weight for topology penalty "
                             "(default: follow --topo_weight)")
    parser.add_argument("--topo_weight_peptide", type=float, default=None,
                        help="Peptide translation guidance weight for topology penalty "
                             "(default: follow --topo_weight)")
    parser.add_argument("--topo_rot_weight",  type=float, default=0.5,
                        help="Rotation guidance weight for peptide topology penalty during sampling (default 0.5)")
    parser.add_argument("--rot_step_clip_rad", type=float, default=0.05,
                        help="Max per-residue SO(3) guidance update in radians (default 0.05)")
    parser.add_argument("--ca_ca_weight",     type=float, default=10.0,
                        help="Guidance weight for CA-CA distance harmonic penalty (default 10.0, 0 to disable)")
    parser.add_argument("--tica_guidance_weight", type=float, default=0.0,
                        help="Guidance weight for differentiable linear TICA target loss (default 0 disables)")
    parser.add_argument("--tica_guidance_clip", type=float, default=0.25,
                        help="Per-coordinate gradient clip for TICA guidance (default 0.25)")
    parser.add_argument("--tica_guidance_start_ratio", type=float, default=0.0,
                        help="Guidance gate start fraction of ODE steps (default 0.0)")
    parser.add_argument("--tica_guidance_end_ratio", type=float, default=0.75,
                        help="Guidance gate end fraction of ODE steps; guidance is fully off after this (default 0.75)")
    parser.add_argument("--tica_guidance_anneal_mode", type=str, default="cosine",
                        choices=["cosine", "linear"],
                        help="Anneal mode inside guidance window (default cosine)")
    parser.add_argument("--tica_guidance_flat_bottom", type=float, default=0.50,
                        help="Flat-bottom threshold in TIC distance; guidance active only when dist > threshold (default 0.50)")
    parser.add_argument("--tica_model", type=str, default=None,
                        help="Path to pca_model.pkl/NumpyTICAModel used for TICA guidance")
    parser.add_argument("--anti_m2_weight", type=float, default=0.0,
                        help="Repulsion guidance weight from anti-M2 center in TIC space (default 0.0 disables)")
    parser.add_argument("--anti_m2_center", type=float, nargs=2, default=None,
                        help="Anti-M2 center in TIC space, e.g. --anti_m2_center tic1 tic2")
    parser.add_argument("--anti_m2_radius", type=float, default=0.9,
                        help="Repulsion active radius around anti-M2 center in TIC space (default 0.9)")
    parser.add_argument("--anti_m2_clip", type=float, default=0.25,
                        help="Per-coordinate gradient clip for anti-M2 guidance (default 0.25)")
    parser.add_argument("--anti_m2_start_ratio", type=float, default=0.0,
                        help="Anti-M2 guidance gate start fraction of ODE steps (default 0.0)")
    parser.add_argument("--anti_m2_end_ratio", type=float, default=0.75,
                        help="Anti-M2 guidance gate end fraction of ODE steps (default 0.75)")
    parser.add_argument("--anti_m2_anneal_mode", type=str, default="cosine",
                        choices=["cosine", "linear"],
                        help="Anneal mode for anti-M2 guidance inside active window (default cosine)")
    from configuration import sampling_defaults
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=None)
    preliminary.add_argument("--inference_context", default=None)
    config_args, _ = preliminary.parse_known_args()
    selected_config = config_args.config
    if selected_config is None and config_args.inference_context:
        selected_config = os.path.join(os.path.dirname(os.path.abspath(config_args.inference_context)), "sampling_config.json")
    selected_config = selected_config or default_config
    profile = sampling_defaults(selected_config)
    parser.set_defaults(config=selected_config, **profile)
    args = parser.parse_args()
    if args.n_samples < 1 or args.num_steps < 1 or not 0 <= args.cfg_warmup_ratio <= 1:
        parser.error("Positive sample/step counts and a warmup ratio in [0, 1] are required")

    if args.minimize:
        parser.error("--minimize used a legacy backbone/clash fallback. Use graft_and_minimize.py for donor graft and the two-stage OpenMM protocol.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    import json
    ckpt_dir = os.path.dirname(args.checkpoint)
    model_args = ckpt.get("args")
    if model_args is None:
        with open(os.path.join(ckpt_dir, "args.json")) as f:
            model_args = json.load(f)

    from checkpoint_protocol import validate_checkpoint_protocol, verify_inference_inputs
    protocol_status = validate_checkpoint_protocol(
        model_args, ckpt.get("run_metadata", {}), args.allow_legacy_checkpoint)

    # ── Dataset metadata ──────────────────────────────────────────
    inference_context = None
    data_path = None
    if args.inference_context:
        from inference_context import load_context, restore_dataset
        from pathlib import Path
        if args.data:
            parser.error("--inference_context replaces --data; provide one input mode")
        if args.use_active:
            parser.error("Select the desired parameter set when exporting the inference bundle")
        bundle_dir = Path(args.inference_context).resolve().parent
        topo_path = args.topology or str(bundle_dir / "complex_dry.prmtop")
        bundled_stats = bundle_dir / "condition_stats.npz"
        if not args.slow_feat_stats and bundled_stats.is_file():
            args.slow_feat_stats = str(bundled_stats)
        inference_context = load_context(
            args.inference_context, args.checkpoint, topo_path,
            args.slow_feat_stats, model_args)
        dataset = restore_dataset(inference_context)
    else:
        from frame_dataset_dual import ProteinPeptideFrameDataset
        data_path = args.data or model_args.get("data", "data_aligned_r1234_pca/train_set.npz")
        topo_path = args.topology or model_args.get("topology", "complex_dry.prmtop")
        if protocol_status == "manuscript_configured_run":
            verify_inference_inputs(ckpt.get("run_metadata", {}), data_path, topo_path, args.slow_feat_stats)
        dataset = ProteinPeptideFrameDataset(
            data_path, topo_path,
            translation_scale_mode=model_args.get("translation_scale_mode", "coordinate_rms"))
    n_protein = dataset.n_protein
    n_peptide = dataset.n_peptide
    print(f"Protein residues: {n_protein}, Peptide residues: {n_peptide}")

    # ── Model ─────────────────────────────────────────────────────
    model = ProteinPeptideFrameDenoiser(
        n_protein=n_protein,
        n_peptide=n_peptide,
        c_s=model_args.get("c_s", 256),
        c_z=model_args.get("c_z", 64),
        n_layers=model_args.get("n_layers", 6),
        condition_dim=model_args.get("condition_dim", 128),
        n_heads=model_args.get("n_heads", 8),
        dropout=0.0,
        slow_feat_dim=model_args.get("slow_feat_dim", 0),
        n_topo_heads=model_args.get("n_topo_heads", 2),
        khop=model_args.get("khop", 3),
        topo_path=topo_path,
        peptide_residue_ids=dataset.peptide_residue_ids,
        cyclic_peptide=model_args.get("cyclic_peptide", False),
        disable_tica_conditioning=model_args.get("disable_tica_conditioning", False),
        disable_topology_bias=model_args.get("disable_topology_bias", False),
        disable_interface_attention=model_args.get("disable_interface_attention", False),
        equivariant_translations=model_args.get("equivariant_translations", False),
        topology_graph_scope=model_args.get("topology_graph_scope", "complex"),
    ).to(device)

    if args.use_active and "active_state_dict" in ckpt:
        model.load_state_dict(ckpt["active_state_dict"], strict=True)
        print("Loaded ACTIVE (non-EMA) weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print("Loaded model_state_dict parameters")
    model.eval()

    # ── Slow features ─────────────────────────────────────────────
    slow_feat = None
    if model_args.get("disable_tica_conditioning", False):
        if args.tica_guidance_weight or args.anti_m2_weight:
            raise ValueError("w/o TICA conditioning must not use target-state gradient guidance")
        print("TICA conditioning disabled by checkpoint; requested state is an evaluation label only.")
    elif args.slow_feat:
        if not args.slow_feat_stats:
            raise ValueError("--slow_feat requires the training-fitted --slow_feat_stats file")
        stats = np.load(args.slow_feat_stats)
        mean, std = stats["mean"], stats["std"]
        expected_dim = model_args.get("slow_feat_dim", 0)
        if (mean.shape != (expected_dim,) or std.shape != (expected_dim,)
                or len(args.slow_feat) != expected_dim or not np.isfinite(mean).all()
                or not np.isfinite(args.slow_feat).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Invalid TICA training statistics")
        slow_feat_arr = np.asarray(args.slow_feat, dtype=np.float32)
        slow_feat_norm = (slow_feat_arr - mean) / std
        slow_feat = torch.from_numpy(slow_feat_norm[None, :]).to(
            device=device, dtype=torch.float32
        ).repeat(args.n_samples, 1)
        print(f"Slow features: {args.slow_feat} → normalized: {slow_feat_norm}")

    # ── Optional TICA guidance model ─────────────────────────────
    tica_mean = None
    tica_components = None
    tica_target = None
    anti_m2_center = None
    need_tica_proj = (args.tica_guidance_weight > 0.0) or (args.anti_m2_weight > 0.0)
    if need_tica_proj:
        if not args.tica_model:
            raise ValueError("--tica_guidance_weight or --anti_m2_weight > 0 requires --tica_model")
        try:
            import joblib
            from conformation_space import NumpyPCAModel, NumpyTICAModel
            import __main__
            __main__.NumpyPCAModel = NumpyPCAModel
            __main__.NumpyTICAModel = NumpyTICAModel
            tica_model = joblib.load(args.tica_model)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TICA guidance model: {args.tica_model}") from exc
        tica_mean_np = np.asarray(tica_model.mean_, dtype=np.float32)
        tica_comp_np = np.asarray(tica_model.components_, dtype=np.float32)
        if tica_comp_np.shape[0] != tica_mean_np.shape[0]:
            tica_comp_np = tica_comp_np.T
        expected_dim = int((n_protein + n_peptide) * 3)
        if tica_mean_np.shape[0] != expected_dim or tica_comp_np.shape != (expected_dim, 2):
            raise ValueError(
                "TICA guidance model shape mismatch: "
                f"mean={tica_mean_np.shape}, components={tica_comp_np.shape}, expected=({expected_dim}, 2)"
            )
        tica_mean = torch.from_numpy(tica_mean_np).to(device=device, dtype=torch.float32)
        tica_components = torch.from_numpy(tica_comp_np).to(device=device, dtype=torch.float32)
        print(
            f"[TICA] model loaded (used for guidance={args.tica_guidance_weight > 0.0}, "
            f"anti_m2={args.anti_m2_weight > 0.0})"
        )

    if args.tica_guidance_weight > 0.0:
        if args.slow_feat is None or len(args.slow_feat) < 2:
            raise ValueError("--tica_guidance_weight > 0 requires raw --slow_feat TIC1 TIC2")
        tica_target = torch.as_tensor(args.slow_feat[:2], device=device, dtype=torch.float32)
        print(f"TICA guidance target: {tica_target.detach().cpu().numpy().tolist()}")
    if args.anti_m2_weight > 0.0:
        if args.anti_m2_center is None or len(args.anti_m2_center) != 2:
            raise ValueError("--anti_m2_weight > 0 requires --anti_m2_center TIC1 TIC2")
        anti_m2_center = torch.as_tensor(args.anti_m2_center[:2], device=device, dtype=torch.float32)
        print(f"Anti-M2 center (train-space): {anti_m2_center.detach().cpu().numpy().tolist()}")

    # ── Topology constraints for sampling guidance ───────────────
    # NOTE: Backbone-level (N/CA/C/O) constraints only.
    # Does NOT model sulfur atoms or explicit S-S bond geometry.
    cyclic_peptide = model_args.get("cyclic_peptide", False)
    topo_prot = TopologyHarmonicPenalty(
        N_local=dataset.protein_N_local,
        O_local=dataset.protein_O_local,
        global_scale=dataset.global_scale,
        complex_com=dataset.complex_com,
        topology_path=topo_path,
        cyclic_disulfide=False,
    ).to(device)

    topo_pep = TopologyHarmonicPenalty(
        N_local=dataset.peptide_N_local,
        O_local=dataset.peptide_O_local,
        global_scale=dataset.global_scale,
        complex_com=dataset.complex_com,
        topology_path=topo_path,
        cyclic_disulfide=cyclic_peptide,  # metadata only, no effect on harmonic penalty
    ).to(device)
    print(f"Topology constraints loaded")

    # ── Disulfide geometry guidance (pseudo-Cβ proxy, peptide only) ──
    disulfide_loss = None
    disulfide_pairs_local = []
    sulfur_local_nm = None
    sulfur_mask = None
    cb_local_nm = None
    cb_mask = None
    sulfur_atom_names = {}
    disulfide_sg_target_nm = {}
    disulfide_sg_angle_target_rad = {}
    disulfide_sg_dihedral_target_rad = {}
    if inference_context is not None:
        from inference_context import restore_geometry
        disulfide_loss, sulfur_values = restore_geometry(inference_context, dataset, device)
        disulfide_pairs_local = list(disulfide_loss.disulfide_pairs) if disulfide_loss is not None else []
        (sulfur_local_nm, sulfur_mask, cb_local_nm, cb_mask, sulfur_atom_names,
         disulfide_sg_target_nm, disulfide_sg_angle_target_rad,
         disulfide_sg_dihedral_target_rad) = sulfur_values
    else:
        if cyclic_peptide:
            print("\nBuilding DisulfideGeometryLoss for sampling guidance...")
            disulfide_loss = build_disulfide_loss(
                topology_path=topo_path,
                peptide_residue_ids=dataset.peptide_residue_ids,
                N_local=dataset.peptide_N_local,
                global_scale=dataset.global_scale,
                complex_com=dataset.complex_com,
                peptide_bb_indices=dataset.peptide_bb_indices,
                coords_nm=dataset.coords_nm,
            )
            if disulfide_loss is not None:
                disulfide_loss = disulfide_loss.to(device)
                disulfide_pairs_local = list(disulfide_loss.disulfide_pairs)

            (
                sulfur_local_nm,
                sulfur_mask,
                cb_local_nm,
                cb_mask,
                sulfur_atom_names,
                disulfide_sg_target_nm,
                disulfide_sg_angle_target_rad,
                disulfide_sg_dihedral_target_rad,
            ) = \
                infer_peptide_sulfur_templates(
                    dataset,
                    disulfide_pairs_local=disulfide_pairs_local,
                    max_ref_frames=4096,
                )

    # ── Sample ────────────────────────────────────────────────────
    print(f"\nSampling {args.n_samples} structures (CFG={args.cfg_scale})")
    topo_weight_protein = (
        args.topo_weight if args.topo_weight_protein is None else args.topo_weight_protein
    )
    topo_weight_peptide = (
        args.topo_weight if args.topo_weight_peptide is None else args.topo_weight_peptide
    )
    disulfide_sg_local_norm = None
    disulfide_sg_target_norm = None
    disulfide_cb_local_norm = None
    if sulfur_local_nm is not None:
        disulfide_sg_local_norm = sulfur_local_nm / float(dataset.global_scale)
    if cb_local_nm is not None and cb_mask is not None and bool(cb_mask.any()):
        disulfide_cb_local_norm = cb_local_nm / float(dataset.global_scale)
    if disulfide_sg_target_nm:
        disulfide_sg_target_norm = {
            (int(i), int(j)): float(d_nm) / float(dataset.global_scale)
            for (i, j), d_nm in disulfide_sg_target_nm.items()
        }

    # Prepare AA type indices for sampling
    protein_aa = dataset.protein_aa.to(device)
    peptide_aa = dataset.peptide_aa.to(device)

    pR, pt, eR, et = ode_sample_dual(
        model, args.n_samples, n_protein, n_peptide, device,
        sigma_min=model_args.get("sigma_min", 1e-4),
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        cfg_warmup_ratio=args.cfg_warmup_ratio,
        slow_feat=slow_feat,
        topo_prot=topo_prot,
        topo_pep=topo_pep,
        topo_weight_protein=topo_weight_protein,
        topo_weight_peptide=topo_weight_peptide,
        disulfide_loss=disulfide_loss,
        disulfide_weight=args.disulfide_weight,
        disulfide_sg_pairs=disulfide_pairs_local,
        disulfide_sg_local_norm=disulfide_sg_local_norm,
        disulfide_cb_local_norm=disulfide_cb_local_norm,
        disulfide_sg_target_norm=disulfide_sg_target_norm,
        disulfide_sg_angle_target_rad=disulfide_sg_angle_target_rad,
        disulfide_sg_dihedral_target_rad=disulfide_sg_dihedral_target_rad,
        disulfide_sg_weight=args.disulfide_sg_weight,
        disulfide_sg_angle_weight=args.disulfide_sg_angle_weight,
        disulfide_sg_dihedral_weight=args.disulfide_sg_dihedral_weight,
        disulfide_sg_rot_weight=args.disulfide_sg_rot_weight,
        topo_rot_weight=args.topo_rot_weight,
        disulfide_rot_weight=args.disulfide_rot_weight,
        rot_step_clip_rad=args.rot_step_clip_rad,
        protein_aa=protein_aa,
        peptide_aa=peptide_aa,
        ca_ca_weight=args.ca_ca_weight,
        global_scale=float(dataset.global_scale),
        complex_com=dataset.complex_com,
        tica_guidance_weight=args.tica_guidance_weight,
        tica_guidance_clip=args.tica_guidance_clip,
        tica_guidance_start_ratio=args.tica_guidance_start_ratio,
        tica_guidance_end_ratio=args.tica_guidance_end_ratio,
        tica_guidance_anneal_mode=args.tica_guidance_anneal_mode,
        tica_guidance_flat_bottom=args.tica_guidance_flat_bottom,
        tica_mean=tica_mean,
        tica_components=tica_components,
        tica_target=tica_target,
        anti_m2_weight=args.anti_m2_weight,
        anti_m2_center=anti_m2_center,
        anti_m2_radius=args.anti_m2_radius,
        anti_m2_clip=args.anti_m2_clip,
        anti_m2_start_ratio=args.anti_m2_start_ratio,
        anti_m2_end_ratio=args.anti_m2_end_ratio,
        anti_m2_anneal_mode=args.anti_m2_anneal_mode,
    )

    # ── Denormalize ───────────────────────────────────────────────
    # Training: t_norm = (t_physical - complex_com_per_frame) / global_scale
    # Both protein and peptide share the same per-frame complex COM.
    # At inference we approximate per_frame_com ≈ dataset.complex_com (train mean).
    com = dataset.complex_com.cpu().reshape(1, 1, 3)
    pt_nm = (pt * dataset.global_scale).cpu() + com
    et_nm = (et * dataset.global_scale).cpu() + com

    print(f"Protein t range: [{pt_nm.min():.3f}, {pt_nm.max():.3f}] nm")
    print(f"Peptide t range: [{et_nm.min():.3f}, {et_nm.max():.3f}] nm")

    # ── Reconstruct backbone coordinates ─────────────────────────
    # Build sequential backbone indices (0, 1, 2, 3 for res0; 4, 5, 6, 7 for res1, etc.)
    n_protein_res = len(dataset.protein_residue_names)
    protein_bb_idx = torch.arange(n_protein_res * 4).reshape(n_protein_res, 4)
    n_protein_atoms = n_protein_res * 4
    protein_coords = frames_to_coords(
        pR.cpu(), pt_nm.float().cpu(),
        dataset.protein_O_local, dataset.protein_N_local,
        protein_bb_idx, n_protein_atoms)

    n_peptide_res = len(dataset.peptide_residue_names)
    peptide_bb_idx = torch.arange(n_peptide_res * 4).reshape(n_peptide_res, 4)
    n_peptide_atoms = n_peptide_res * 4
    peptide_coords = frames_to_coords(
        eR.cpu(), et_nm.float().cpu(),
        dataset.peptide_O_local, dataset.peptide_N_local,
        peptide_bb_idx, n_peptide_atoms)

    peptide_cb_coords = None
    peptide_cb_mask = None
    peptide_sulfur_coords = None
    if sulfur_local_nm is not None and sulfur_mask is not None and bool(sulfur_mask.any()):
        peptide_cb_coords = reconstruct_peptide_cb_coords(peptide_coords.float().cpu())
        peptide_cb_mask = sulfur_mask.cpu().numpy()
        peptide_sulfur_coords = reconstruct_peptide_sulfur_coords(
            eR.detach().cpu(), et_nm.float().cpu(), sulfur_local_nm.float().cpu())
        print(f"Sulfur atoms reconstructed for {int(sulfur_mask.sum().item())} peptide residues")

    # ── Save raw PDB ──────────────────────────────────────────────
    save_pdb(
        protein_coords.float().numpy(),
        peptide_coords.float().numpy(),
        dataset.protein_residue_names,
        dataset.peptide_residue_names,
        args.output,
        protein_residue_ids=dataset.protein_residue_ids,
        peptide_residue_ids=dataset.peptide_residue_ids,
        peptide_cb_coords=(
            peptide_cb_coords.float().numpy() if peptide_cb_coords is not None else None
        ),
        peptide_cb_mask=peptide_cb_mask,
        peptide_sulfur_coords=(
            peptide_sulfur_coords.float().numpy() if peptide_sulfur_coords is not None else None
        ),
        peptide_sulfur_mask=(
            sulfur_mask.cpu().numpy() if sulfur_mask is not None else None
        ),
        peptide_sulfur_atom_names=sulfur_atom_names,
        disulfide_pairs=disulfide_pairs_local,
    )

    # Save the effective sampling settings with the raw structures.
    from data_protocol import sha256_file
    metadata = {
        "stage": "raw_generation", "sample_count": args.n_samples,
        "checkpoint_protocol_status": protocol_status,
        "profile_sha256": sha256_file(args.config),
        "matches_reported_sampling_settings": (args.n_samples == 100 and all(
            getattr(args, key) == value for key, value in profile.items())),
        "full_manuscript_reproduction": False,
        "effective_model_semantics": {
            "translation_scale_mode": model_args.get("translation_scale_mode", "coordinate_rms"),
            "equivariant_translations": model_args.get("equivariant_translations", False),
            "topology_graph_scope": model_args.get("topology_graph_scope", "complex"),
        },
        "inference_input_sha256": {"train_data": sha256_file(data_path) if data_path else None,
            "inference_context": sha256_file(args.inference_context) if args.inference_context else None,
            "topology": sha256_file(topo_path),
            "condition_stats": sha256_file(args.slow_feat_stats) if args.slow_feat_stats else None},
        "sampling_seed": args.seed, "arguments": vars(args),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "training_split": ckpt.get("run_metadata", {}).get("split"),
        "ablations": {key: model_args.get(key, False) for key in
                      ("disable_tica_conditioning", "disable_topology_bias", "disable_interface_attention")},
    }
    with open(args.output + ".sampling.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Raw structures saved. All-atom completion precedes separate OpenMM minimization.")


if __name__ == "__main__":
    main()
