#!/usr/bin/env python3
"""
HighStateGen training — SE(3) Flow Matching with Physics-Informed Losses.

Backbone-level (N/CA/C/O) generation with topology-aware constraints.

Loss formula:
  L = λ_t·L_trans + λ_r·L_rot_geodesic + λ_aux·L_aux + λ_dist·L_dist + λ_topo·L_topo + λ_interface·L_interface

NOTE: This is a backbone-level model. Does NOT reconstruct or constrain
      sulfur atoms, explicit S-S bonds, or full sidechain geometry.
      cyclic_peptide enables disulfide-aware topology and a backbone proxy loss.

Removed vs V10/V11:
  - lambda_local      (local velocity-diff MSE)     → replaced by L_aux
  - lambda_rot_local  (local omega-diff MSE)         → replaced by L_aux
  - lambda_chain      (C_i-N_{i+1} 1.33Å penalty)   → replaced by L_topo
  - lambda_dist_local (windowed local distance MSE)  → replaced by L_dist (global)
"""
import wandb
import os
import sys
import json
import shlex
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from frame_flow_matching import (
    frame_flow_interpolate, frame_velocity_target, frame_flow_loss,
    sample_time_logit_normal, pred_frames_from_velocity,
)
from frame_dataset_dual import ProteinPeptideFrameDataset
from ipa_model_dual import ProteinPeptideFrameDenoiser
from configuration import training_defaults, validate_manuscript_arguments
from data_protocol import verify_manifest, sha256_file
from physics_losses import (aux_coord_loss, pairwise_dist_loss, backbone_fape_loss,
                            TopologyHarmonicPenalty, interface_distance_loss,
                            build_disulfide_loss, cross_chain_distogram_loss,
                            torsion_loss, pair_frame_loss)

try:
    from lddt_loss import compute_lddt_loss as _compute_lddt_loss
    _LDDT_AVAILABLE = True
except ImportError:
    _LDDT_AVAILABLE = False


# ─── EMA ─────────────────────────────────────────────────────────
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply(self, model):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])
        self.backup = None

    def state_dict(self):
        return {n: p.clone() for n, p in self.shadow.items()}

    def load_state_dict(self, sd):
        self.shadow = {n: p.clone() for n, p in sd.items()}


def _coerce_wandb_value(current, value):
    if current is None or value is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def init_wandb_run(args):
    use_wandb = args.wandb or bool(os.environ.get("WANDB_SWEEP_ID"))
    if not use_wandb:
        return None, []

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=vars(args).copy(),
    )
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("val/loss", summary="min")

    override_keys = []
    for key, value in dict(wandb.config).items():
        if not hasattr(args, key):
            continue
        new_value = _coerce_wandb_value(getattr(args, key), value)
        if getattr(args, key) != new_value:
            setattr(args, key, new_value)
            override_keys.append(key)

    wandb.config.update(vars(args), allow_val_change=True)
    return run, sorted(override_keys)


def log_wandb_metrics(epoch, train_metrics, val_metrics, lr, best_val_loss, patience_counter):
    if wandb.run is None:
        return

    payload = {
        "epoch": epoch,
        "lr": float(lr),
        "patience_counter": int(patience_counter),
        "best_val_loss": float(best_val_loss),
    }
    payload.update({f"train/{k}": float(v) for k, v in train_metrics.items()})
    if val_metrics is not None:
        payload.update({f"val/{k}": float(v) for k, v in val_metrics.items()})
    wandb.log(payload, step=epoch)


# ─── Core loss computation (shared by train and validate) ─────────
def compute_all_losses(
    protein_v_pred, protein_omega_pred, peptide_v_pred, peptide_omega_pred,
    protein_v_target, protein_omega_target, peptide_v_target, peptide_omega_target,
    protein_R0, protein_t0, peptide_R0, peptide_t0,
    protein_Rs, protein_ts, peptide_Rs, peptide_ts,
    s, args, chain_meta, topo_prot, topo_pep, device, disulfide_loss=None,
):
    """Compute all loss terms. Returns (total_loss, dict_of_components).

    L = λ_t·L_trans + λ_r·L_rot + λ_fape·L_fape + λ_aux·L_aux + λ_dist·L_dist
        + λ_topo·L_topo + λ_interface·L_interface + λ_cross·L_cross + λ_ss·L_disulfide
    """

    # ── L_trans + L_rot_geodesic (flow matching base) ─────────────
    # omega target = log(R0^T R1) — geodesic velocity, NOT matrix subtraction
    loss_protein, loss_protein_trans, loss_protein_rot = frame_flow_loss(
        protein_v_pred.float(), protein_omega_pred.float(),
        protein_v_target.float(), protein_omega_target.float(),
        lambda_rot=args.lambda_rot, reduction="none")

    loss_peptide, loss_peptide_trans, loss_peptide_rot = frame_flow_loss(
        peptide_v_pred.float(), peptide_omega_pred.float(),
        peptide_v_target.float(), peptide_omega_target.float(),
        lambda_rot=args.lambda_rot, reduction="none")

    loss = loss_protein + args.lambda_peptide * loss_peptide

    # ── Predicted clean frames (computed once for all downstream losses) ──
    need_pred = (args.lambda_aux_coord > 0 or args.lambda_dist_global > 0
                 or args.lambda_topo > 0 or args.lambda_fape > 0
                 or args.lambda_disulfide > 0 or args.lambda_cross_dist > 0
                 or args.lambda_interface_dist > 0
                 or args.lambda_torsion > 0 or args.lambda_pair_frame > 0
                 or args.lambda_lddt > 0)
    pred_protein_R0 = pred_protein_t0 = pred_peptide_R0 = pred_peptide_t0 = None
    time_weight_per_sample = torch.tensor(1.0, device=device)
    if need_pred and chain_meta is not None:
        pred_protein_R0, pred_protein_t0 = pred_frames_from_velocity(
            protein_Rs.float(), protein_ts.float(),
            protein_v_pred.float(), protein_omega_pred.float(), s,
            sigma_min=args.sigma_min)
        pred_peptide_R0, pred_peptide_t0 = pred_frames_from_velocity(
            peptide_Rs.float(), peptide_ts.float(),
            peptide_v_pred.float(), peptide_omega_pred.float(), s,
            sigma_min=args.sigma_min)
        # Time-weighting: use milder gating for physics losses
        # Original (1-s)^2 suppresses too aggressively at high noise (s→1)
        # New: 0.1 + 0.9*(1-s) ensures minimum 10% weight even at s=1
        time_weight_per_sample = 0.1 + 0.9 * (1.0 - s)

    # ── L_fape: SE(3)-invariant local frame alignment ─────────────
    # Computes FAPE in each residue's local coordinate system.
    # Clamped at clamp_dist=1.0 nm to suppress gradient explosion early in training.
    loss_fape = torch.tensor(0.0, device=device)
    if args.lambda_fape > 0 and pred_protein_R0 is not None:
        loss_fape = (
            backbone_fape_loss(
                pred_protein_R0, pred_protein_t0,
                protein_R0.float(), protein_t0.float(),
                chain_meta["global_scale"], reduction="none") +
            backbone_fape_loss(
                pred_peptide_R0, pred_peptide_t0,
                peptide_R0.float(), peptide_t0.float(),
                chain_meta["global_scale"], reduction="none")
        )
        loss = loss + args.lambda_fape * loss_fape * time_weight_per_sample

    # ── L_aux: backbone coord reconstruction (Euclidean, no clamp) ─
    loss_aux_coord = torch.tensor(0.0, device=device)
    if args.lambda_aux_coord > 0 and pred_protein_R0 is not None:
        loss_aux_coord = (
            aux_coord_loss(
                pred_protein_R0, pred_protein_t0,
                protein_R0.float(), protein_t0.float(),
                chain_meta["protein_N_local"], chain_meta["protein_O_local"],
                chain_meta["global_scale"], chain_meta["complex_com"], reduction="none") +
            aux_coord_loss(
                pred_peptide_R0, pred_peptide_t0,
                peptide_R0.float(), peptide_t0.float(),
                chain_meta["peptide_N_local"], chain_meta["peptide_O_local"],
                chain_meta["global_scale"], chain_meta["complex_com"], reduction="none")
        )
        loss = loss + args.lambda_aux_coord * loss_aux_coord * time_weight_per_sample

    # ── L_dist: global N×N CA-CA distogram MSE ────────────────────
    # Long-range fold guidance — teaches the model macroscopic topology.
    loss_dist_global = torch.tensor(0.0, device=device)
    if args.lambda_dist_global > 0 and pred_protein_t0 is not None:
        loss_dist_global = (
            pairwise_dist_loss(pred_protein_t0, protein_t0.float(),
                               chain_meta["global_scale"], reduction="none") +
            pairwise_dist_loss(pred_peptide_t0, peptide_t0.float(),
                               chain_meta["global_scale"], reduction="none")
        )
        loss = loss + args.lambda_dist_global * loss_dist_global * time_weight_per_sample

    # ── L_topo: harmonic bond + angle penalty from real prmtop (NO dihedral) ──
    # TopologyHarmonicPenalty operates on per-residue frames — no global atom
    # indices, no dimension mismatch.  Real bond/angle equilibrium values are
    # parsed from the prmtop at init time via parmed.
    loss_topo = torch.tensor(0.0, device=device)
    if args.lambda_topo > 0 and pred_protein_R0 is not None:
        lb_prot, la_prot = topo_prot(pred_protein_R0, pred_protein_t0, reduction="none")
        lb_pep,  la_pep  = topo_pep(pred_peptide_R0, pred_peptide_t0, reduction="none")
        loss_topo = lb_prot + la_prot + lb_pep + la_pep
        # Apply per-sample time weight: mean over batch dimension
        loss = loss + args.lambda_topo * loss_topo * time_weight_per_sample

    # ── L_interface: cross-chain interface CA-CA distance ─────────────
    loss_interface_dist = torch.tensor(0.0, device=device)
    if args.lambda_interface_dist > 0 and pred_protein_t0 is not None:
        loss_interface_dist = interface_distance_loss(
            pred_protein_t0, pred_peptide_t0,
            protein_t0.float(), peptide_t0.float(),
            chain_meta["global_scale"], distance_threshold_nm=0.8, reduction="none")
        loss = loss + args.lambda_interface_dist * loss_interface_dist * time_weight_per_sample

    # ── L_cross_dist: cross-chain global distogram (all protein×peptide pairs) ──
    # Complements interface_distance_loss (which only covers <0.8nm contacts).
    # Supervises ALL cross-chain distances → constrains global docking pose,
    # non-contact geometry, and overall relative chain orientation.
    loss_cross_dist = torch.tensor(0.0, device=device)
    if args.lambda_cross_dist > 0 and pred_protein_t0 is not None:
        loss_cross_dist = cross_chain_distogram_loss(
            pred_protein_t0, pred_peptide_t0,
            protein_t0.float(), peptide_t0.float(),
            chain_meta["global_scale"], reduction="none")
        loss = loss + args.lambda_cross_dist * loss_cross_dist * time_weight_per_sample

    # ── L_disulfide: pseudo-Cβ distance proxy for internal disulfide ──
    # Operates on peptide frames only. Uses backbone geometry to reconstruct
    # pseudo-Cβ and penalizes deviation from reference Cβ-Cβ distance.
    # NOT a head-tail closure loss — targets specific internal CYS pair(s).
    loss_disulfide = torch.tensor(0.0, device=device)
    if args.lambda_disulfide > 0 and disulfide_loss is not None and pred_peptide_R0 is not None:
        loss_disulfide = disulfide_loss(pred_peptide_R0, pred_peptide_t0, reduction="none")
        loss = loss + args.lambda_disulfide * loss_disulfide * time_weight_per_sample

    # ── L_torsion: φ/ψ/ω backbone dihedral supervision ───────────────
    # Teaches the model correct secondary structure angles.
    # Applied to both protein and peptide predicted x0 frames.
    loss_torsion = torch.tensor(0.0, device=device)
    if args.lambda_torsion > 0 and pred_protein_R0 is not None:
        loss_torsion = (
            torsion_loss(pred_protein_R0, pred_protein_t0,
                         protein_R0.float(), protein_t0.float(),
                         chain_meta["protein_N_local"],
                         chain_meta["global_scale"], chain_meta["complex_com"], reduction="none") +
            torsion_loss(pred_peptide_R0, pred_peptide_t0,
                         peptide_R0.float(), peptide_t0.float(),
                         chain_meta["peptide_N_local"],
                         chain_meta["global_scale"], chain_meta["complex_com"], reduction="none")
        )
        loss = loss + args.lambda_torsion * loss_torsion * time_weight_per_sample

    # ── L_pair_frame: residue-pair relative orientation supervision ───
    # Supervises relative rotation + translation between residue pairs.
    # max_seq_sep limits to local neighbourhood to keep cost manageable.
    loss_pair_frame = torch.tensor(0.0, device=device)
    if args.lambda_pair_frame > 0 and pred_protein_R0 is not None:
        loss_pair_frame = (
            pair_frame_loss(pred_protein_R0, pred_protein_t0,
                            protein_R0.float(), protein_t0.float(),
                            chain_meta["global_scale"],
                            max_seq_sep=args.pair_frame_sep, reduction="none") +
            pair_frame_loss(pred_peptide_R0, pred_peptide_t0,
                            peptide_R0.float(), peptide_t0.float(),
                            chain_meta["global_scale"],
                            max_seq_sep=None, reduction="none")   # peptide is small (13 res), use all pairs
        )
        loss = loss + args.lambda_pair_frame * loss_pair_frame * time_weight_per_sample

    # ── L_lddt: local distance difference test ────────────────────────
    # Supervises preservation of local pairwise distances (< 1.5 nm).
    # Uses CA positions reconstructed from predicted x0 frames.
    loss_lddt = torch.tensor(0.0, device=device)
    if args.lambda_lddt > 0 and _LDDT_AVAILABLE and pred_protein_t0 is not None:
        gs = float(chain_meta["global_scale"])
        com = torch.as_tensor(chain_meta["complex_com"],
                              device=device, dtype=torch.float32).reshape(1, 1, 3)
        pred_ca_prot = pred_protein_t0.float() * gs + com   # (B, N_prot, 3) nm
        true_ca_prot = protein_t0.float() * gs + com
        pred_ca_pep  = pred_peptide_t0.float() * gs + com
        true_ca_pep  = peptide_t0.float() * gs + com
        # Concatenate protein + peptide CA for joint LDDT
        pred_ca_all = torch.cat([pred_ca_prot, pred_ca_pep], dim=1)
        true_ca_all = torch.cat([true_ca_prot, true_ca_pep], dim=1)
        loss_lddt = _compute_lddt_loss(pred_ca_all, true_ca_all,
                                       distance_threshold=1.5, reduction="none")
        loss = loss + args.lambda_lddt * loss_lddt * time_weight_per_sample

    components = dict(
        loss=loss,
        protein_trans=loss_protein_trans, protein_rot=loss_protein_rot,
        peptide_trans=loss_peptide_trans,  peptide_rot=loss_peptide_rot,
        fape=loss_fape, aux_coord=loss_aux_coord,
        dist_global=loss_dist_global, topo=loss_topo,
        interface_dist=loss_interface_dist,
        cross_dist=loss_cross_dist,
        disulfide=loss_disulfide,
        torsion=loss_torsion,
        pair_frame=loss_pair_frame,
        lddt=loss_lddt,
        # Raw (unweighted) values for analysis
        fape_raw=loss_fape,
        aux_raw=loss_aux_coord,
        dist_raw=loss_dist_global,
        interface_raw=loss_interface_dist,
        cross_raw=loss_cross_dist,
        topo_raw=loss_topo,
        disulfide_raw=loss_disulfide,
        torsion_raw=loss_torsion,
        pair_frame_raw=loss_pair_frame,
        lddt_raw=loss_lddt,
    )
    return loss.mean(), {name: value.mean() for name, value in components.items()}


# ─── Training epoch ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device, args, epoch,
                scheduler, ema, chain_meta, topo_prot, topo_pep, disulfide_loss=None):
    model.train()
    totals = {k: 0.0 for k in [
        'loss', 'protein_trans', 'protein_rot', 'peptide_trans', 'peptide_rot',
        'fape', 'aux_coord', 'dist_global', 'topo', 'interface_dist',
        'cross_dist', 'disulfide', 'torsion', 'pair_frame', 'lddt',
        'fape_raw', 'aux_raw', 'dist_raw', 'interface_raw', 'cross_raw',
        'topo_raw', 'disulfide_raw', 'torsion_raw', 'pair_frame_raw', 'lddt_raw',
    ]}
    n_batches = 0
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch in pbar:
        protein_R0 = batch["protein_R"].to(device)
        protein_t0 = batch["protein_t"].to(device)
        peptide_R0 = batch["peptide_R"].to(device)
        peptide_t0 = batch["peptide_t"].to(device)
        protein_aa = batch["protein_aa"].to(device)
        peptide_aa = batch["peptide_aa"].to(device)
        B = protein_R0.shape[0]
        slow_feat = batch.get("slow_feat", None)
        if slow_feat is not None:
            slow_feat = slow_feat.to(device)

        cond_mask = None
        if args.cfg_dropout > 0:
            cond_mask = torch.rand(B, device=device) < args.cfg_dropout
            if cond_mask.all():
                cond_mask[0] = False

        s = sample_time_logit_normal(B, device, sigma_min=args.sigma_min)

        from frame_utils import random_so3
        protein_R1 = random_so3((B, protein_R0.shape[1]), device, dtype=protein_R0.dtype)
        protein_t1 = torch.randn_like(protein_t0)
        peptide_R1 = random_so3((B, peptide_R0.shape[1]), device, dtype=peptide_R0.dtype)
        peptide_t1 = torch.randn_like(peptide_t0)

        protein_Rs, protein_ts = frame_flow_interpolate(
            protein_R0, protein_t0, protein_R1, protein_t1, s, args.sigma_min)
        peptide_Rs, peptide_ts = frame_flow_interpolate(
            peptide_R0, peptide_t0, peptide_R1, peptide_t1, s, args.sigma_min)

        protein_v_target, protein_omega_target = frame_velocity_target(
            protein_R0, protein_t0, protein_R1, protein_t1, args.sigma_min)
        peptide_v_target, peptide_omega_target = frame_velocity_target(
            peptide_R0, peptide_t0, peptide_R1, peptide_t1, args.sigma_min)

        with torch.amp.autocast('cuda', enabled=args.bf16, dtype=amp_dtype):
            protein_v_pred, protein_omega_pred, peptide_v_pred, peptide_omega_pred = model(
                protein_Rs, protein_ts, peptide_Rs, peptide_ts, s,
                slow_feat=slow_feat, cond_mask=cond_mask,
                protein_aa=protein_aa, peptide_aa=peptide_aa)

            loss, comp = compute_all_losses(
                protein_v_pred, protein_omega_pred, peptide_v_pred, peptide_omega_pred,
                protein_v_target, protein_omega_target, peptide_v_target, peptide_omega_target,
                protein_R0, protein_t0, peptide_R0, peptide_t0,
                protein_Rs, protein_ts, peptide_Rs, peptide_ts,
                s, args, chain_meta, topo_prot, topo_pep, device, disulfide_loss)

        if not torch.isfinite(loss):
            print(f"\n[WARN] NaN/Inf loss at epoch {epoch}, skipping batch")
            torch.save({'protein_R': protein_R0.cpu(), 'protein_t': protein_t0.cpu(),
                        'peptide_R': peptide_R0.cpu(), 'peptide_t': peptide_t0.cpu(),
                        'time': s.cpu()},
                       os.path.join(args.output_dir, f"nan_batch_e{epoch}.pt"))
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        for k in totals:
            totals[k] += comp[k].item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            prot_t=f"{comp['protein_trans'].item():.4f}",
            prot_r=f"{comp['protein_rot'].item():.4f}",
            pep_t=f"{comp['peptide_trans'].item():.4f}",
            pep_r=f"{comp['peptide_rot'].item():.4f}",
            fape=f"{comp['fape'].item():.4f}",
            aux=f"{comp['aux_coord'].item():.4f}",
            dg=f"{comp['dist_global'].item():.4f}",
            topo=f"{comp['topo'].item():.4f}",
            tor=f"{comp['torsion'].item():.4f}",
            pf=f"{comp['pair_frame'].item():.4f}",
            lddt=f"{comp['lddt'].item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    n_train = max(n_batches, 1)
    return {k: v / n_train for k, v in totals.items()}


# ─── Validation ───────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, args, chain_meta, topo_prot, topo_pep, disulfide_loss=None):
    model.eval()
    totals = {k: 0.0 for k in [
        'loss', 'protein_trans', 'protein_rot', 'peptide_trans', 'peptide_rot',
        'fape', 'aux_coord', 'dist_global', 'topo', 'interface_dist',
        'cross_dist', 'disulfide', 'torsion', 'pair_frame', 'lddt',
    ]}
    n_batches = 0
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    for batch in loader:
        protein_R0 = batch["protein_R"].to(device)
        protein_t0 = batch["protein_t"].to(device)
        peptide_R0 = batch["peptide_R"].to(device)
        peptide_t0 = batch["peptide_t"].to(device)
        protein_aa = batch["protein_aa"].to(device)
        peptide_aa = batch["peptide_aa"].to(device)
        B = protein_R0.shape[0]
        slow_feat = batch.get("slow_feat", None)
        if slow_feat is not None:
            slow_feat = slow_feat.to(device)

        s = sample_time_logit_normal(B, device, sigma_min=args.sigma_min)

        from frame_utils import random_so3
        protein_R1 = random_so3((B, protein_R0.shape[1]), device, dtype=protein_R0.dtype)
        protein_t1 = torch.randn_like(protein_t0)
        peptide_R1 = random_so3((B, peptide_R0.shape[1]), device, dtype=peptide_R0.dtype)
        peptide_t1 = torch.randn_like(peptide_t0)

        protein_Rs, protein_ts = frame_flow_interpolate(
            protein_R0, protein_t0, protein_R1, protein_t1, s, args.sigma_min)
        peptide_Rs, peptide_ts = frame_flow_interpolate(
            peptide_R0, peptide_t0, peptide_R1, peptide_t1, s, args.sigma_min)

        protein_v_target, protein_omega_target = frame_velocity_target(
            protein_R0, protein_t0, protein_R1, protein_t1, args.sigma_min)
        peptide_v_target, peptide_omega_target = frame_velocity_target(
            peptide_R0, peptide_t0, peptide_R1, peptide_t1, args.sigma_min)

        with torch.amp.autocast('cuda', enabled=args.bf16, dtype=amp_dtype):
            protein_v_pred, protein_omega_pred, peptide_v_pred, peptide_omega_pred = model(
                protein_Rs, protein_ts, peptide_Rs, peptide_ts, s,
                slow_feat=slow_feat, cond_mask=None,
                protein_aa=protein_aa, peptide_aa=peptide_aa)

            _, comp = compute_all_losses(
                protein_v_pred, protein_omega_pred, peptide_v_pred, peptide_omega_pred,
                protein_v_target, protein_omega_target, peptide_v_target, peptide_omega_target,
                protein_R0, protein_t0, peptide_R0, peptide_t0,
                protein_Rs, protein_ts, peptide_Rs, peptide_ts,
                s, args, chain_meta, topo_prot, topo_pep, device, disulfide_loss)

        for k in totals:
            totals[k] += comp[k].item()
        n_batches += 1

    n_val = max(n_batches, 1)
    return {k: v / n_val for k, v in totals.items()}


# ─── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(_HERE, "..", "configs", "jnj2113_manuscript.json")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--split_manifest", type=str, default=None)
    parser.add_argument("--translation_scale_mode", choices=["coordinate_rms", "radial_rms"], default="coordinate_rms")
    parser.add_argument("--disable_tica_conditioning", action="store_true")
    parser.add_argument("--disable_topology_bias", action="store_true")
    parser.add_argument("--disable_interface_attention", action="store_true")
    parser.add_argument("--equivariant_translations", action="store_true")
    parser.add_argument("--topology_graph_scope", choices=["peptide", "complex"], default="complex")
    parser.add_argument("--data",         type=str, required=True)
    parser.add_argument("--val_data",     type=str, default=None)
    parser.add_argument("--topology",     type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)

    # Slow features
    parser.add_argument("--slow_feat_path",     type=str, default=None)
    parser.add_argument("--slow_feat_val_path", type=str, default=None)
    parser.add_argument("--slow_feat_dim",      type=int, default=0)
    parser.add_argument("--train_basin_labels", type=str, default=None,
                        help="Optional .npy per-frame basin indices for weighted basin sampling")
    parser.add_argument("--basin_boost", type=float, nargs="+", default=None,
                        help="Optional per-basin multipliers on sampling weights, e.g. --basin_boost 1 1 1 1 3")

    # Model
    parser.add_argument("--c_s",           type=int,   default=256)
    parser.add_argument("--c_z",           type=int,   default=64)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--n_heads",       type=int,   default=8)
    parser.add_argument("--condition_dim", type=int,   default=128)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--n_topo_heads",  type=int,   default=2)
    parser.add_argument("--khop",          type=int,   default=3)

    # Training
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--finetune_extra_epochs", type=int, default=0,
                        help="If >0 and resuming, override end epoch to start_epoch + this value.")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--warmup_ratio",  type=float, default=0.05)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--sigma_min",     type=float, default=1e-4)

    # ── Loss weights ─────────────────────────
    parser.add_argument("--lambda_rot",         type=float, default=1.0,
                        help="λ_r: SO(3) geodesic velocity MSE (log-map target)")
    parser.add_argument("--lambda_peptide",     type=float, default=5.0,
                        help="Weight multiplier for peptide flow loss (compensate for fewer residues)")
    parser.add_argument("--lambda_fape",        type=float, default=5.0,
                        help="λ_fape: SE(3)-invariant FAPE loss (clamped at 1.0 nm)")
    parser.add_argument("--lambda_aux_coord",   type=float, default=1.5,
                        help="λ_aux: backbone N/CA/C/O coord reconstruction L2")
    parser.add_argument("--lambda_dist_global", type=float, default=0.0,
                        help="λ_dist: global N×N CA-CA distogram MSE (V16: disabled)")
    parser.add_argument("--lambda_topo",        type=float, default=0.1,
                        help="λ_topo: harmonic bond+angle penalty from real prmtop (no dihedral)")
    parser.add_argument("--lambda_interface_dist", type=float, default=0.0,
                        help="λ_interface: cross-chain interface CA-CA distance loss (V16: disabled)")
    parser.add_argument("--lambda_cross_dist",    type=float, default=0.0,
                        help="λ_cross: cross-chain global distogram (V16: disabled)")
    parser.add_argument("--lambda_disulfide",    type=float, default=1.0,
                        help="λ_disulfide: pseudo-Cβ distance proxy for internal disulfide ring")
    # RF-like structure supervision losses
    parser.add_argument("--lambda_torsion",      type=float, default=0.0,
                        help="λ_torsion: backbone φ/ψ/ω dihedral angle supervision (V16: disabled)")
    parser.add_argument("--lambda_pair_frame",   type=float, default=0.0,
                        help="λ_pair_frame: residue-pair relative frame (V16: disabled)")
    parser.add_argument("--pair_frame_sep",      type=int, default=6,
                        help="max sequence separation for pair_frame_loss on protein (None=all)")
    parser.add_argument("--lambda_lddt",         type=float, default=0.0,
                        help="λ_lddt: local distance difference test loss (V16: disabled)")

    # Misc
    parser.add_argument("--cyclic_peptide", action="store_true")
    parser.add_argument("--cfg_dropout",    type=float, default=0.15)
    parser.add_argument("--ema_decay",      type=float, default=0.9999)
    parser.add_argument("--bf16",           action="store_true")
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_interval",  type=int,   default=5)
    parser.add_argument("--patience",       type=int,   default=30)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="When resuming, restart LR scheduler from step 0 instead of fast-forwarding.")
    parser.add_argument("--device",         type=str,   default="cuda")
    parser.add_argument("--seed",           type=int,   default=42)

    # W&B
    parser.add_argument("--wandb",              action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",      type=str, default=None)
    parser.add_argument("--wandb_entity",       type=str, default=None)
    parser.add_argument("--wandb_name",         type=str, default=None)
    parser.add_argument("--wandb_group",        type=str, default=None)
    parser.add_argument("--wandb_tags",         nargs="*", default=None)
    parser.add_argument("--wandb_mode",         type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_subdir",   action="store_true",
                        help="Write outputs to output_dir/<wandb_run_id>; recommended for sweeps")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=default_config)
    config_args, _ = config_parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**training_defaults(config_args.config))
    args = parser.parse_args()
    validate_manuscript_arguments(args)
    split_record = None
    if args.split_manifest:
        split_record = verify_manifest(args.split_manifest, {
            "train_data": args.data, "validation_data": args.val_data,
            "train_conditions": args.slow_feat_path,
            "validation_conditions": args.slow_feat_val_path,
            "train_labels": args.train_basin_labels,
        })
    if args.val_data and args.slow_feat_path and not args.slow_feat_val_path:
        raise ValueError("Supply validation-specific --slow_feat_val_path; training conditions cannot be reused")
    if getattr(args, "manuscript_protocol", None) and os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Use a new output directory for each retrained model")

    wandb_run, wandb_override_keys = init_wandb_run(args)
    if wandb_run is not None and (args.wandb_run_subdir or os.environ.get("WANDB_SWEEP_ID")) and not args.resume:
        args.output_dir = os.path.join(args.output_dir, wandb_run.id)

    # W&B sweeps can replace command-line values; bind the effective files too.
    validate_manuscript_arguments(args)
    if args.split_manifest:
        split_record = verify_manifest(args.split_manifest, {
            "train_data": args.data, "validation_data": args.val_data,
            "train_conditions": args.slow_feat_path,
            "validation_conditions": args.slow_feat_val_path,
            "train_labels": args.train_basin_labels})
        if split_record["system"] != "jnj2113":
            raise ValueError("This release trainer supports the JNJ-2113 mapping only")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    args_dict = vars(args).copy()
    if wandb_run is not None:
        wandb.config.update(args_dict, allow_val_change=True)

    cli_argv = sys.argv[1:]
    cli_command = "python " + " ".join(shlex.quote(x) for x in sys.argv)
    loss_weights = {
        "lambda_rot": args.lambda_rot,
        "lambda_peptide": args.lambda_peptide,
        "lambda_fape": args.lambda_fape,
        "lambda_aux_coord": args.lambda_aux_coord,
        "lambda_dist_global": args.lambda_dist_global,
        "lambda_topo": args.lambda_topo,
        "lambda_interface_dist": args.lambda_interface_dist,
        "lambda_cross_dist": args.lambda_cross_dist,
        "lambda_disulfide": args.lambda_disulfide,
        # RF-like structure supervision losses
        "lambda_torsion": args.lambda_torsion,
        "lambda_pair_frame": args.lambda_pair_frame,
        "lambda_lddt": args.lambda_lddt,
    }
    run_metadata = {
        "implementation": "HighStateGen source 2026-10-27",
        "loss_semantics": "per-sample local masks and time weights, then batch mean",
        "split": split_record,
        "config_sha256": sha256_file(args.config) if args.config else None,
        "source_sha256": {name: sha256_file(os.path.join(_HERE, name)) for name in
                          ("train.py", "ipa_model_dual.py", "frame_flow_matching.py", "frame_utils.py",
                           "frame_dataset_dual.py", "physics_losses.py", "lddt_loss.py",
                           "topo_graph_encoder.py", "configuration.py", "data_protocol.py")},
        "argv": cli_argv,
        "command": cli_command,
        "loss_weights": loss_weights,
        "resume_source": args.resume or "",
        "wandb_enabled": wandb_run is not None,
        "wandb_run_id": wandb_run.id if wandb_run is not None else "",
        "wandb_run_name": wandb_run.name if wandb_run is not None else "",
        "wandb_sweep_id": os.environ.get("WANDB_SWEEP_ID", ""),
        "wandb_override_keys": wandb_override_keys,
    }

    if args.split_manifest:
        manifest_path = os.path.abspath(args.split_manifest)
        with open(manifest_path, encoding="utf-8") as handle:
            bound_manifest = json.load(handle)
        stats_path = os.path.join(os.path.dirname(manifest_path),
                                  bound_manifest["outputs"]["condition_stats"]["file"])
        if sha256_file(stats_path) != bound_manifest["outputs"]["condition_stats"]["sha256"]:
            raise ValueError("Condition statistics differ from the prepared split")
        run_metadata["inference_input_sha256"] = {
            "train_data": sha256_file(args.data), "topology": sha256_file(args.topology),
            "condition_stats": sha256_file(stats_path)}
    print(f"Device: {device}")
    print(f"Args: {args_dict}")
    print(f"CLI argv: {cli_argv}")
    print(f"CLI command: {cli_command}")
    print(f"Loss weights: {loss_weights}")
    if wandb_run is not None:
        print(f"W&B run: {wandb_run.name} ({wandb_run.id})")
        if wandb_override_keys:
            print(f"W&B overrides: {wandb_override_keys}")

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    # ── Dataset ──────────────────────────────────────────────────
    print("\nLoading training data...")
    train_ds = ProteinPeptideFrameDataset(
        args.data, args.topology,
        slow_feat_path=args.slow_feat_path,
        basin_label_path=args.train_basin_labels,
        translation_scale_mode=args.translation_scale_mode)
    precomputed = train_ds.get_precomputed()
    n_protein = train_ds.n_protein
    n_peptide = train_ds.n_peptide
    print(f"  Protein: {n_protein} residues, Peptide: {n_peptide} residues")

    val_ds = None
    if args.val_data:
        print("Loading validation data...")
        val_slow = args.slow_feat_val_path
        val_ds = ProteinPeptideFrameDataset(
            args.val_data, args.topology, precomputed=precomputed,
            slow_feat_path=val_slow,
            translation_scale_mode=args.translation_scale_mode)

    if split_record and (len(train_ds) != split_record["counts"]["train"] or val_ds is None
                         or len(val_ds) != split_record["counts"]["validation"]):
        raise ValueError("Dataset lengths differ from the prepared split")

    if getattr(train_ds, "basin_labels", None) is not None:
        basin_idx = np.asarray(train_ds.basin_labels, dtype=np.int64)
        basin_counts = np.bincount(basin_idx)
        basin_weight = np.zeros_like(basin_counts, dtype=np.float64)
        nonzero = basin_counts > 0
        basin_weight[nonzero] = 1.0 / basin_counts[nonzero].astype(np.float64)
        if args.basin_boost is not None:
            boost = np.asarray(args.basin_boost, dtype=np.float64)
            if boost.shape[0] != basin_weight.shape[0]:
                raise ValueError(
                    f"--basin_boost length mismatch: got {boost.shape[0]}, expected {basin_weight.shape[0]}"
                )
            basin_weight = basin_weight * boost
        per_sample_weight = torch.from_numpy(basin_weight[basin_idx]).double()
        sampler = WeightedRandomSampler(
            weights=per_sample_weight,
            num_samples=len(train_ds),
            replacement=True,
        )
        print("\nUsing weighted basin sampling:")
        print(f"  train_basin_labels={args.train_basin_labels}")
        print(f"  basin_counts={basin_counts.tolist()}")
        if args.basin_boost is not None:
            print(f"  basin_boost={np.asarray(args.basin_boost, dtype=np.float64).tolist()}")
        print(f"  basin_weight={basin_weight.tolist()}")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    val_loader = None
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    chain_meta = {
        "global_scale":    train_ds.global_scale,
        "complex_com":     train_ds.complex_com,
        "protein_N_local": train_ds.protein_N_local,
        "protein_O_local": train_ds.protein_O_local,
        "peptide_N_local": train_ds.peptide_N_local,
        "peptide_O_local": train_ds.peptide_O_local,
    }

    # ── Topology harmonic penalty — real prmtop bond/angle params ─
    # TopologyHarmonicPenalty uses parmed to extract true equilibrium
    # values from the prmtop, operating on per-residue frames (no global
    # atom indices, no dimension mismatch with backbone-only tensors).
    # NOTE: Backbone-level (N/CA/C/O) only. Does NOT model sulfur atoms
    # or explicit S-S bond geometry.
    print("\nLoading prmtop bond/angle geometry via parmed...")
    topo_prot = TopologyHarmonicPenalty(
        N_local=train_ds.protein_N_local,
        O_local=train_ds.protein_O_local,
        global_scale=train_ds.global_scale,
        complex_com=train_ds.complex_com,
        topology_path=args.topology,
        cyclic_disulfide=False,
    ).to(device)

    topo_pep = TopologyHarmonicPenalty(
        N_local=train_ds.peptide_N_local,
        O_local=train_ds.peptide_O_local,
        global_scale=train_ds.global_scale,
        complex_com=train_ds.complex_com,
        topology_path=args.topology,
        cyclic_disulfide=args.cyclic_peptide,  # metadata only, no effect on harmonic penalty
    ).to(device)

    # ── Disulfide geometry loss (pseudo-Cβ proxy, peptide only) ───
    # Built only when --cyclic_peptide is set. Parses disulfide pairs from
    # prmtop, computes reference Cβ-Cβ and CA-CA distances from training data.
    disulfide_loss = None
    if args.cyclic_peptide and args.lambda_disulfide > 0:
        print("\nBuilding DisulfideGeometryLoss from prmtop...")
        disulfide_loss = build_disulfide_loss(
            topology_path=args.topology,
            peptide_residue_ids=train_ds.peptide_residue_ids,
            N_local=train_ds.peptide_N_local,
            global_scale=train_ds.global_scale,
            complex_com=train_ds.complex_com,
            peptide_bb_indices=train_ds.peptide_bb_indices,
            coords_nm=train_ds.coords_nm,
        )
        if disulfide_loss is not None:
            disulfide_loss = disulfide_loss.to(device)

    print(f"\nLoss formula (flow + geometric + structure)")
    print(f"  L = λ_t·L_trans + λ_r·L_rot + λ_fape·L_fape + λ_aux·L_aux + λ_dist·L_dist")
    print(f"      + λ_topo·L_topo + λ_interface·L_interface + λ_cross·L_cross + λ_ss·L_disulfide")
    print(f"      + λ_torsion·L_torsion + λ_pair_frame·L_pair_frame + λ_lddt·L_lddt")
    print(f"  λ_r={args.lambda_rot}  λ_fape={args.lambda_fape}  λ_aux={args.lambda_aux_coord}  "
          f"λ_dist={args.lambda_dist_global}  λ_topo={args.lambda_topo}")
    print(f"  λ_interface={args.lambda_interface_dist}  λ_cross={args.lambda_cross_dist}  λ_ss={args.lambda_disulfide}")
    print(f"  λ_torsion={args.lambda_torsion}  λ_pair_frame={args.lambda_pair_frame}  λ_lddt={args.lambda_lddt}")

    # ── Model ─────────────────────────────────────────────────────
    print("\nCreating model...")
    model = ProteinPeptideFrameDenoiser(
        n_protein=n_protein,
        n_peptide=n_peptide,
        c_s=args.c_s,
        c_z=args.c_z,
        n_layers=args.n_layers,
        condition_dim=args.condition_dim,
        n_heads=args.n_heads,
        dropout=args.dropout,
        slow_feat_dim=args.slow_feat_dim,
        n_topo_heads=args.n_topo_heads,
        khop=args.khop,
        topo_path=args.topology,
        peptide_residue_ids=train_ds.peptide_residue_ids,
        cyclic_peptide=args.cyclic_peptide,
        disable_tica_conditioning=args.disable_tica_conditioning,
        disable_topology_bias=args.disable_topology_bias,
        disable_interface_attention=args.disable_interface_attention,
        equivariant_translations=args.equivariant_translations,
        topology_graph_scope=args.topology_graph_scope,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer / EMA ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=not args.bf16)
    ema = EMAModel(model, decay=args.ema_decay)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        resume_args = ckpt.get('args') or {}
        if not isinstance(resume_args, dict):
            resume_args = vars(resume_args)
        for key, default in {"translation_scale_mode": "coordinate_rms",
                             "equivariant_translations": False,
                             "topology_graph_scope": "complex"}.items():
            if resume_args.get(key, default) != getattr(args, key):
                raise ValueError(f"Resume checkpoint uses a different {key}; select matching settings")
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        if 'ema_state_dict' in ckpt:
            try:
                ema.load_state_dict(ckpt['ema_state_dict'])
            except Exception as e:
                print(f"  EMA load failed ({e}), reinitializing from model")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = float('inf')
        print(f"  Resumed from epoch {start_epoch}, best_val_loss reset to inf")

    if args.finetune_extra_epochs > 0 and start_epoch > 0:
        old_epochs = args.epochs
        args.epochs = start_epoch + int(args.finetune_extra_epochs)
        print(f"  finetune_extra_epochs={args.finetune_extra_epochs}: "
              f"override epochs {old_epochs} -> {args.epochs}")

    if args.epochs <= start_epoch:
        raise ValueError(
            f"Invalid epoch plan: start_epoch={start_epoch}, epochs={args.epochs}. "
            "Set a larger --epochs or use --finetune_extra_epochs."
        )

    # ── Scheduler (built after resume/epoch-plan is finalized) ────
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if start_epoch > 0:
        if args.reset_scheduler:
            print("  reset_scheduler=True: scheduler starts from step 0")
        else:
            for _ in range(start_epoch * len(train_loader)):
                scheduler.step()

    planned_epochs = args.epochs - start_epoch
    print(f"  Epoch plan: start={start_epoch}, end={args.epochs - 1}, run={planned_epochs}")

    print(f"\n{'='*60}")
    print(f"Starting HighStateGen training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        train_m = train_epoch(
            model, train_loader, optimizer, scaler, device, args,
            epoch, scheduler, ema, chain_meta, topo_prot, topo_pep, disulfide_loss)

        log = (f"E{epoch:03d} | loss={train_m['loss']:.5f} "
               f"prot_t={train_m['protein_trans']:.4f} prot_r={train_m['protein_rot']:.4f} "
               f"pep_t={train_m['peptide_trans']:.4f} pep_r={train_m['peptide_rot']:.4f} "
               f"aux={train_m['aux_coord']:.4f} dg={train_m['dist_global']:.4f} "
               f"topo={train_m['topo']:.4f} cross={train_m['cross_dist']:.4f} ss={train_m['disulfide']:.4f}")

        # Log raw (unweighted) loss values for gradient balance analysis
        log_raw = (f"  [RAW] fape={train_m['fape_raw']:.4f} aux={train_m['aux_raw']:.4f} "
                   f"dist={train_m['dist_raw']:.4f} iface={train_m['interface_raw']:.4f} "
                   f"cross={train_m['cross_raw']:.4f} topo={train_m['topo_raw']:.4f} ss={train_m['disulfide_raw']:.4f}")

        val_m = None
        if val_loader:
            ema.apply(model)
            val_m = validate(model, val_loader, device, args, chain_meta, topo_prot, topo_pep, disulfide_loss)
            ema.restore(model)
            log += (f" | val={val_m['loss']:.5f} "
                    f"prot_r={val_m['protein_rot']:.4f} "
                    f"aux={val_m['aux_coord']:.4f} dg={val_m['dist_global']:.4f} "
                    f"topo={val_m['topo']:.4f} cross={val_m['cross_dist']:.4f} ss={val_m['disulfide']:.4f}")

        lr_now = optimizer.param_groups[0]["lr"]
        log += f" | lr={lr_now:.2e}"
        print(log)
        print(log_raw)  # Print raw loss values for gradient balance analysis

        entry = {"epoch": epoch, "train": train_m, "lr": lr_now}
        if val_m:
            entry["val"] = val_m
        history.append(entry)

        val_loss = val_m["loss"] if val_m else train_m["loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            ema.apply(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'active_state_dict': ema.backup,
                'ema_state_dict': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args_dict,
                'run_metadata': run_metadata,
            }, os.path.join(args.output_dir, "best_model.pt"))
            ema.restore(model)
            print(f"  -> New best: {best_val_loss:.6f}")
            if wandb.run is not None:
                wandb.summary["best_epoch"] = epoch
                wandb.summary["best_val_loss"] = float(best_val_loss)
                wandb.summary["best_model_path"] = os.path.join(args.output_dir, "best_model.pt")
        else:
            patience_counter += 1

        log_wandb_metrics(epoch, train_m, val_m, lr_now, best_val_loss, patience_counter)
        if wandb.run is not None:
            wandb.log({"epoch/is_best": int(is_best)}, step=epoch)

        if (epoch + 1) % args.save_interval == 0:
            ema.apply(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'active_state_dict': ema.backup,
                'ema_state_dict': ema.shadow,
                'args': args_dict,
                'run_metadata': run_metadata,
            }, os.path.join(args.output_dir, f"ckpt_epoch_{epoch+1}.pt"))
            ema.restore(model)

        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print("\nHighStateGen training complete!")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
