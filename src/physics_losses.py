"""
HighStateGen physics-informed losses.

Modules:
  1. aux_coord_loss      - FAPE-style auxiliary coordinate reconstruction loss
  2. pairwise_dist_loss  - Global CA-CA distogram loss
  3. TopologyHarmonicPenalty - Bond/angle harmonic penalty from AMBER prmtop
  4. DisulfideGeometryLoss   - Pseudo-Cβ distance proxy for internal disulfide ring
"""

import torch
import torch.nn.functional as F
import numpy as np


def _reduce_batch(values, reduction="mean"):
    """Reduce within each sample before optionally averaging over the batch."""
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be 'none' or 'mean'")


# ─── 1. Auxiliary Coordinate Reconstruction Loss ─────────────────────────────

def aux_coord_loss(pred_R0, pred_t0_std, true_R0, true_t0_std,
                   N_local, O_local, global_scale, complex_com, reduction="mean"):
    """FAPE-style loss: reconstruct N/CA/C/O and compare to ground truth.

    Uses predicted frames to reconstruct backbone atom positions in physical
    space, then computes L2 distance against true atom positions.

    Args:
        pred_R0: (B, N, 3, 3) predicted clean rotations
        pred_t0_std: (B, N, 3) predicted clean translations (normalized)
        true_R0: (B, N, 3, 3) ground truth rotations
        true_t0_std: (B, N, 3) ground truth translations (normalized)
        N_local: (N, 3) mean N position in local frame (nm)
        O_local: (N, 3) mean O position in local frame (nm)
        global_scale: scalar normalization factor (nm)
        complex_com: (3,) complex center of mass (nm)
    Returns:
        scalar loss (nm^2)
    """
    device = pred_R0.device
    dtype = torch.float32

    R_pred = pred_R0.float()
    R_true = true_R0.float()

    gs = torch.as_tensor(global_scale, device=device, dtype=dtype)
    com = torch.as_tensor(complex_com, device=device, dtype=dtype).reshape(1, 1, 3)
    N_loc = torch.as_tensor(N_local, device=device, dtype=dtype)   # (N, 3)
    O_loc = torch.as_tensor(O_local, device=device, dtype=dtype)   # (N, 3)

    # Denormalize translations to physical space (nm)
    t_pred = pred_t0_std.float() * gs + com   # (B, N, 3)
    t_true = true_t0_std.float() * gs + com

    # C_local = [BOND_CA_C, 0, 0] by frame definition
    from frame_utils import BOND_CA_C  # type: ignore
    N_res = R_pred.shape[1]
    C_loc = torch.zeros(N_res, 3, device=device, dtype=dtype)
    C_loc[:, 0] = BOND_CA_C

    def reconstruct(R, t):
        CA = t                                                          # (B, N, 3)
        C  = torch.einsum('bnij,nj->bni', R, C_loc) + t
        N  = torch.einsum('bnij,nj->bni', R, N_loc) + t
        O  = torch.einsum('bnij,nj->bni', R, O_loc) + t
        return torch.stack([N, CA, C, O], dim=2)                       # (B, N, 4, 3)

    pred_coords = reconstruct(R_pred, t_pred)
    true_coords = reconstruct(R_true, t_true)

    # L2 distance per atom, mean over all
    loss = _reduce_batch(torch.linalg.vector_norm(pred_coords - true_coords, dim=-1), reduction)
    return loss


# ─── 2. Global Pairwise Distance (Distogram) Loss ────────────────────────────

def pairwise_dist_loss(pred_t0_std, true_t0_std, global_scale, reduction="mean"):
    """Full N×N CA-CA pairwise distance MSE.

    Provides long-range topological guidance that local chain loss cannot.

    Args:
        pred_t0_std: (B, N, 3) predicted CA positions (normalized)
        true_t0_std: (B, N, 3) ground truth CA positions (normalized)
        global_scale: scalar (nm) — converts to physical units before distance
    Returns:
        scalar loss (nm^2)
    """
    gs = float(global_scale)
    pred = pred_t0_std.float() * gs   # (B, N, 3)
    true = true_t0_std.float() * gs

    # (B, N, N)
    pred_dist = torch.cdist(pred, pred)
    true_dist = torch.cdist(true, true)

    # Upper triangle only (avoid double-counting and diagonal)
    N = pred.shape[1]
    mask = torch.triu(torch.ones(N, N, device=pred.device, dtype=torch.bool), diagonal=1)
    loss = _reduce_batch((pred_dist[:, mask] - true_dist[:, mask]) ** 2, reduction)
    return loss


# ─── 3. Topology Harmonic Penalty ────────────────────────────────────────────

def _parse_backbone_geometry(topology_path):
    """Extract ideal backbone bond lengths and angles from a prmtop file.

    Uses parmed to read the real AMBER force field parameters for the three
    backbone bond types (N-CA, CA-C, C-N) and three backbone angle types
    (N-CA-C, CA-C-N, C-N-CA).  Falls back to standard AMBER values if a
    bond/angle type is not found in the topology.

    Returns:
        bonds:  dict {(name_i, name_j): r0_nm}
        angles: dict {(name_i, name_j, name_k): theta0_rad}
    """
    import parmed as pmd

    FALLBACK_BONDS = {
        ('N',  'CA'): 0.1465,
        ('CA', 'C' ): 0.1529,
        ('C',  'N' ): 0.1335,
    }
    FALLBACK_ANGLES = {
        ('N',  'CA', 'C' ): np.radians(110.6),
        ('CA', 'C',  'N' ): np.radians(116.2),
        ('C',  'N',  'CA'): np.radians(121.7),
    }

    try:
        struct = pmd.load_file(topology_path)
    except Exception as e:
        print(f"[TopologyHarmonicPenalty] parmed load failed ({e}), using fallback geometry")
        return FALLBACK_BONDS, FALLBACK_ANGLES

    # ── Bond lengths ──────────────────────────────────────────────
    # Accumulate per bond-type and average (handles multiple residue types)
    bond_sums  = {k: [] for k in FALLBACK_BONDS}
    for bond in struct.bonds:
        if bond.type is None:
            continue
        n1, n2 = bond.atom1.name, bond.atom2.name
        r0_nm = bond.type.req * 0.1  # Angstrom → nm
        for key in [(n1, n2), (n2, n1)]:
            if key in bond_sums:
                bond_sums[key].append(r0_nm)

    bonds = {}
    for key, vals in bond_sums.items():
        if vals:
            bonds[key] = float(np.mean(vals))
            print(f"  [prmtop bond] {key}: {bonds[key]*10:.4f} Å  "
                  f"(n={len(vals)}, fallback={FALLBACK_BONDS[key]*10:.4f} Å)")
        else:
            bonds[key] = FALLBACK_BONDS[key]
            print(f"  [prmtop bond] {key}: not found, using fallback {FALLBACK_BONDS[key]*10:.4f} Å")

    # ── Bond angles ───────────────────────────────────────────────
    angle_sums = {k: [] for k in FALLBACK_ANGLES}
    for angle in struct.angles:
        if angle.type is None:
            continue
        n1, n2, n3 = angle.atom1.name, angle.atom2.name, angle.atom3.name
        theta0_rad = np.radians(angle.type.theteq)
        for key in [(n1, n2, n3), (n3, n2, n1)]:
            if key in angle_sums:
                angle_sums[key].append(theta0_rad)

    angles = {}
    for key, vals in angle_sums.items():
        if vals:
            angles[key] = float(np.mean(vals))
            print(f"  [prmtop angle] {key}: {np.degrees(angles[key]):.2f}°  "
                  f"(n={len(vals)}, fallback={np.degrees(FALLBACK_ANGLES[key]):.2f}°)")
        else:
            angles[key] = FALLBACK_ANGLES[key]
            print(f"  [prmtop angle] {key}: not found, using fallback "
                  f"{np.degrees(FALLBACK_ANGLES[key]):.2f}°")

    return bonds, angles


# Spring constants — calibrated so raw topo loss lands in ~0.1-1.0 range,
# comparable to FAPE/aux/dist losses.  Previous values (1000/100) produced
# topo_raw ~500-600, drowning out all other gradient signals.
K_BOND  = 10.0
K_ANGLE = 1.0


def _bond_angle(a, b, c):
    """Compute bond angle at atom b given positions a, b, c.
    Args: each (B, N, 3)
    Returns: (B, N) angles in radians
    """
    ba = F.normalize(a - b, dim=-1)
    bc = F.normalize(c - b, dim=-1)
    cos = (ba * bc).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos)


class TopologyHarmonicPenalty(torch.nn.Module):
    """Harmonic bond + angle penalty using real prmtop geometry.

    Operates on per-residue frames (R, t) — does NOT use global atom indices,
    so it works correctly on the separated protein/peptide tensors.

    Bond and angle equilibrium values are parsed from the real prmtop via
    parmed, falling back to standard AMBER values if not found.

    Does NOT penalize dihedrals — phi/psi remain free for conformational change.

    IMPORTANT: This is a backbone-level (N/CA/C/O) constraint only.
    - Does NOT reconstruct or constrain sulfur atoms
    - Does NOT impose explicit S-S bond geometry
    - Does NOT output full sidechain structures

    Args:
        N_local:       (N_res, 3) mean N position in local frame (nm)
        O_local:       (N_res, 3) mean O position in local frame (nm)
        global_scale:  scalar normalization factor
        complex_com:   (3,) complex COM
        topology_path: path to .prmtop — used to extract real bond/angle params
        k_bond:        spring constant for bonds (kJ/mol/nm^2 equivalent scale)
        k_angle:       spring constant for angles (kJ/mol/rad^2 equivalent scale)
        cyclic_disulfide: METADATA ONLY. Does NOT change bond/angle penalty behavior.
                          Backbone harmonic penalty is identical for linear and
                          disulfide-cyclized peptides (both use sequential C_i-N_{i+1}
                          bonds only, no head-tail closure). Actual disulfide geometry
                          is handled by DisulfideGeometryLoss.
    """

    def __init__(self, N_local, O_local, global_scale, complex_com,
                 topology_path=None, k_bond=K_BOND, k_angle=K_ANGLE, cyclic_disulfide=False):
        super().__init__()
        self.register_buffer('N_local', torch.as_tensor(N_local, dtype=torch.float32))
        self.register_buffer('O_local', torch.as_tensor(O_local, dtype=torch.float32))
        self.global_scale = float(global_scale)
        self.register_buffer('complex_com',
                             torch.as_tensor(complex_com, dtype=torch.float32).reshape(1, 1, 3))
        self.k_bond  = k_bond
        self.k_angle = k_angle
        self.cyclic_disulfide = cyclic_disulfide  # metadata only, no effect on loss formula

        if topology_path is not None:
            print(f"\n[TopologyHarmonicPenalty] Parsing real geometry from {topology_path}")
            ideal_bonds, ideal_angles = _parse_backbone_geometry(topology_path)
        else:
            print("[TopologyHarmonicPenalty] No topology_path — using hardcoded AMBER fallback")
            ideal_bonds  = {('N','CA'): 0.1465, ('CA','C'): 0.1529, ('C','N'): 0.1335}
            ideal_angles = {('N','CA','C'): np.radians(110.6),
                            ('CA','C','N'): np.radians(116.2),
                            ('C','N','CA'): np.radians(121.7)}

        self.r0_NCA = ideal_bonds[('N',  'CA')]
        self.r0_CAC = ideal_bonds[('CA', 'C' )]
        self.r0_CN  = ideal_bonds[('C',  'N' )]
        self.t0_NCAc = ideal_angles[('N',  'CA', 'C' )]
        self.t0_CACn = ideal_angles[('CA', 'C',  'N' )]
        self.t0_CNca = ideal_angles[('C',  'N',  'CA')]

    def _reconstruct(self, R, t_std):
        """Reconstruct N, CA, C positions in physical space."""
        from frame_utils import BOND_CA_C  # type: ignore
        device, dtype = R.device, torch.float32
        R = R.float()
        N_res = R.shape[1]

        t_phys = t_std.float() * self.global_scale + self.complex_com.to(device)

        C_loc = torch.zeros(N_res, 3, device=device, dtype=dtype)
        C_loc[:, 0] = BOND_CA_C

        CA = t_phys
        C  = torch.einsum('bnij,nj->bni', R, C_loc) + t_phys
        N  = torch.einsum('bnij,nj->bni', R, self.N_local.to(device)) + t_phys
        return N, CA, C

    def forward(self, pred_R0, pred_t0_std, reduction="mean"):
        """
        Args:
            pred_R0:      (B, N, 3, 3)
            pred_t0_std:  (B, N, 3)
        Returns:
            loss_bond, loss_angle: scalars
        """
        N_pos, CA_pos, C_pos = self._reconstruct(pred_R0, pred_t0_std)

        bond_losses  = []
        angle_losses = []

        # ── Intra-residue bonds ──────────────────────────────────────
        d_NCA = (N_pos - CA_pos).norm(dim=-1)
        bond_losses.append(self.k_bond * (d_NCA - self.r0_NCA) ** 2)

        d_CAC = (CA_pos - C_pos).norm(dim=-1)
        bond_losses.append(self.k_bond * (d_CAC - self.r0_CAC) ** 2)

        # ── Inter-residue peptide bond C_i - N_{i+1} ────────────────
        # Only applies to linearly adjacent residues (i, i+1).
        # Head-tail bond is naturally excluded by the slicing [:, :-1] and [:, 1:],
        # which only covers sequential pairs — this is NOT controlled by cyclic_disulfide flag.
        if N_pos.shape[1] > 1:
            d_CN = (C_pos[:, :-1] - N_pos[:, 1:]).norm(dim=-1)
            bond_losses.append(self.k_bond * (d_CN - self.r0_CN) ** 2)

        # ── Bond angles ──────────────────────────────────────────────
        ang_NCAc = _bond_angle(N_pos, CA_pos, C_pos)
        angle_losses.append(self.k_angle * (ang_NCAc - self.t0_NCAc) ** 2)

        if N_pos.shape[1] > 1:
            ang_CACn = _bond_angle(CA_pos[:, :-1], C_pos[:, :-1], N_pos[:, 1:])
            angle_losses.append(self.k_angle * (ang_CACn - self.t0_CACn) ** 2)

            ang_CNca = _bond_angle(C_pos[:, :-1], N_pos[:, 1:], CA_pos[:, 1:])
            angle_losses.append(self.k_angle * (ang_CNca - self.t0_CNca) ** 2)

        loss_bond = _reduce_batch(torch.cat(bond_losses, dim=1), reduction)
        loss_angle = _reduce_batch(torch.cat(angle_losses, dim=1), reduction)
        return loss_bond, loss_angle


# ─── 4. Disulfide Geometry Loss ───────────────────────────────────────────────

class DisulfideGeometryLoss(torch.nn.Module):
    """Pseudo-Cβ distance proxy for internal disulfide ring geometry.

    Operates on backbone frames (N/CA/C/O) only. Reconstructs pseudo-Cβ
    positions from backbone atoms using standard geometry, then penalizes
    deviation of the Cβ-Cβ distance from the reference value.

    This is NOT a head-tail closure loss. It targets a specific internal
    disulfide bond between two known residue indices.

    Args:
        disulfide_pairs_local: list of (i, j) local residue index pairs
                               within the peptide (0-indexed)
        N_local:       (N_res, 3) mean N position in local frame (nm)
        global_scale:  scalar normalization factor
        complex_com:   (3,) complex COM
        ref_cbcb_dist: dict mapping (i,j) -> reference Cβ-Cβ distance (nm)
                       If None, uses standard disulfide geometry (0.38 nm)
        k_cbcb:        spring constant for Cβ-Cβ distance penalty
        k_caca:        spring constant for auxiliary CA-CA distance penalty
                       (set to 0 to disable)
    """

    # Standard disulfide Cβ-Cβ distance (nm)
    # Typical range: 0.36-0.40 nm, mean ~0.38 nm
    DEFAULT_CBCB_DIST = 0.38

    def __init__(self, disulfide_pairs_local, N_local, global_scale, complex_com,
                 ref_cbcb_dist=None, ref_caca_dist=None, k_cbcb=100.0, k_caca=10.0):
        super().__init__()
        self.disulfide_pairs = disulfide_pairs_local
        self.register_buffer('N_local', torch.as_tensor(N_local, dtype=torch.float32))
        self.global_scale = float(global_scale)
        self.register_buffer('complex_com',
                             torch.as_tensor(complex_com, dtype=torch.float32).reshape(1, 1, 3))
        self.k_cbcb = k_cbcb
        self.k_caca = k_caca

        # Reference Cβ-Cβ and CA-CA distances per pair
        if ref_cbcb_dist is None:
            ref_cbcb_dist = {}
        if ref_caca_dist is None:
            ref_caca_dist = {}
        self.ref_cbcb_dist = {}
        self.ref_caca_dist = {}
        for i, j in disulfide_pairs_local:
            key = (int(i), int(j))
            self.ref_cbcb_dist[key] = float(ref_cbcb_dist.get(key, self.DEFAULT_CBCB_DIST))
            # CA-CA default: typically ~0.65-0.75 nm for disulfide bonds (longer than Cβ-Cβ)
            # Approximate as Cβ-Cβ + 0.27 nm based on typical geometry
            self.ref_caca_dist[key] = float(ref_caca_dist.get(key,
                                            self.ref_cbcb_dist[key] + 0.27))

        if disulfide_pairs_local:
            pairs_str = ", ".join(
                f"({i},{j}) d_cbcb={self.ref_cbcb_dist[(i,j)]:.3f}nm "
                f"d_caca={self.ref_caca_dist[(i,j)]:.3f}nm"
                for i, j in disulfide_pairs_local)
            print(f"[DisulfideGeometryLoss] pairs: {pairs_str}")

    def _reconstruct(self, R, t_std):
        """Reconstruct N, CA, C in physical space from frames."""
        from frame_utils import BOND_CA_C  # type: ignore
        device = R.device
        R = R.float()
        N_res = R.shape[1]

        t_phys = t_std.float() * self.global_scale + self.complex_com.to(device)

        C_loc = torch.zeros(N_res, 3, device=device, dtype=torch.float32)
        C_loc[:, 0] = BOND_CA_C

        CA = t_phys
        C  = torch.einsum('bnij,nj->bni', R, C_loc) + t_phys
        N  = torch.einsum('bnij,nj->bni', R, self.N_local.to(device)) + t_phys
        return N, CA, C

    def forward(self, pred_R0, pred_t0_std, reduction="mean"):
        """
        Args:
            pred_R0:      (B, N_pep, 3, 3) predicted clean peptide rotations
            pred_t0_std:  (B, N_pep, 3)   predicted clean peptide translations (normalized)
        Returns:
            scalar loss
        """
        if not self.disulfide_pairs:
            return _reduce_batch(pred_t0_std.sum(dim=(1, 2))[:, None] * 0.0, reduction)

        from frame_utils import reconstruct_pseudo_cb  # type: ignore

        N_pos, CA_pos, C_pos = self._reconstruct(pred_R0, pred_t0_std)
        CB_pos = reconstruct_pseudo_cb(N_pos, CA_pos, C_pos)  # (B, N_pep, 3)

        losses = []
        for i, j in self.disulfide_pairs:
            key = (int(i), int(j))
            d0 = self.ref_cbcb_dist[key]

            # Cβ-Cβ distance
            d_cbcb = (CB_pos[:, i] - CB_pos[:, j]).norm(dim=-1)  # (B,)
            losses.append(self.k_cbcb * (d_cbcb - d0) ** 2)

            # Auxiliary CA-CA distance (weaker, as sanity check)
            if self.k_caca > 0:
                d_caca = (CA_pos[:, i] - CA_pos[:, j]).norm(dim=-1)
                # CA-CA ref stored separately; default: Cβ-Cβ - 0.10 nm
                d_caca_ref = self.ref_caca_dist.get(key, d0 - 0.10)
                losses.append(self.k_caca * (d_caca - d_caca_ref) ** 2)

        return _reduce_batch(torch.stack(losses, dim=1), reduction)


def build_disulfide_loss(topology_path, peptide_residue_ids,
                         N_local, global_scale, complex_com,
                         peptide_bb_indices=None, coords_nm=None,
                         k_cbcb=100.0, k_caca=10.0):
    """Factory: parse disulfide bonds from prmtop and build DisulfideGeometryLoss.

    Args:
        topology_path:       path to .prmtop
        peptide_residue_ids: array of global residue indices for the peptide
        N_local:             (N_pep, 3) mean N position in local frame (nm)
        global_scale:        scalar normalization factor
        complex_com:         (3,) complex COM
        peptide_bb_indices:  (N_pep, 4) [N, CA, C, O] atom indices into coords_nm
                             Required if coords_nm is provided.
        coords_nm:           (N_samples, N_atoms, 3) reference coords (nm)
                             Used to compute reference Cβ-Cβ and CA-CA distances.
        k_cbcb:              spring constant for Cβ-Cβ penalty
        k_caca:              spring constant for auxiliary CA-CA penalty

    Returns:
        DisulfideGeometryLoss instance (or None if no disulfide bonds found)
    """
    from physics_constraints import extract_disulfide_bonds  # type: ignore

    global_pairs = extract_disulfide_bonds(topology_path)
    if not global_pairs:
        print("[build_disulfide_loss] No disulfide bonds found in topology")
        return None

    pep_ids = list(peptide_residue_ids)
    g2l = {int(g): l for l, g in enumerate(pep_ids)}

    local_pairs = []
    for g1, g2 in global_pairs:
        if g1 in g2l and g2 in g2l:
            local_pairs.append((g2l[g1], g2l[g2]))
        else:
            print(f"[build_disulfide_loss] Disulfide pair ({g1},{g2}) not in peptide — skipping")

    if not local_pairs:
        print("[build_disulfide_loss] No disulfide bonds within peptide residues")
        return None

    print(f"[build_disulfide_loss] Found {len(local_pairs)} disulfide pair(s) in peptide: {local_pairs}")

    # Compute reference Cβ-Cβ and CA-CA distances from data
    ref_cbcb_dist = {}
    ref_caca_dist = {}
    if coords_nm is not None and peptide_bb_indices is not None:
        import torch
        from frame_utils import reconstruct_pseudo_cb  # type: ignore

        print("[build_disulfide_loss] Computing reference distances from training data...")
        bb = peptide_bb_indices  # (N_pep, 4): [N, CA, C, O]
        coords_t = torch.from_numpy(coords_nm).float()

        for i, j in local_pairs:
            key = (int(i), int(j))
            # Extract backbone atoms for residues i and j
            N_i  = coords_t[:, bb[i, 0], :]
            CA_i = coords_t[:, bb[i, 1], :]
            C_i  = coords_t[:, bb[i, 2], :]
            N_j  = coords_t[:, bb[j, 0], :]
            CA_j = coords_t[:, bb[j, 1], :]
            C_j  = coords_t[:, bb[j, 2], :]

            CB_i = reconstruct_pseudo_cb(N_i, CA_i, C_i)
            CB_j = reconstruct_pseudo_cb(N_j, CA_j, C_j)

            d_cbcb = (CB_i - CB_j).norm(dim=-1).mean().item()
            d_caca = (CA_i - CA_j).norm(dim=-1).mean().item()

            ref_cbcb_dist[key] = d_cbcb
            ref_caca_dist[key] = d_caca
            print(f"  Pair {key}: Cβ-Cβ={d_cbcb:.4f} nm, CA-CA={d_caca:.4f} nm (from data)")

    return DisulfideGeometryLoss(
        disulfide_pairs_local=local_pairs,
        N_local=N_local,
        global_scale=global_scale,
        complex_com=complex_com,
        ref_cbcb_dist=ref_cbcb_dist,
        ref_caca_dist=ref_caca_dist,
        k_cbcb=k_cbcb,
        k_caca=k_caca,
    )

def backbone_fape_loss(pred_R, pred_t_std, true_R, true_t_std,
                       global_scale, clamp_dist=1.0, eps=1e-4, reduction="mean"):
    """SE(3)-invariant Frame Aligned Point Error (FAPE).

    For each reference frame i, transforms all CA displacement vectors
    (j - i) into the local coordinate system of frame i, then compares
    predicted vs true local vectors.  The clamp prevents gradient explosion
    from large errors early in training.

    Args:
        pred_R:      (B, N, 3, 3) predicted clean rotations
        pred_t_std:  (B, N, 3)   predicted clean translations (normalized)
        true_R:      (B, N, 3, 3) ground truth rotations
        true_t_std:  (B, N, 3)   ground truth translations (normalized)
        global_scale: scalar (nm) — converts to physical units before FAPE
        clamp_dist:  max distance contribution per pair (nm), default 1.0
        eps:         numerical stability for sqrt
    Returns:
        scalar loss (nm)
    """
    gs = float(global_scale)

    pred_t = pred_t_std.float() * gs   # (B, N, 3) in nm
    true_t = true_t_std.float() * gs
    pred_R = pred_R.float()
    true_R = true_R.float()

    # Pairwise displacement vectors: (B, N, N, 3)
    pred_diff = pred_t.unsqueeze(2) - pred_t.unsqueeze(1)
    true_diff = true_t.unsqueeze(2) - true_t.unsqueeze(1)

    # Rotate into each frame's local coordinate system: R_i^T @ (t_j - t_i)
    pred_R_inv = pred_R.transpose(-1, -2)   # (B, N, 3, 3)
    true_R_inv = true_R.transpose(-1, -2)

    # einsum: for each frame i, rotate all N displacement vectors
    pred_local = torch.einsum('bnij,bnmj->bnmi', pred_R_inv, pred_diff)  # (B, N, N, 3)
    true_local = torch.einsum('bnij,bnmj->bnmi', true_R_inv, true_diff)

    dist = torch.sqrt(((pred_local - true_local) ** 2).sum(dim=-1) + eps)  # (B, N, N)
    loss = _reduce_batch(torch.clamp(dist, max=clamp_dist), reduction)
    return loss


# ─── 5. Interface Distance Loss ───────────────────────────────────────────────

def interface_distance_loss(pred_protein_t_std, pred_peptide_t_std,
                            true_protein_t_std, true_peptide_t_std,
                            global_scale, distance_threshold_nm=0.8, reduction="mean"):
    """Cross-chain interface CA-CA distance loss.

    Supervises the relative positioning between protein and peptide by comparing
    pairwise CA-CA distances across chains. Only residue pairs within the distance
    threshold in the reference structure are supervised (interface contacts).

    Why masked mean:
        Using F.mse_loss(pred * mask, ref * mask) would average over all pairs
        including non-interface zeros, diluting the true interface loss. Masked
        mean computes loss only over interface pairs.

    Why threshold = 0.8 nm (8 Å):
        1.0 nm would be too restrictive (misses many interface contacts).
        0.8 nm captures typical protein-peptide interface interactions.

    Args:
        pred_protein_t_std: (B, N_prot, 3) predicted protein CA (normalized)
        pred_peptide_t_std: (B, N_pep, 3) predicted peptide CA (normalized)
        true_protein_t_std: (B, N_prot, 3) ground truth protein CA (normalized)
        true_peptide_t_std: (B, N_pep, 3) ground truth peptide CA (normalized)
        global_scale: scalar (nm) — converts to physical units
        distance_threshold_nm: interface contact threshold (default 0.8 nm = 8 Å)

    Returns:
        scalar loss (nm^2)

    Note:
        This loss primarily constrains cross-chain relative position, not full
        interface orientation. For complete interface geometry, consider adding
        contact loss or cross-chain distogram loss in future versions.
    """
    gs = float(global_scale)

    # Denormalize to physical space
    pred_prot = pred_protein_t_std.float() * gs  # (B, N_prot, 3)
    pred_pep = pred_peptide_t_std.float() * gs   # (B, N_pep, 3)
    true_prot = true_protein_t_std.float() * gs
    true_pep = true_peptide_t_std.float() * gs

    # Cross-chain pairwise distances (B, N_prot, N_pep)
    pred_cross_dist = torch.cdist(pred_prot, pred_pep)
    true_cross_dist = torch.cdist(true_prot, true_pep)

    # Interface mask: only supervise pairs within threshold in reference
    interface_mask = (true_cross_dist < distance_threshold_nm).float()  # (B, N_prot, N_pep)
    diff_sq = (pred_cross_dist - true_cross_dist) ** 2
    if reduction == "none":
        count = interface_mask.sum(dim=(1, 2))
        return (diff_sq * interface_mask).sum(dim=(1, 2)) / count.clamp_min(1)
    if reduction != "mean":
        raise ValueError("reduction must be 'none' or 'mean'")
    # Preserve the historical pair-weighted scalar API for inference callers.
    return (diff_sq * interface_mask).sum() / interface_mask.sum().clamp_min(1)


def cross_chain_distogram_loss(pred_protein_t_std, pred_peptide_t_std,
                                true_protein_t_std, true_peptide_t_std,
                                global_scale, reduction="mean"):
    """Cross-chain global distogram loss (all protein-peptide pairs).

    Complements interface_distance_loss by supervising ALL cross-chain distances,
    not just interface contacts. Provides global docking pose constraint.

    Args:
        pred_protein_t_std: (B, N_prot, 3) predicted protein CA (normalized)
        pred_peptide_t_std: (B, N_pep, 3) predicted peptide CA (normalized)
        true_protein_t_std: (B, N_prot, 3) ground truth protein CA (normalized)
        true_peptide_t_std: (B, N_pep, 3) ground truth peptide CA (normalized)
        global_scale: scalar (nm)

    Returns:
        scalar loss (nm^2)
    """
    gs = float(global_scale)
    pred_prot = pred_protein_t_std.float() * gs
    pred_pep = pred_peptide_t_std.float() * gs
    true_prot = true_protein_t_std.float() * gs
    true_pep = true_peptide_t_std.float() * gs

    # (B, N_prot, N_pep) cross-chain distance matrices
    pred_cross = torch.cdist(pred_prot, pred_pep)
    true_cross = torch.cdist(true_prot, true_pep)

    # MSE over all cross-chain pairs
    return _reduce_batch((pred_cross - true_cross) ** 2, reduction)


# ─── 6. Torsion Loss (φ/ψ/ω) ─────────────────────────────────────────────────

def _compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle for four consecutive atom positions.

    Args:
        p0, p1, p2, p3: (..., 3) atom positions
    Returns:
        angle: (...,) dihedral in radians [-π, π]
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = F.normalize(b1, dim=-1)

    # Project b0 and b2 onto plane perpendicular to b1
    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _periodic_angle_loss(pred_angle, true_angle, reduction="mean"):
    """Periodic MSE for angles: min over ±2π wrapping.

    Args:
        pred_angle, true_angle: (...,) in radians
    Returns:
        scalar MSE
    """
    diff = pred_angle - true_angle
    # Wrap to [-π, π]
    diff = diff - 2.0 * torch.pi * torch.round(diff / (2.0 * torch.pi))
    return _reduce_batch(diff ** 2, reduction)


def torsion_loss(pred_R, pred_t_std, true_R, true_t_std,
                 N_local, global_scale, complex_com, reduction="mean"):
    """Backbone torsion loss: φ, ψ, ω angles.

    Reconstructs N, CA, C positions from frames, then computes backbone
    dihedral angles and supervises them with periodic MSE.

    φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
    ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1})
    ω_i = dihedral(CA_i, C_i, N_{i+1}, CA_{i+1})

    Args:
        pred_R:      (B, N, 3, 3) predicted clean rotations
        pred_t_std:  (B, N, 3)   predicted clean translations (normalized)
        true_R:      (B, N, 3, 3) ground truth rotations
        true_t_std:  (B, N, 3)   ground truth translations (normalized)
        N_local:     (N, 3) mean N position in local frame (nm)
        global_scale: scalar normalization factor
        complex_com: (3,) complex COM
    Returns:
        scalar loss (radians^2)
    """
    from frame_utils import BOND_CA_C  # type: ignore
    device = pred_R.device
    dtype = torch.float32

    pred_R  = pred_R.float()
    true_R  = true_R.float()
    gs = float(global_scale)
    com = torch.as_tensor(complex_com, device=device, dtype=dtype).reshape(1, 1, 3)
    N_loc = torch.as_tensor(N_local, device=device, dtype=dtype)  # (N, 3)

    N_res = pred_R.shape[1]
    C_loc = torch.zeros(N_res, 3, device=device, dtype=dtype)
    C_loc[:, 0] = BOND_CA_C

    def reconstruct(R, t_std):
        t_phys = t_std.float() * gs + com
        CA = t_phys
        C  = torch.einsum('bnij,nj->bni', R, C_loc) + t_phys
        N  = torch.einsum('bnij,nj->bni', R, N_loc) + t_phys
        return N, CA, C

    pred_N, pred_CA, pred_C = reconstruct(pred_R, pred_t_std)
    true_N, true_CA, true_C = reconstruct(true_R, true_t_std)

    if N_res < 2:
        return _reduce_batch(pred_t_std.sum(dim=(1, 2))[:, None] * 0.0, reduction)

    loss = torch.tensor(0.0, device=device)
    n_terms = 0

    # ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1})  — residues 0..N-2
    pred_psi = _compute_dihedral(
        pred_N[:, :-1], pred_CA[:, :-1], pred_C[:, :-1], pred_N[:, 1:])
    true_psi = _compute_dihedral(
        true_N[:, :-1], true_CA[:, :-1], true_C[:, :-1], true_N[:, 1:])
    loss = loss + _periodic_angle_loss(pred_psi, true_psi, reduction)
    n_terms += 1

    # ω_i = dihedral(CA_i, C_i, N_{i+1}, CA_{i+1})  — residues 0..N-2
    pred_omega = _compute_dihedral(
        pred_CA[:, :-1], pred_C[:, :-1], pred_N[:, 1:], pred_CA[:, 1:])
    true_omega = _compute_dihedral(
        true_CA[:, :-1], true_C[:, :-1], true_N[:, 1:], true_CA[:, 1:])
    loss = loss + _periodic_angle_loss(pred_omega, true_omega, reduction)
    n_terms += 1

    # φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i)  — residues 1..N-1
    if N_res >= 3:
        pred_phi = _compute_dihedral(
            pred_C[:, :-1], pred_N[:, 1:], pred_CA[:, 1:], pred_C[:, 1:])
        true_phi = _compute_dihedral(
            true_C[:, :-1], true_N[:, 1:], true_CA[:, 1:], true_C[:, 1:])
        loss = loss + _periodic_angle_loss(pred_phi, true_phi, reduction)
        n_terms += 1

    return loss / n_terms


# ─── 7. Pair Frame Loss (residue relative orientation) ───────────────────────

def pair_frame_loss(pred_R, pred_t_std, true_R, true_t_std,
                    global_scale, max_seq_sep=None, top_k_contacts=None, reduction="mean"):
    """Residue-pair relative frame loss.

    For each pair (i, j), computes the relative rotation and translation
    between frames i and j, then supervises both with MSE.

    Relative rotation:  R_rel = R_i^T @ R_j  → supervised via Frobenius norm
    Relative translation: t_rel = R_i^T @ (t_j - t_i)  → supervised via L2

    This teaches the model the correct relative spatial orientation between
    residues, which is critical for β-sheet registry and strand pairing.

    Args:
        pred_R:         (B, N, 3, 3) predicted clean rotations
        pred_t_std:     (B, N, 3)   predicted clean translations (normalized)
        true_R:         (B, N, 3, 3) ground truth rotations
        true_t_std:     (B, N, 3)   ground truth translations (normalized)
        global_scale:   scalar normalization factor
        max_seq_sep:    if set, only supervise pairs |i-j| <= max_seq_sep
                        (None = all pairs, but expensive for large N)
        top_k_contacts: if set, only supervise the top-k closest pairs
                        in the true structure (contact-focused supervision)
    Returns:
        scalar loss
    """
    device = pred_R.device
    gs = float(global_scale)

    pred_R  = pred_R.float()
    true_R  = true_R.float()
    pred_t  = pred_t_std.float() * gs   # (B, N, 3) in nm
    true_t  = true_t_std.float() * gs

    B, N = pred_R.shape[:2]

    # Build pair index mask
    i_idx = torch.arange(N, device=device)
    j_idx = torch.arange(N, device=device)
    ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')  # (N, N)
    pair_mask = ii < jj  # upper triangle only

    if max_seq_sep is not None:
        pair_mask = pair_mask & ((jj - ii) <= max_seq_sep)

    if top_k_contacts is not None and top_k_contacts < pair_mask.sum():
        # Select top-k closest pairs in true structure (averaged over batch)
        true_dist = torch.cdist(true_t[0:1], true_t[0:1]).squeeze(0)  # (N, N)
        true_dist_masked = true_dist.clone()
        true_dist_masked[~pair_mask] = float('inf')
        flat_idx = true_dist_masked.reshape(-1).topk(top_k_contacts, largest=False).indices
        top_mask = torch.zeros(N * N, device=device, dtype=torch.bool)
        top_mask[flat_idx] = True
        pair_mask = pair_mask & top_mask.reshape(N, N)

    i_sel, j_sel = pair_mask.nonzero(as_tuple=True)  # (P,)
    if len(i_sel) == 0:
        return _reduce_batch(pred_t_std.sum(dim=(1, 2))[:, None] * 0.0, reduction)

    # Relative rotation: R_rel = R_i^T @ R_j  →  (B, P, 3, 3)
    Ri = pred_R[:, i_sel]   # (B, P, 3, 3)
    Rj = pred_R[:, j_sel]
    pred_R_rel = Ri.transpose(-1, -2) @ Rj

    Ri_true = true_R[:, i_sel]
    Rj_true = true_R[:, j_sel]
    true_R_rel = Ri_true.transpose(-1, -2) @ Rj_true

    # Frobenius norm of rotation difference
    rot_loss = _reduce_batch(((pred_R_rel - true_R_rel) ** 2).sum(dim=(-1, -2)), reduction)

    # Relative translation: t_rel = R_i^T @ (t_j - t_i)  →  (B, P, 3)
    pred_dt = pred_t[:, j_sel] - pred_t[:, i_sel]   # (B, P, 3)
    true_dt = true_t[:, j_sel] - true_t[:, i_sel]

    pred_t_rel = torch.einsum('bpij,bpj->bpi', Ri.transpose(-1, -2), pred_dt)
    true_t_rel = torch.einsum('bpij,bpj->bpi', Ri_true.transpose(-1, -2), true_dt)

    trans_loss = _reduce_batch((pred_t_rel - true_t_rel) ** 2, reduction)

    # Normalize rotation loss to comparable scale with trans loss
    # Frobenius norm of a rotation matrix is sqrt(3) ≈ 1.73, so max diff^2 ≈ 12
    # Divide by 9 (number of matrix elements) to normalize
    return rot_loss / 9.0 + trans_loss
