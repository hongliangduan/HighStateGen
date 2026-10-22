"""
Dual-molecule frame dataset: protein + peptide with interface awareness.

Loads train_set.npz, identifies protein/peptide residues, builds separate frames.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Standard 20 amino acids + unknown
AA_VOCAB = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}
AA_UNK = 20  # index for non-standard / unknown residues


def residue_names_to_indices(names):
    """Convert list of 3-letter residue names to integer indices (LongTensor)."""
    return torch.tensor([AA_VOCAB.get(n.upper(), AA_UNK) for n in names], dtype=torch.long)


class ProteinPeptideFrameDataset(Dataset):
    """Dataset for protein-peptide complex with separate frame representations.

    Each sample returns:
        protein_R: (N_protein, 3, 3) protein rotation matrices
        protein_t: (N_protein, 3) protein CA translations (standardized)
        peptide_R: (N_peptide, 3, 3) peptide rotation matrices
        peptide_t: (N_peptide, 3) peptide CA translations (standardized)
        slow_feat: (optional) normalized TICA/PCA features
    """

    def __init__(self, npz_path, topology_path,
                 protein_residue_range=(0, 293),
                 peptide_residue_range=(295, 307),
                 skip_residues=None,
                 precomputed=None,
                 slow_feat_path=None,
                 basin_label_path=None,
                 translation_scale_mode="coordinate_rms"):
        """
        Args:
            npz_path: path to train_set.npz
            topology_path: complex_dry.prmtop
            protein_residue_range: (start, end) inclusive residue indices for protein
            peptide_residue_range: (start, end) inclusive residue indices for peptide
            skip_residues: residue indices to skip (e.g. ACE caps)
            precomputed: dict with precomputed metadata from train set
            slow_feat_path: path to normalized TICA/PCA features (N_samples, n_dims)
        """
        from frame_utils import (coords_to_frames, build_residue_bb_indices,
                                 compute_reference_locals)
        import mdtraj as md

        if skip_residues is None:
            skip_residues = []

        self.protein_range = protein_residue_range
        self.peptide_range = peptide_residue_range
        if translation_scale_mode not in ("coordinate_rms", "radial_rms"):
            raise ValueError("Unknown translation_scale_mode")
        self.translation_scale_mode = translation_scale_mode

        data = np.load(npz_path, allow_pickle=True)
        coords_std = data["coords"]  # (N_samples, N_atoms_full, 3) standardized
        mean_full = np.asarray(data["mean"], dtype=np.float32)
        std_full = np.asarray(data["std"], dtype=np.float32)

        # Load topology
        if topology_path.endswith('.prmtop'):
            base = topology_path.replace('.prmtop', '')
            for ext in ['.pdb', '.inpcrd', '.rst7']:
                if os.path.exists(base + ext):
                    traj = md.load(base + ext, top=topology_path)
                    break
            else:
                raise FileNotFoundError(
                    f"No companion file (.pdb/.inpcrd/.rst7) found for {topology_path}")
        else:
            traj = md.load(topology_path)
        topo = traj.topology

        # Extract backbone_enhanced subset (same as before)
        import sys
        sys.path.insert(0, os.path.dirname(__file__))  # repository source directory
        from physics_constraints import extract_enhanced_subset_indices

        subset_info = extract_enhanced_subset_indices(
            topo, include_1hop_neighbors=True, ref_coords_nm=traj.xyz[0])
        sel_idx = subset_info["selected_atom_indices"]
        self.subset_info = subset_info

        # Slice coords to subset
        coords_std = coords_std[:, sel_idx, :]
        n_atoms_sub = len(sel_idx)

        # Slice mean/std
        if mean_full.ndim == 1 and mean_full.size == 3:
            mean_sub = mean_full
            std_sub = std_full
        else:
            mean_sub = mean_full[sel_idx]
            std_sub = std_full[sel_idx]

        # Denormalize to physical nm coordinates
        coords_nm = coords_std * std_sub + mean_sub  # (N_samples, M, 3)

        # ── Build residue → backbone atom mapping (ALL residues) ──
        residue_bb_indices, valid_residue_ids = build_residue_bb_indices(
            topo, selected_atom_indices=sel_idx, skip_residues=skip_residues)

        # ── Separate protein and peptide residues ──
        protein_mask = np.array([
            (self.protein_range[0] <= rid <= self.protein_range[1])
            for rid in valid_residue_ids
        ])
        peptide_mask = np.array([
            (self.peptide_range[0] <= rid <= self.peptide_range[1])
            for rid in valid_residue_ids
        ])

        self.protein_residue_ids = valid_residue_ids[protein_mask]
        self.peptide_residue_ids = valid_residue_ids[peptide_mask]
        self.protein_residue_names = [topo.residue(int(rid)).name for rid in self.protein_residue_ids]
        self.peptide_residue_names = [topo.residue(int(rid)).name for rid in self.peptide_residue_ids]
        self.n_protein = len(self.protein_residue_ids)
        self.n_peptide = len(self.peptide_residue_ids)

        # Convert residue names to AA type indices
        self.protein_aa = residue_names_to_indices(self.protein_residue_names)
        self.peptide_aa = residue_names_to_indices(self.peptide_residue_names)

        protein_bb_indices = residue_bb_indices[protein_mask]
        peptide_bb_indices = residue_bb_indices[peptide_mask]

        # Save for reconstruction
        self.protein_bb_indices = protein_bb_indices
        self.peptide_bb_indices = peptide_bb_indices
        self.coords_nm = coords_nm  # (N_samples, M_subset, 3) — used by build_disulfide_loss

        print(f"[ProteinPeptideDataset] {len(coords_nm)} samples, {n_atoms_sub} subset atoms")
        print(f"  Protein: {self.n_protein} residues (IDs {self.protein_residue_ids[0]}-{self.protein_residue_ids[-1]})")
        print(f"  Peptide: {self.n_peptide} residues (IDs {self.peptide_residue_ids[0]}-{self.peptide_residue_ids[-1]})")

        # ── Convert to frames (separate for protein and peptide) ──
        bb_idx_protein = torch.from_numpy(protein_bb_indices)
        bb_idx_peptide = torch.from_numpy(peptide_bb_indices)
        coords_nm_t = torch.from_numpy(coords_nm).float()

        protein_R, protein_t, protein_O = coords_to_frames(coords_nm_t, bb_idx_protein)
        peptide_R, peptide_t, peptide_O = coords_to_frames(coords_nm_t, bb_idx_peptide)

        # ── Compute reference locals ──
        if precomputed is not None:
            self.protein_N_local = precomputed["protein_N_local"]
            self.protein_O_local = precomputed["protein_O_local"]
            self.peptide_N_local = precomputed["peptide_N_local"]
            self.peptide_O_local = precomputed["peptide_O_local"]
            self.global_scale = precomputed["global_scale"]
            self.complex_com = precomputed["complex_com"]
        else:
            protein_N_local, protein_O_local = compute_reference_locals(
                coords_nm, protein_bb_indices)
            self.protein_N_local = protein_N_local
            self.protein_O_local = protein_O_local

            peptide_N_local, peptide_O_local = compute_reference_locals(
                coords_nm, peptide_bb_indices)
            self.peptide_N_local = peptide_N_local
            self.peptide_O_local = peptide_O_local

            # ── Global scalar normalization (preserves SE(3) geometry) ──
            # 1. Compute complex COM per frame
            complex_t = torch.cat([protein_t, peptide_t], dim=1)  # (N, N_total, 3)
            com = complex_t.mean(dim=1, keepdim=True)              # (N, 1, 3)

            # 2. Center both molecules around complex COM
            protein_t_centered = protein_t - com
            peptide_t_centered = peptide_t - com

            # 3. Global scalar scale: RMS distance from COM across all residues/frames
            all_centered = torch.cat([protein_t_centered, peptide_t_centered], dim=1)
            squared = all_centered ** 2
            if translation_scale_mode == "radial_rms":
                self.global_scale = float(torch.sqrt(squared.sum(dim=-1).mean()))
            else:
                # Legacy checkpoints used an RMS per Cartesian component.
                self.global_scale = float(torch.sqrt(squared.mean()))
            self.complex_com = com.squeeze(1).mean(dim=0)  # (3,) mean COM for val reuse

        if not np.isfinite(self.global_scale) or self.global_scale <= 0:
            raise ValueError("Training-frame coordinate scale must be finite and positive")

        # Apply normalization
        complex_t = torch.cat([protein_t, peptide_t], dim=1)
        com = complex_t.mean(dim=1, keepdim=True)
        protein_t_norm = (protein_t - com) / self.global_scale
        peptide_t_norm = (peptide_t - com) / self.global_scale

        # Store
        self.protein_R = protein_R.numpy()
        self.protein_t = protein_t_norm.numpy()
        self.peptide_R = peptide_R.numpy()
        self.peptide_t = peptide_t_norm.numpy()

        # Load slow features (TICA/PCA) if provided
        if slow_feat_path:
            self.slow_feat = torch.from_numpy(np.load(slow_feat_path)).float()
            assert len(self.slow_feat) == len(self.protein_R), \
                f"Slow features length {len(self.slow_feat)} != dataset length {len(self.protein_R)}"
        else:
            self.slow_feat = None

        # Optional per-frame basin labels (int indices)
        if basin_label_path:
            basin_labels = np.load(basin_label_path)
            if len(basin_labels) != len(self.protein_R):
                raise ValueError(
                    f"Basin labels length {len(basin_labels)} != dataset length {len(self.protein_R)}"
                )
            self.basin_labels = basin_labels.astype(np.int64, copy=False)
        else:
            self.basin_labels = None

        print(f"  Global scale: {self.global_scale:.4f} nm")
        print(f"  Protein CA translation range (norm): [{self.protein_t.min():.2f}, {self.protein_t.max():.2f}]")
        print(f"  Peptide CA translation range (norm): [{self.peptide_t.min():.2f}, {self.peptide_t.max():.2f}]")

    def __len__(self):
        return len(self.protein_R)

    def __getitem__(self, idx):
        item = {
            "protein_R": torch.from_numpy(self.protein_R[idx]),  # (N_protein, 3, 3)
            "protein_t": torch.from_numpy(self.protein_t[idx]),  # (N_protein, 3)
            "peptide_R": torch.from_numpy(self.peptide_R[idx]),  # (N_peptide, 3, 3)
            "peptide_t": torch.from_numpy(self.peptide_t[idx]),  # (N_peptide, 3)
            "protein_aa": self.protein_aa,  # (N_protein,) LongTensor
            "peptide_aa": self.peptide_aa,  # (N_peptide,) LongTensor
        }
        if self.slow_feat is not None:
            item["slow_feat"] = self.slow_feat[idx]
        if self.basin_labels is not None:
            item["basin"] = int(self.basin_labels[idx])
        return item

    def get_precomputed(self):
        """Return precomputed stats for val set to reuse."""
        return {
            "protein_N_local": self.protein_N_local,
            "protein_O_local": self.protein_O_local,
            "peptide_N_local": self.peptide_N_local,
            "peptide_O_local": self.peptide_O_local,
            "global_scale": self.global_scale,
            "complex_com": self.complex_com,
            "translation_scale_mode": self.translation_scale_mode,
        }
