"""
Protein-Peptide IPA model with interface cross-attention.

Architecture (V10):
1. Separate IPA stacks for protein and peptide
2. Cross-attention layers for interface coupling
3. Separate velocity outputs for protein and peptide
4. K-Hop topo bias injected into first n_topo_heads scalar heads only (head specialization)
   - active_bias = topo_bias * sigmoid(log_scale) * (1 - t)  [time-gated annealing]
   - Only scalar attention receives topo bias; point attention stays pure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from frame_utils import so3_exp_map


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous time."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * 1000.0 * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConditionEncoder(nn.Module):
    """Encode slow features with a learnable null embedding for CFG."""
    def __init__(self, slow_feat_dim, condition_dim):
        super().__init__()
        self.slow_feat_dim = slow_feat_dim
        hidden_dim = max(condition_dim // 2, slow_feat_dim) if slow_feat_dim > 0 else condition_dim
        if slow_feat_dim > 0:
            self.slow_feat_mlp = nn.Sequential(
                nn.Linear(slow_feat_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, condition_dim))
        self.null_emb = nn.Parameter(torch.zeros(condition_dim))

    def forward(self, slow_feat=None, cond_mask=None, batch_size=None):
        """
        Args:
            slow_feat: (B, slow_feat_dim) normalized TICA/PCA features
            cond_mask: (B,) bool, True = drop condition and use null embedding
            batch_size: batch size for unconditional sampling when slow_feat is None
        """
        if slow_feat is not None:
            cond = self.slow_feat_mlp(slow_feat)
        else:
            if batch_size is None:
                raise ValueError("batch_size is required when slow_feat is None")
            cond = self.null_emb.unsqueeze(0).expand(batch_size, -1)

        if cond_mask is not None and cond_mask.any():
            cond = cond.clone()
            cond[cond_mask] = self.null_emb.to(cond.dtype)

        return cond


class InvariantPointAttention(nn.Module):
    """IPA from AlphaFold2 (simplified)."""

    def __init__(self, c_s, c_z, c_hidden=16, n_heads=8,
                 n_qk_points=4, n_v_points=8, n_topo_heads=2,
                 use_gaussian_bias=False, max_seq_len=None):
        super().__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.n_heads = n_heads
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points
        self.n_topo_heads = n_topo_heads  # first n heads receive K-Hop topo bias
        self.use_gaussian_bias = use_gaussian_bias

        # Scalar projections
        self.q_scalar = nn.Linear(c_s, n_heads * c_hidden, bias=False)
        self.k_scalar = nn.Linear(c_s, n_heads * c_hidden, bias=False)
        self.v_scalar = nn.Linear(c_s, n_heads * c_hidden, bias=False)

        # Point projections
        self.q_points = nn.Linear(c_s, n_heads * n_qk_points * 3, bias=False)
        self.k_points = nn.Linear(c_s, n_heads * n_qk_points * 3, bias=False)
        self.v_points = nn.Linear(c_s, n_heads * n_v_points * 3, bias=False)

        # Pair bias
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)

        # Learnable point attention weights (AlphaFold2 gamma_h)
        self.point_weights = nn.Parameter(torch.ones(n_heads))

        # Mac-Diff style Gaussian sequence bias (optional)
        if use_gaussian_bias:
            if max_seq_len is None:
                raise ValueError("max_seq_len required when use_gaussian_bias=True")
            self.gaussian_weight = nn.Parameter(torch.tensor(1.0))
            self.gaussian_sigma = nn.Parameter(torch.tensor(2.0))
            # Precompute distance matrix
            idx = torch.arange(max_seq_len)
            dist_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().float()
            self.register_buffer('seq_dist_matrix', dist_matrix, persistent=False)

        # Output projection
        out_dim = n_heads * (c_hidden + n_v_points * 4) + c_z
        self.proj_out = nn.Linear(out_dim, c_s)

    def forward(self, s, z, R, t, time_step=None, topo_bias=None):
        """
        Args:
            s: (B, N, c_s) single repr
            z: (B, N, N, c_z) pair repr
            R: (B, N, 3, 3) frames
            t: (B, N, 3) translations
            time_step: (B,) flow time scalar in [0,1], used for (1-t) gating
            topo_bias: (B, N, N) K-Hop mask (0 for in-hop, -1e9 for out-of-hop)
        """
        B, N, _ = s.shape
        h = self.n_heads

        # Scalar attention
        q = self.q_scalar(s).view(B, N, h, self.c_hidden)
        k = self.k_scalar(s).view(B, N, h, self.c_hidden)
        v = self.v_scalar(s).view(B, N, h, self.c_hidden)

        attn_scalar = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(self.c_hidden)

        # Point attention
        q_pts = self.q_points(s).view(B, N, h, self.n_qk_points, 3)
        k_pts = self.k_points(s).view(B, N, h, self.n_qk_points, 3)
        v_pts = self.v_points(s).view(B, N, h, self.n_v_points, 3)

        # Transform points to global frame
        q_pts_global = torch.einsum('bnij,bnhpj->bnhpi', R.float(), q_pts.float()) + t.unsqueeze(2).unsqueeze(2).float()
        k_pts_global = torch.einsum('bnij,bnhpj->bnhpi', R.float(), k_pts.float()) + t.unsqueeze(2).unsqueeze(2).float()
        v_pts_global = torch.einsum('bnij,bnhpj->bnhpi', R.float(), v_pts.float()) + t.unsqueeze(2).unsqueeze(2).float()

        # Point distances with learnable per-head weight (AF2 gamma_h)
        diff = q_pts_global.unsqueeze(2) - k_pts_global.unsqueeze(1)  # (B, N, N, h, p, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, N, h, p)
        gamma = F.softplus(self.point_weights).view(1, 1, 1, self.n_heads, 1)
        attn_points = -0.5 * (gamma * dist_sq).sum(dim=-1)  # (B, N, N, h)
        attn_points = attn_points.permute(0, 3, 1, 2)  # (B, h, N, N)

        # Pair bias
        pair_bias = self.pair_bias(z).permute(0, 3, 1, 2)  # (B, h, N, N)

        # Combined attention
        attn = attn_scalar + attn_points + pair_bias

        # 1. Clamp first to protect normal logits from overflow
        attn = attn.clamp(-30.0, 30.0)

        # 2. Gaussian sequence bias (Mac-Diff style, soft constraint)
        if self.use_gaussian_bias:
            N = attn.shape[-1]
            dist = self.seq_dist_matrix[:N, :N].to(attn.device)
            sigma_sq = F.softplus(self.gaussian_sigma) ** 2 + 1e-6
            gaussian_bias = self.gaussian_weight * torch.exp(-dist ** 2 / (2 * sigma_sq))
            attn = attn + gaussian_bias.unsqueeze(0).unsqueeze(0)

        # 3. K-Hop topo bias injection AFTER clamp (head specialization, hard constraint)
        if topo_bias is not None and self.n_topo_heads > 0:
            active_bias = torch.zeros_like(attn)
            active_bias[:, :self.n_topo_heads, :, :] = \
                topo_bias.unsqueeze(1).to(attn.dtype)
            attn = attn + active_bias

        # 3. Softmax: -1e9 positions output mathematically clean 0.0
        attn = torch.softmax(attn.float(), dim=-1).to(s.dtype)

        # Aggregate scalar
        out_scalar = torch.einsum('bhij,bjhd->bihd', attn.float(), v.float())
        out_scalar = out_scalar.to(s.dtype).reshape(B, N, -1)

        # Aggregate points - keep in float32 until after subtraction to avoid catastrophic cancellation
        out_pts_global = torch.einsum('bhij,bjhpd->bihpd', attn.float(), v_pts_global.float())
        # DO NOT downcast here - global coords can be large, subtraction must stay in float32

        # Transform back to local frame (all in float32)
        out_pts_local = torch.einsum('bnij,bnhpj->bnhpi',
                                     R.transpose(-1, -2).float(),
                                     (out_pts_global - t.unsqueeze(2).unsqueeze(2).float()))
        out_pts_local = out_pts_local.to(s.dtype)  # safe to downcast now (small local coords)

        out_pts_flat = out_pts_local.reshape(B, N, -1)
        out_pts_norm = torch.sqrt((out_pts_local ** 2).sum(dim=-1) + 1e-8).reshape(B, N, -1)

        # Pair aggregation
        pair_out = torch.einsum('bhij,bijd->bid', attn.float(), z.float())
        pair_out = pair_out.to(s.dtype)

        # Concatenate and project
        out = torch.cat([out_scalar, out_pts_flat, out_pts_norm, pair_out], dim=-1)
        return self.proj_out(out)


class CrossInterfaceAttention(nn.Module):
    """Cross-attention between protein and peptide with 3D distance awareness."""

    def __init__(self, c_s, n_heads=8, dropout=0.1, n_qk_points=4):
        super().__init__()
        self.c_s = c_s
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.n_qk_points = n_qk_points

        # Scalar projections
        self.q_proj = nn.Linear(c_s, c_s)
        self.k_proj = nn.Linear(c_s, c_s)
        self.v_proj = nn.Linear(c_s, c_s)
        self.out_proj = nn.Linear(c_s, c_s)

        # 3D point projections for distance-aware attention
        self.q_points = nn.Linear(c_s, n_heads * n_qk_points * 3, bias=False)
        self.k_points = nn.Linear(c_s, n_heads * n_qk_points * 3, bias=False)
        self.point_weights = nn.Parameter(torch.ones(n_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_s, query_R, query_t, kv_s, kv_R, kv_t):
        """
        Args:
            query_s: (B, N_q, c_s)
            query_R: (B, N_q, 3, 3)
            query_t: (B, N_q, 3)
            kv_s:    (B, N_kv, c_s)
            kv_R:    (B, N_kv, 3, 3)
            kv_t:    (B, N_kv, 3)
        Returns:
            (B, N_q, c_s)
        """
        B, N_q, _ = query_s.shape
        N_kv = kv_s.shape[1]
        h = self.n_heads

        # Scalar attention
        q = self.q_proj(query_s).view(B, N_q, h, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_s).view(B, N_kv, h, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_s).view(B, N_kv, h, self.head_dim).transpose(1, 2)
        attn_scalar = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3D point attention: query points in global frame vs kv points in global frame
        q_pts = self.q_points(query_s).view(B, N_q, h, self.n_qk_points, 3)
        k_pts = self.k_points(kv_s).view(B, N_kv, h, self.n_qk_points, 3)

        q_pts_global = torch.einsum('bnij,bnhpj->bnhpi', query_R.float(), q_pts.float()) \
                       + query_t.unsqueeze(2).unsqueeze(2).float()
        k_pts_global = torch.einsum('bnij,bnhpj->bnhpi', kv_R.float(), k_pts.float()) \
                       + kv_t.unsqueeze(2).unsqueeze(2).float()

        # Cross-distance: (B, N_q, N_kv, h, p)
        diff = q_pts_global.unsqueeze(2) - k_pts_global.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        gamma = F.softplus(self.point_weights).view(1, 1, 1, h, 1)
        attn_points = -0.5 * (gamma * dist_sq).sum(dim=-1)  # (B, N_q, N_kv, h)
        attn_points = attn_points.permute(0, 3, 1, 2)        # (B, h, N_q, N_kv)

        attn = (attn_scalar + attn_points).clamp(-30.0, 30.0)
        attn = torch.softmax(attn.float(), dim=-1).to(query_s.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.c_s)
        return self.out_proj(out)


class IPABlock(nn.Module):
    """IPA block with FiLM conditioning."""

    def __init__(self, c_s, c_z, condition_dim, n_heads=8, dropout=0.1,
                 c_hidden=16, n_qk_points=4, n_v_points=8, n_topo_heads=2,
                 use_gaussian_bias=False, max_seq_len=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(c_s)
        self.ipa = InvariantPointAttention(
            c_s, c_z, c_hidden, n_heads, n_qk_points, n_v_points, n_topo_heads,
            use_gaussian_bias, max_seq_len)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(c_s)
        self.transition = nn.Sequential(
            nn.Linear(c_s, c_s * 4), nn.SiLU(),
            nn.Linear(c_s * 4, c_s))
        self.dropout2 = nn.Dropout(dropout)

        # FiLM modulation
        self.film = nn.Sequential(
            nn.Linear(condition_dim, c_s * 2), nn.SiLU(),
            nn.Linear(c_s * 2, c_s * 2))

    def forward(self, s, z, R, t, cond, time_step=None, topo_bias=None):
        """
        Args:
            s: (B, N, c_s), z: (B, N, N, c_z)
            R: (B, N, 3, 3), t: (B, N, 3)
            cond: (B, condition_dim)
            time_step: (B,) flow time scalar for topo gating
            topo_bias: (B, N, N) K-Hop mask
        """
        s = s + self.dropout1(self.ipa(self.ln1(s), z, R, t,
                                       time_step=time_step, topo_bias=topo_bias))

        # FiLM
        gamma_beta = self.film(cond).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        s = (1.0 + gamma) * s + beta

        s = s + self.dropout2(self.transition(self.ln2(s)))
        return s


class ProteinPeptideFrameDenoiser(nn.Module):
    """Dual-molecule frame denoiser with interface cross-attention.

    Input: protein frames + peptide frames + time + PC conditions
    Output: (v_trans, omega) for protein and peptide separately
    """

    def __init__(self, n_protein, n_peptide, c_s=256, c_z=64, n_layers=6,
                 condition_dim=128, n_heads=8, dropout=0.1,
                 max_rel_pos=32, slow_feat_dim=0,
                 n_topo_heads=2, khop=3,
                 topo_path=None, peptide_residue_ids=None,
                 cyclic_peptide=False, disable_tica_conditioning=False,
                 disable_topology_bias=False, disable_interface_attention=False,
                 equivariant_translations=True, topology_graph_scope="peptide"):
        super().__init__()
        self.n_protein = n_protein
        self.n_peptide = n_peptide
        self.c_s = c_s
        self.c_z = c_z
        self.n_layers = n_layers
        self.slow_feat_dim = slow_feat_dim
        self.n_topo_heads = n_topo_heads
        self.khop = khop
        self.cyclic_peptide = cyclic_peptide
        self.disable_tica_conditioning = disable_tica_conditioning
        self.disable_topology_bias = disable_topology_bias
        self.disable_interface_attention = disable_interface_attention
        self.equivariant_translations = equivariant_translations
        if topology_graph_scope not in ("peptide", "complex"):
            raise ValueError("topology_graph_scope must be peptide or complex")
        self.topology_graph_scope = topology_graph_scope

        # Precompute K-Hop topo bias + full distance matrix for peptide
        # Use real bond topology from prmtop when available, including disulfide bonds
        # Store both k-hop mask and full distance matrix for consistent pair encoding
        if disable_topology_bias:
            self.register_buffer('pep_topo_bias', torch.zeros(1, n_peptide, n_peptide), persistent=False)
            self.register_buffer('pep_dist_matrix', self._build_linear_dist_matrix(
                n_peptide, cyclic=False), persistent=False)
        elif topo_path is not None and peptide_residue_ids is not None:
            pep_bias, pep_dist = self._build_khop_from_topology(
                topo_path, peptide_residue_ids, khop, return_dist=True)
            self.register_buffer('pep_topo_bias', pep_bias, persistent=False)
            self.register_buffer('pep_dist_matrix', pep_dist, persistent=False)
            print(f"[Model] Peptide topo bias + distance matrix from real bond topology "
                  f"(khop={khop}, includes disulfide bonds)")
        else:
            pep_bias = self._build_linear_khop_bias(
                n_peptide, khop, cyclic=self.cyclic_peptide)
            pep_dist = self._build_linear_dist_matrix(n_peptide, cyclic=self.cyclic_peptide)
            self.register_buffer('pep_topo_bias', pep_bias, persistent=False)
            self.register_buffer('pep_dist_matrix', pep_dist, persistent=False)
            print(f"[Model] Peptide topo bias from linear/cyclic sequence "
                  f"(khop={khop}) — pass topo_path for real topology")

        # Time embedding
        self.time_emb = SinusoidalPosEmb(condition_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, c_s), nn.SiLU(), nn.Linear(c_s, c_s))

        # Condition embedding (TICA / slow features only)
        self.cond_enc = ConditionEncoder(0 if disable_tica_conditioning else slow_feat_dim, condition_dim)

        # Fuse time + condition
        self.fuse = nn.Sequential(
            nn.Linear(c_s + condition_dim, c_s), nn.SiLU(),
            nn.Linear(c_s, condition_dim))

        # Residue position embeddings
        self.protein_res_emb = nn.Embedding(n_protein, c_s)
        self.peptide_res_emb = nn.Embedding(n_peptide, c_s)

        # Amino acid type embeddings (20 standard AA + 1 unknown)
        self.aa_type_emb = nn.Embedding(21, c_s)

        # Single repr init
        self.protein_single_init = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.peptide_single_init = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, c_s))

        # Pair repr init (relative sequence position only - no noisy spatial distances)
        self.max_rel_pos = max_rel_pos
        # Protein uses signed relative position [-max_rel_pos, max_rel_pos] → 2*max_rel_pos+1 bins
        # Peptide uses unsigned shortest-path distance [0, max_rel_pos] → max_rel_pos+1 bins
        # (topology-aware: disulfide shortcut reduces distance between CYS pair)
        protein_pair_input_dim = 2 * max_rel_pos + 1
        peptide_pair_input_dim = max_rel_pos + 1

        self.protein_pair_init = nn.Sequential(
            nn.Linear(protein_pair_input_dim, c_z), nn.SiLU(), nn.Linear(c_z, c_z))
        self.peptide_pair_init = nn.Sequential(
            nn.Linear(peptide_pair_input_dim, c_z), nn.SiLU(), nn.Linear(c_z, c_z))

        # IPA blocks (separate for protein and peptide)
        # Peptide blocks get n_topo_heads + gaussian_bias; protein blocks use 0 (no topo injection)
        self.protein_ipa_blocks = nn.ModuleList([
            IPABlock(c_s, c_z, condition_dim, n_heads, dropout, n_topo_heads=0,
                     use_gaussian_bias=False, max_seq_len=None)
            for _ in range(n_layers)
        ])
        self.peptide_ipa_blocks = nn.ModuleList([
            IPABlock(c_s, c_z, condition_dim, n_heads, dropout,
                     n_topo_heads=0 if disable_topology_bias else n_topo_heads,
                     use_gaussian_bias=True, max_seq_len=n_peptide)
            for _ in range(n_layers)
        ])

        # Cross-attention layers (every 2 layers)
        self.cross_attn_protein = nn.ModuleList([
            CrossInterfaceAttention(c_s, n_heads, dropout)
            for _ in range(0 if disable_interface_attention else n_layers // 2)
        ])
        self.cross_attn_peptide = nn.ModuleList([
            CrossInterfaceAttention(c_s, n_heads, dropout)
            for _ in range(0 if disable_interface_attention else n_layers // 2)
        ])

        # Per-layer frame update heads (FrameDiff-style iterative refinement)
        # Each IPA layer predicts a delta-frame that is composed onto the current frame
        self.protein_frame_update_t = nn.ModuleList([
            nn.Linear(c_s, 3) for _ in range(n_layers)])
        self.protein_frame_update_R = nn.ModuleList([
            nn.Linear(c_s, 3) for _ in range(n_layers)])
        self.peptide_frame_update_t = nn.ModuleList([
            nn.Linear(c_s, 3) for _ in range(n_layers)])
        self.peptide_frame_update_R = nn.ModuleList([
            nn.Linear(c_s, 3) for _ in range(n_layers)])

        # Zero-init frame update heads so initial behavior matches v12
        for heads in [self.protein_frame_update_t, self.protein_frame_update_R,
                      self.peptide_frame_update_t, self.peptide_frame_update_R]:
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        # Output heads
        self.protein_v_head = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, 3))
        self.protein_omega_head = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, 3))
        self.peptide_v_head = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, 3))
        self.peptide_omega_head = nn.Sequential(
            nn.Linear(c_s, c_s), nn.SiLU(), nn.Linear(c_s, 3))

        # TopoGraphEncoder: 从共价键图生成 pair representation（残差相加）
        self.use_topo_graph = (not disable_topology_bias and topo_path is not None
                               and peptide_residue_ids is not None)
        if self.use_topo_graph:
            from topo_graph_encoder import TopoGraphEncoder, build_topology_graph

            # 构建拓扑图（只需一次）
            protein_res_ids = list(range(n_protein)) if topology_graph_scope == "complex" else []
            peptide_res_ids = list(peptide_residue_ids)

            _, edge_index = build_topology_graph(
                topo_path, protein_res_ids, peptide_res_ids)

            # 注册 edge_index 为 buffer
            self.register_buffer('topo_edge_index', edge_index, persistent=False)

            # Graph encoder
            self.topo_graph_encoder = TopoGraphEncoder(
                c_z=c_z, n_layers=2, n_heads=4, dropout=dropout)
            print(f"[Model] TopoGraphEncoder enabled: {len(protein_res_ids) + len(peptide_res_ids)} residues, {edge_index.shape[1]} edges")
        else:
            self.topo_graph_encoder = None

    @staticmethod
    def _build_khop_from_topology(topo_path, peptide_residue_ids, k, return_dist=False):
        """Build K-Hop bias from real bond topology using ParmEd.

        ParmEd can load a .prmtop directly without a companion coordinate file,
        making this robust to any directory layout.

        Args:
            topo_path: path to .prmtop
            peptide_residue_ids: array of global residue indices for the peptide
            k: K-hop cutoff
            return_dist: if True, also return full distance matrix
        Returns:
            bias: (1, N_pep, N_pep) float tensor: 0.0 within k hops, -1e9 beyond
            dist: (1, N_pep, N_pep) int tensor (only if return_dist=True)
        """
        import parmed as pmd
        import numpy as np
        from collections import deque

        pep_ids = np.asarray(peptide_residue_ids)
        N = len(pep_ids)
        # Map global residue index → local peptide index
        g2l = {int(g): l for l, g in enumerate(pep_ids)}

        struct = pmd.load_file(topo_path)

        # Build adjacency list (local indices only, cross-residue bonds)
        adj = [set() for _ in range(N)]
        for bond in struct.bonds:
            ri = bond.atom1.residue.idx
            rj = bond.atom2.residue.idx
            if ri == rj:
                continue  # skip intra-residue bonds
            if ri in g2l and rj in g2l:
                li, lj = g2l[ri], g2l[rj]
                adj[li].add(lj)
                adj[lj].add(li)

        # BFS from each node to compute shortest-hop distances
        dist = np.full((N, N), N + 1, dtype=np.int32)
        for src in range(N):
            dist[src, src] = 0
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist[src, nb] > dist[src, node] + 1:
                        dist[src, nb] = dist[src, node] + 1
                        queue.append(nb)

        bias = np.where(dist <= k, 0.0, -1e9).astype(np.float32)
        if return_dist:
            dist_t = torch.from_numpy(dist).long().unsqueeze(0)   # (1, N, N) int64
            return torch.from_numpy(bias).unsqueeze(0), dist_t
        return torch.from_numpy(bias).unsqueeze(0)

    @staticmethod
    def _build_linear_khop_bias(N, k, cyclic=False):
        """Precompute K-Hop topo bias for a linear or cyclic chain.

        Returns (1, N, N) float tensor:
            0.0  for residue pairs within k hops
            -1e9 for pairs beyond k hops (will be zeroed by softmax)
        """
        idx = torch.arange(N)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (N, N) linear distance
        if cyclic:
            dist = torch.minimum(dist, N - dist)  # shortest path on ring
        bias = torch.where(dist <= k, torch.zeros(N, N), torch.full((N, N), -1e9))
        return bias.unsqueeze(0)  # (1, N, N)

    @staticmethod
    def _build_linear_dist_matrix(N, cyclic=False):
        """Build unsigned shortest-path distance matrix for linear or cyclic chain.

        Returns (1, N, N) int64 tensor.
        """
        idx = torch.arange(N)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        if cyclic:
            dist = torch.minimum(dist, N - dist)
        return dist.unsqueeze(0).long()

    def _init_pair_repr(self, N, device, B, dist_matrix=None):
        """Initialize pair representation with relative position encoding.

        Do NOT use spatial distances here - the input t is noisy (flow-interpolated),
        so distance features would be pure noise at t≈1 and mislead the network.

        Two encoding modes:
        - dist_matrix=None (protein): signed relative position i-j, clamped to
          [-max_rel_pos, max_rel_pos], one-hot into 2*max_rel_pos+1 bins.
          Preserves directionality (N→C vs C→N).
        - dist_matrix provided (peptide with real topology): unsigned shortest-path
          distance from the actual bond graph (includes disulfide shortcut), clamped
          to [0, max_rel_pos], one-hot into max_rel_pos+1 bins.
          Correctly encodes that the two CYS residues are graph-close even if
          sequence-distant. Does NOT assume head-tail ring topology.

        Args:
            N: number of residues
            device: torch device
            B: batch size
            dist_matrix: (1, N, N) int tensor of shortest-path distances from real topology.
                         If None, uses signed linear sequence distance (protein mode).
        """
        if dist_matrix is not None:
            # Peptide mode: real topology shortest-path distances (unsigned)
            # Disulfide bond creates a shortcut: CYS pair appears graph-close
            rel_pos = dist_matrix.squeeze(0).to(device).long()          # (N, N), unsigned
            rel_pos = rel_pos.clamp(0, self.max_rel_pos)
            rel_pos_onehot = F.one_hot(rel_pos, num_classes=self.max_rel_pos + 1).float()
        else:
            # Protein mode: signed linear sequence distance, preserves directionality
            idx = torch.arange(N, device=device)
            rel_pos = (idx.unsqueeze(0) - idx.unsqueeze(1))             # (N, N), signed
            rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
            rel_pos_onehot = F.one_hot(rel_pos, num_classes=2 * self.max_rel_pos + 1).float()

        return rel_pos_onehot.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, protein_R, protein_t, peptide_R, peptide_t,
                time, slow_feat=None, cond_mask=None,
                protein_aa=None, peptide_aa=None):
        """
        Args:
            protein_R: (B, N_protein, 3, 3)
            protein_t: (B, N_protein, 3)
            peptide_R: (B, N_peptide, 3, 3)
            peptide_t: (B, N_peptide, 3)
            time: (B,) flow time
            slow_feat: (B, slow_feat_dim) TICA/PCA features (normalized)
            cond_mask: (B,) bool, True = drop slow_feat condition and use null embedding
            protein_aa: (B, N_protein) or (N_protein,) amino acid type indices
            peptide_aa: (B, N_peptide) or (N_peptide,) amino acid type indices
        Returns:
            protein_v, protein_omega, peptide_v, peptide_omega
        """
        B = protein_R.shape[0]
        device = protein_R.device

        # Time + condition embedding
        t_emb = self.time_emb(time)
        t_emb = self.time_mlp(t_emb)
        if self.disable_tica_conditioning:
            slow_feat = None
            cond_mask = None
        cond_emb = self.cond_enc(slow_feat=slow_feat, cond_mask=cond_mask, batch_size=B)

        fused_cond = self.fuse(torch.cat([t_emb, cond_emb], dim=-1))

        # Initialize single repr: position embedding + AA type embedding
        protein_res_idx = torch.arange(self.n_protein, device=device).unsqueeze(0).expand(B, -1)
        peptide_res_idx = torch.arange(self.n_peptide, device=device).unsqueeze(0).expand(B, -1)

        protein_s = self.protein_res_emb(protein_res_idx)
        peptide_s = self.peptide_res_emb(peptide_res_idx)

        # Add AA type embeddings if provided
        if protein_aa is not None:
            if protein_aa.dim() == 1:
                protein_aa = protein_aa.unsqueeze(0).expand(B, -1)
            protein_s = protein_s + self.aa_type_emb(protein_aa)
        if peptide_aa is not None:
            if peptide_aa.dim() == 1:
                peptide_aa = peptide_aa.unsqueeze(0).expand(B, -1)
            peptide_s = peptide_s + self.aa_type_emb(peptide_aa)

        protein_s = self.protein_single_init(protein_s)
        peptide_s = self.peptide_single_init(peptide_s)

        # Initialize pair repr (sequence position only, no noisy spatial distances)
        protein_z = self._init_pair_repr(self.n_protein, device, B)
        protein_z = self.protein_pair_init(protein_z)

        peptide_z = self._init_pair_repr(
            self.n_peptide, device, B, dist_matrix=self.pep_dist_matrix)
        peptide_z = self.peptide_pair_init(peptide_z)

        # 残差相加：TopoGraphEncoder 的图特征
        if self.use_topo_graph and self.topo_graph_encoder is not None:
            # 拼接 protein_aa 和 peptide_aa 作为动态节点类型
            if protein_aa is not None and peptide_aa is not None:
                if protein_aa.dim() == 1:
                    protein_aa_input = protein_aa
                else:
                    protein_aa_input = protein_aa[0]  # 取第一个样本（假设 batch 内相同）

                if peptide_aa.dim() == 1:
                    peptide_aa_input = peptide_aa
                else:
                    peptide_aa_input = peptide_aa[0]

                full_residue_types = (torch.cat([protein_aa_input, peptide_aa_input], dim=0)
                                      if self.topology_graph_scope == "complex" else peptide_aa_input)

                # 调用 Graph Encoder
                graph_pair_repr = self.topo_graph_encoder(
                    full_residue_types, self.topo_edge_index, batch_size=B)
                # (B, N_total, N_total, c_z)

                # 切分并残差相加
                n_prot = self.n_protein if self.topology_graph_scope == "complex" else 0
                graph_protein_z = graph_pair_repr[:, :n_prot, :n_prot, :]
                graph_peptide_z = graph_pair_repr[:, n_prot:, n_prot:, :]

                if self.topology_graph_scope == "complex":
                    protein_z = protein_z + graph_protein_z
                peptide_z = peptide_z + graph_peptide_z

        # IPA blocks with iterative frame refinement + 3D-aware cross-attention
        # pep_topo_bias: (1, N_pep, N_pep) -> broadcast to (B, N_pep, N_pep)
        pep_topo = None if self.disable_topology_bias else self.pep_topo_bias.expand(B, -1, -1)
        for i in range(self.n_layers):
            # Self-attention (protein: no topo bias; peptide: K-Hop topo bias)
            protein_s = self.protein_ipa_blocks[i](
                protein_s, protein_z, protein_R, protein_t, fused_cond,
                time_step=time, topo_bias=None)
            peptide_s = self.peptide_ipa_blocks[i](
                peptide_s, peptide_z, peptide_R, peptide_t, fused_cond,
                time_step=time, topo_bias=pep_topo)

            # ── Per-layer frame update (FrameDiff-style iterative refinement) ──
            # Zero-initialized at start, so v12 checkpoint loads cleanly
            protein_delta_t = self.protein_frame_update_t[i](protein_s)
            peptide_delta_t = self.peptide_frame_update_t[i](peptide_s)
            if self.equivariant_translations:
                protein_delta_t = (protein_R @ protein_delta_t.unsqueeze(-1)).squeeze(-1)
                peptide_delta_t = (peptide_R @ peptide_delta_t.unsqueeze(-1)).squeeze(-1)
            protein_t = protein_t + protein_delta_t
            protein_R = protein_R @ so3_exp_map(self.protein_frame_update_R[i](protein_s))
            peptide_t = peptide_t + peptide_delta_t
            peptide_R = peptide_R @ so3_exp_map(self.peptide_frame_update_R[i](peptide_s))

            # ── Symmetric cross-attention every 2 layers ──
            # Compute both outputs from pre-update representations to avoid
            # asymmetric information leak
            if i % 2 == 1 and i // 2 < len(self.cross_attn_protein):
                idx = i // 2
                protein_cross = self.cross_attn_protein[idx](
                    protein_s, protein_R, protein_t,
                    peptide_s, peptide_R, peptide_t)
                peptide_cross = self.cross_attn_peptide[idx](
                    peptide_s, peptide_R, peptide_t,
                    protein_s, protein_R, protein_t)
                protein_s = protein_s + protein_cross
                peptide_s = peptide_s + peptide_cross

        # Output
        protein_v = self.protein_v_head(protein_s)
        protein_omega = self.protein_omega_head(protein_s)
        peptide_v = self.peptide_v_head(peptide_s)
        peptide_omega = self.peptide_omega_head(peptide_s)

        if self.equivariant_translations:
            # IPA features are invariant; map local vector heads to world-space
            # translation velocities. Body-frame angular velocities stay local.
            protein_v = (protein_R @ protein_v.unsqueeze(-1)).squeeze(-1)
            peptide_v = (peptide_R @ peptide_v.unsqueeze(-1)).squeeze(-1)

        return protein_v, protein_omega, peptide_v, peptide_omega
