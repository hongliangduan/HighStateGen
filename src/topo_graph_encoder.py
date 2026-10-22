#!/usr/bin/env python3
"""
Topology Graph Encoder - 从共价键图生成 pair representation

用途：为 Flow Matching 提供"骨架蓝图"，解决长链蛋白在纯噪声下找不到邻居的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import parmed as pmd
import numpy as np


class TopoGraphEncoder(nn.Module):
    """
    从分子拓扑图生成 pair representation

    输入：
        - 残基类型 (N_res,)
        - 邻接矩阵 (N_res, N_res) 或 edge_index
    输出：
        - pair_repr (B, N_res, N_res, c_z)
    """

    def __init__(self, c_z=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.c_z = c_z
        self.n_layers = n_layers

        # 残基类型 embedding (20 种氨基酸 + 1 unknown)
        self.residue_embedding = nn.Embedding(21, c_z)

        # Graph layers (使用简单的 self-attention 作为 GNN)
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(c_z, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Pair representation 生成：outer product + MLP
        self.pair_proj = nn.Sequential(
            nn.Linear(c_z * 2, c_z * 2),
            nn.ReLU(),
            nn.Linear(c_z * 2, c_z)
        )

    def forward(self, residue_types, edge_index, batch_size=1):
        """
        Args:
            residue_types: (N_res,) 残基类型索引 (0-19 for 20 AA)
            edge_index: (2, N_edges) 边索引 [src, dst]
            batch_size: batch 大小

        Returns:
            pair_repr: (B, N_res, N_res, c_z)
        """
        N_res = residue_types.shape[0]
        device = residue_types.device

        # 1. 节点特征初始化
        node_feat = self.residue_embedding(residue_types)  # (N_res, c_z)

        # 2. Graph message passing
        for layer in self.graph_layers:
            node_feat = layer(node_feat, edge_index)  # (N_res, c_z)

        # 3. 生成 pair representation (outer product)
        # node_i: (N_res, 1, c_z), node_j: (1, N_res, c_z)
        node_i = node_feat.unsqueeze(1).expand(N_res, N_res, self.c_z)
        node_j = node_feat.unsqueeze(0).expand(N_res, N_res, self.c_z)
        pair_feat = torch.cat([node_i, node_j], dim=-1)  # (N_res, N_res, 2*c_z)

        pair_repr = self.pair_proj(pair_feat)  # (N_res, N_res, c_z)

        # 4. 扩展到 batch 维度
        pair_repr = pair_repr.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return pair_repr


class GraphAttentionLayer(nn.Module):
    """简化的 Graph Attention Layer"""

    def __init__(self, c_z, n_heads=4, dropout=0.1):
        super().__init__()
        self.c_z = c_z
        self.n_heads = n_heads
        self.head_dim = c_z // n_heads

        self.q_proj = nn.Linear(c_z, c_z)
        self.k_proj = nn.Linear(c_z, c_z)
        self.v_proj = nn.Linear(c_z, c_z)
        self.out_proj = nn.Linear(c_z, c_z)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(c_z)

    def forward(self, x, edge_index):
        """
        Args:
            x: (N, c_z) 节点特征
            edge_index: (2, E) 边索引
        Returns:
            x_out: (N, c_z) 更新后的节点特征
        """
        N = x.shape[0]
        residual = x

        # Multi-head attention
        q = self.q_proj(x).view(N, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.n_heads, self.head_dim)

        # 构建邻接矩阵（稀疏 -> 稠密）
        adj = torch.zeros(N, N, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Attention scores: (H, N, N)
        attn = torch.einsum('nhd,mhd->hnm', q, k) / (self.head_dim ** 0.5)

        # Mask: 扩展 adj 到 (1, N, N) 用于 broadcasting
        mask = adj.unsqueeze(0) == 0  # (1, N, N)
        attn = attn.masked_fill(mask, float('-inf'))  # (H, N, N)

        attn = F.softmax(attn, dim=-1)  # (H, N, N)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum('hnm,mhd->nhd', attn, v)  # (N, H, D)
        out = out.reshape(N, self.c_z)
        out = self.out_proj(out)

        return self.norm(residual + out)


def build_topology_graph(topology_path, protein_residue_ids, peptide_residue_ids):
    """
    从 prmtop 构建拓扑图

    Args:
        topology_path: .prmtop 文件路径
        protein_residue_ids: 蛋白残基 ID 列表
        peptide_residue_ids: 肽链残基 ID 列表

    Returns:
        residue_types: (N_res,) 残基类型索引
        edge_index: (2, N_edges) 边索引
    """
    parm = pmd.load_file(topology_path)

    # 氨基酸名称到索引的映射
    AA_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    aa_to_idx = {name: i for i, name in enumerate(AA_NAMES)}

    all_residue_ids = list(protein_residue_ids) + list(peptide_residue_ids)
    N_res = len(all_residue_ids)

    # 提取残基类型
    residue_types = []
    for res_id in all_residue_ids:
        res = parm.residues[res_id]
        res_name = res.name[:3]  # 取前3个字符
        idx = aa_to_idx.get(res_name, 20)  # 20 = unknown
        residue_types.append(idx)

    residue_types = torch.tensor(residue_types, dtype=torch.long)

    # 提取骨架成键关系（只保留 C-N 肽键）
    edges = []
    for bond in parm.bonds:
        atom1, atom2 = bond.atom1, bond.atom2
        res1_id = atom1.residue.idx
        res2_id = atom2.residue.idx

        # 只保留残基间的键（肽键）
        if res1_id != res2_id and res1_id in all_residue_ids and res2_id in all_residue_ids:
            idx1 = all_residue_ids.index(res1_id)
            idx2 = all_residue_ids.index(res2_id)
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, N_edges)

    return residue_types, edge_index
