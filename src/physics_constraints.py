"""
增强的物理约束模块
包含：键长、键角、二面角约束
支持 backbone heavy atom subset 提取与子图生成
支持 backbone_enhanced subset（backbone + S + crosslink + 1-hop neighbors）
支持二硫键拓扑提取

NOTE: Topology recognition only. The enhanced subset identifies sulfur atoms
      and crosslink bonds for topology-aware processing, but the actual model
      operates on backbone frames (N/CA/C/O) only. Does NOT reconstruct or
      constrain sulfur geometry explicitly.
"""
import numpy as np
import torch
import torch.nn as nn
import mdtraj as md


def extract_disulfide_bonds(topology_path):
    """从 prmtop 提取二硫键残基对。

    Args:
        topology_path: path to .prmtop file

    Returns:
        list of (res_idx1, res_idx2) tuples for disulfide-bonded residues
    """
    try:
        import parmed as pmd
    except ImportError:
        print("[WARN] parmed not installed, cannot extract disulfide bonds")
        return []

    struct = pmd.load_file(topology_path)
    disulfide_pairs = []

    for bond in struct.bonds:
        a1, a2 = bond.atom1, bond.atom2
        # Check if both atoms are sulfur
        if a1.element_name == 'S' and a2.element_name == 'S':
            res1_idx = a1.residue.idx
            res2_idx = a2.residue.idx
            # Ensure res1 < res2 for consistency
            if res1_idx > res2_idx:
                res1_idx, res2_idx = res2_idx, res1_idx
            disulfide_pairs.append((res1_idx, res2_idx))

    return disulfide_pairs


# ─────────────────────────────────────────────────────────────
# Backbone heavy atom extraction & subgraph utilities
# ─────────────────────────────────────────────────────────────

# Backbone heavy atom names (covers standard + most non-standard residues)
_BACKBONE_HEAVY_NAMES = {"N", "CA", "C", "O"}


def extract_enhanced_subset_indices(topology, include_1hop_neighbors=True,
                                    ref_coords_nm=None):
    """提取增强子集：backbone + S + crosslink/ring-closure atoms + 1-hop neighbors.

    选择规则（全自动从 topology 识别）：
    1. 所有 backbone heavy atoms: N, CA, C, O
    2. 所有硫原子 (element == 'S')
    3. 所有参与非局部键的原子（crosslink/ring-closure）
       - 非局部键 = 连接残基 |i-j| > 1 的键
       - 包括 head-tail 距离检测（< 0.2 nm 视为潜在闭环）
    4. (可选) 规则 2-3 中原子的 1-hop 重原子邻居

    Args:
        topology: mdtraj.Topology
        include_1hop_neighbors: 是否包含关键原子的 1-hop 重原子邻居
        ref_coords_nm: (n_atoms, 3) 参考坐标 (nm)，用于 head-tail 距离检测

    Returns:
        dict with keys:
            selected_atom_indices : np.ndarray (M,)
            atom_names            : list[str]
            residue_names         : list[str]
            residue_ids           : np.ndarray (M,)
            atom_categories       : list[str]  ("backbone", "sulfur", "crosslink", "neighbor")
    """
    # Step 1: Collect backbone atoms
    backbone_set = set()
    atom_info = {}  # global_idx -> (name, res_name, res_id)

    for atom in topology.atoms:
        if atom.element is None or atom.element.symbol == "H":
            continue
        atom_info[atom.index] = (atom.name, atom.residue.name, atom.residue.index)
        if atom.name in _BACKBONE_HEAVY_NAMES:
            backbone_set.add(atom.index)

    # Step 2: Collect all S atoms
    sulfur_set = set()
    for atom in topology.atoms:
        if atom.element is not None and atom.element.symbol == "S":
            sulfur_set.add(atom.index)

    # Step 3: Find non-local bonds (crosslinks/ring-closures)
    crosslink_set = set()
    for bond in topology.bonds:
        a, b = bond[0].index, bond[1].index
        if a not in atom_info or b not in atom_info:
            continue
        res_a = atom_info[a][2]
        res_b = atom_info[b][2]
        if abs(res_a - res_b) > 1:  # Non-local bond
            crosslink_set.add(a)
            crosslink_set.add(b)

    # Step 3b: Head-tail proximity detection per chain
    #   If ref_coords_nm is provided, check if chain head and tail heavy atoms
    #   are within bonding distance but lack an explicit bond → warn and include.
    if ref_coords_nm is not None:
        for chain in topology.chains:
            residues = list(chain.residues)
            if len(residues) < 3:
                continue
            first_res, last_res = residues[0], residues[-1]
            for a in first_res.atoms:
                if a.index not in atom_info:
                    continue
                for b in last_res.atoms:
                    if b.index not in atom_info:
                        continue
                    dist = float(np.linalg.norm(
                        ref_coords_nm[a.index] - ref_coords_nm[b.index]))
                    if dist < 0.20:
                        has_bond = any(
                            (bd[0].index == a.index and bd[1].index == b.index) or
                            (bd[1].index == a.index and bd[0].index == b.index)
                            for bd in topology.bonds
                        )
                        if not has_bond:
                            print(f"[EnhancedSubset] WARNING: potential head-tail "
                                  f"bond missing in topology: "
                                  f"{a.index} {a.name}({first_res.name}"
                                  f"{first_res.index}) -- "
                                  f"{b.index} {b.name}({last_res.name}"
                                  f"{last_res.index})  dist={dist:.3f} nm")
                        # Include both atoms regardless
                        crosslink_set.add(a.index)
                        crosslink_set.add(b.index)

    # Step 4: Optional 1-hop neighbors of S and crosslink atoms
    neighbor_set = set()
    if include_1hop_neighbors:
        critical_atoms = (sulfur_set | crosslink_set) - backbone_set
        for bond in topology.bonds:
            a, b = bond[0].index, bond[1].index
            if a in critical_atoms and b in atom_info and b not in backbone_set:
                neighbor_set.add(b)
            if b in critical_atoms and a in atom_info and a not in backbone_set:
                neighbor_set.add(a)

    # Combine all sets with priority: backbone > sulfur > crosslink > neighbor
    selected = backbone_set | sulfur_set | crosslink_set | neighbor_set
    selected_sorted = sorted(selected)

    # Assign categories
    indices, names, res_names, res_ids, categories = [], [], [], [], []
    for idx in selected_sorted:
        name, rname, rid = atom_info[idx]
        indices.append(idx)
        names.append(name)
        res_names.append(rname)
        res_ids.append(rid)

        if idx in backbone_set:
            categories.append("backbone")
        elif idx in sulfur_set:
            categories.append("sulfur")
        elif idx in crosslink_set:
            categories.append("crosslink")
        else:
            categories.append("neighbor")

    if len(indices) == 0:
        raise ValueError("No atoms selected in enhanced subset")

    info = {
        "selected_atom_indices": np.array(indices, dtype=np.int64),
        "atom_names": names,
        "residue_names": res_names,
        "residue_ids": np.array(res_ids, dtype=np.int64),
        "atom_categories": categories,
    }

    # Statistics
    n_res = len(set(res_ids))
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    count_str = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
    print(f"[EnhancedSubset] {len(indices)} atoms from {n_res} residues ({count_str})")

    return info


def extract_backbone_heavy_indices(topology):
    """从 mdtraj topology 中提取 backbone heavy atom 的索引及元信息。

    识别规则：
      1. atom.element.symbol != 'H'  (排除所有氢)
      2. atom.name in {N, CA, C, O}  (标准主链命名)
    不依赖标准 20 AA，只要残基保留常规主链命名即可兼容。

    Returns:
        dict with keys:
            selected_atom_indices : np.ndarray (M,)  全局原子索引
            atom_names            : list[str]         对应 atom name
            residue_names         : list[str]         对应 residue name
            residue_ids           : np.ndarray (M,)   对应 residue index (0-based)
    """
    indices = []
    atom_names = []
    residue_names = []
    residue_ids = []

    for atom in topology.atoms:
        if atom.element is not None and atom.element.symbol == "H":
            continue
        if atom.name in _BACKBONE_HEAVY_NAMES:
            indices.append(atom.index)
            atom_names.append(atom.name)
            residue_names.append(atom.residue.name)
            residue_ids.append(atom.residue.index)

    if len(indices) == 0:
        raise ValueError(
            "No backbone heavy atoms (N/CA/C/O) found. "
            "Check that the topology uses standard backbone atom naming."
        )

    info = {
        "selected_atom_indices": np.array(indices, dtype=np.int64),
        "atom_names": atom_names,
        "residue_names": residue_names,
        "residue_ids": np.array(residue_ids, dtype=np.int64),
    }

    # 统计
    n_res = len(set(residue_ids))
    counts = {}
    for name in atom_names:
        counts[name] = counts.get(name, 0) + 1
    count_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[BackboneSubset] {len(indices)} atoms from {n_res} residues ({count_str})")

    return info


def build_backbone_subgraph(topology, backbone_info, ref_coords_nm):
    """基于 backbone heavy atom subset 生成子图和物理约束索引。

    Args:
        topology      : mdtraj.Topology
        backbone_info : extract_backbone_heavy_indices() 的返回值
        ref_coords_nm : (n_atoms_full, 3) 参考坐标 (nm)

    Returns:
        dict with keys:
            sub_bonds          : np.ndarray (M_b, 2)   subset 内部 bond (local idx)
            sub_ref_bond_lengths : torch.Tensor (M_b,)
            sub_angles         : np.ndarray (M_a, 3)   subset 内部 angle (local idx)
            sub_ref_angles     : torch.Tensor (M_a,) or None
            sub_dihedrals      : np.ndarray (M_d, 4)   subset 内部 dihedral (local idx)
            sub_ref_dihedrals  : torch.Tensor (M_d,) or None
            sub_edge_index     : torch.LongTensor (2, E)  1-hop bond edges (local idx, directed)
            sub_edge_type      : torch.FloatTensor (E, 1)  all ones (single type)
            global_to_local    : dict {global_idx -> local_idx}
    """
    selected = set(backbone_info["selected_atom_indices"].tolist())
    # global -> local index mapping
    g2l = {g: l for l, g in enumerate(backbone_info["selected_atom_indices"])}
    n_sub = len(g2l)

    # ── 1. Filter bonds ──
    sub_bonds = []
    for bond in topology.bonds:
        a, b = bond[0].index, bond[1].index
        if a in selected and b in selected:
            sub_bonds.append((g2l[a], g2l[b]))

    sub_bonds = np.array(sub_bonds, dtype=np.int64) if sub_bonds else np.empty((0, 2), dtype=np.int64)

    # Reference bond lengths
    if len(sub_bonds) > 0:
        sel_idx = backbone_info["selected_atom_indices"]
        sub_coords = ref_coords_nm[sel_idx]  # (n_sub, 3)
        p0 = sub_coords[sub_bonds[:, 0]]
        p1 = sub_coords[sub_bonds[:, 1]]
        sub_ref_bl = torch.from_numpy(np.linalg.norm(p0 - p1, axis=-1).astype(np.float32))
    else:
        sub_ref_bl = torch.zeros(0)

    # ── 2. Build adjacency for angles / dihedrals ──
    adjacency = {}
    for i, j in sub_bonds:
        adjacency.setdefault(int(i), []).append(int(j))
        adjacency.setdefault(int(j), []).append(int(i))

    # ── 3. Angles (i-j-k): 只保留骨架 N/CA/C 原子的角度 ──
    # 骨架角度定义二级结构的局部几何，侧链/S/crosslink 角度不需要约束
    atom_names_list = backbone_info.get("atom_names", [])
    backbone_heavy = {"N", "CA", "C", "O"}

    sub_angles = []
    for j, neighbors in adjacency.items():
        j_name = atom_names_list[j] if j < len(atom_names_list) else ""
        if j_name not in backbone_heavy:
            continue  # 中心原子必须是骨架原子
        for ni in range(len(neighbors)):
            for nk in range(ni + 1, len(neighbors)):
                i_name = atom_names_list[neighbors[ni]] if neighbors[ni] < len(atom_names_list) else ""
                k_name = atom_names_list[neighbors[nk]] if neighbors[nk] < len(atom_names_list) else ""
                if i_name in backbone_heavy and k_name in backbone_heavy:
                    sub_angles.append((neighbors[ni], j, neighbors[nk]))
    sub_angles = np.array(sub_angles, dtype=np.int64) if sub_angles else np.empty((0, 3), dtype=np.int64)

    sub_ref_angles = None
    if len(sub_angles) > 0:
        sel_idx = backbone_info["selected_atom_indices"]
        sub_coords = ref_coords_nm[sel_idx][np.newaxis]  # (1, n_sub, 3)
        ai, aj, ak = sub_angles[:, 0], sub_angles[:, 1], sub_angles[:, 2]
        v_ji = sub_coords[0, ai] - sub_coords[0, aj]
        v_jk = sub_coords[0, ak] - sub_coords[0, aj]
        v_ji_n = v_ji / (np.linalg.norm(v_ji, axis=-1, keepdims=True) + 1e-8)
        v_jk_n = v_jk / (np.linalg.norm(v_jk, axis=-1, keepdims=True) + 1e-8)
        cos_a = np.clip(np.sum(v_ji_n * v_jk_n, axis=-1), -1.0, 1.0)
        sub_ref_angles = torch.from_numpy(np.arccos(cos_a).astype(np.float32))

    # ── 4. Dihedrals (i-j-k-l): 只保留骨架 φ/ψ/ω 二面角 ──
    # φ = C(i-1)-N(i)-CA(i)-C(i)
    # ψ = N(i)-CA(i)-C(i)-N(i+1)
    # ω = CA(i)-C(i)-N(i+1)-CA(i+1)
    # 这三类二面角直接决定二级结构，其他二面角不约束
    sub_dihedrals = []
    for j, k in sub_bonds:
        j, k = int(j), int(k)
        j_name = atom_names_list[j] if j < len(atom_names_list) else ""
        k_name = atom_names_list[k] if k < len(atom_names_list) else ""
        # 只处理骨架键：N-CA, CA-C, C-N
        if not ({j_name, k_name} <= backbone_heavy):
            continue
        if not ({j_name, k_name} & {"N", "CA", "C"}):
            continue
        for i in adjacency.get(j, []):
            if i == k:
                continue
            i_name = atom_names_list[i] if i < len(atom_names_list) else ""
            if i_name not in backbone_heavy:
                continue
            for l in adjacency.get(k, []):
                if l == j:
                    continue
                l_name = atom_names_list[l] if l < len(atom_names_list) else ""
                if l_name not in backbone_heavy:
                    continue
                sub_dihedrals.append((i, j, k, l))
    sub_dihedrals = np.array(sub_dihedrals, dtype=np.int64) if sub_dihedrals else np.empty((0, 4), dtype=np.int64)

    sub_ref_dihedrals = None
    if len(sub_dihedrals) > 0:
        sel_idx = backbone_info["selected_atom_indices"]
        sub_coords = ref_coords_nm[sel_idx]
        di, dj, dk, dl = sub_dihedrals[:, 0], sub_dihedrals[:, 1], sub_dihedrals[:, 2], sub_dihedrals[:, 3]
        b1 = sub_coords[dj] - sub_coords[di]
        b2 = sub_coords[dk] - sub_coords[dj]
        b3 = sub_coords[dl] - sub_coords[dk]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_n = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
        n2_n = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
        cos_d = np.clip(np.sum(n1_n * n2_n, axis=-1), -1.0, 1.0)
        b2_n = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
        m1 = np.cross(n1_n, b2_n)
        sign = np.sign(np.sum(m1 * n2_n, axis=-1))
        sub_ref_dihedrals = torch.from_numpy((sign * np.arccos(cos_d)).astype(np.float32))

    # ── 5. Edge index / edge type (directed, from bonds) ──
    src_list, dst_list = [], []
    for i, j in sub_bonds:
        src_list.extend([int(i), int(j)])
        dst_list.extend([int(j), int(i)])
    sub_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.zeros(2, 0, dtype=torch.long)
    sub_edge_type = torch.ones(sub_edge_index.shape[1], 1, dtype=torch.float32)

    print(f"[BackboneSubgraph] {n_sub} atoms, {len(sub_bonds)} bonds, "
          f"{len(sub_angles)} angles, {len(sub_dihedrals)} dihedrals, "
          f"{sub_edge_index.shape[1]} directed edges")

    return {
        "sub_bonds": sub_bonds,
        "sub_ref_bond_lengths": sub_ref_bl,
        "sub_angles": sub_angles,
        "sub_ref_angles": sub_ref_angles,
        "sub_dihedrals": sub_dihedrals,
        "sub_ref_dihedrals": sub_ref_dihedrals,
        "sub_edge_index": sub_edge_index,
        "sub_edge_type": sub_edge_type,
        "global_to_local": g2l,
    }


class EnhancedPhysicsConstraints:
    """
    增强的物理约束

    包含：
    1. 键长约束（Bond）
    2. 键角约束（Angle）
    3. 二面角约束（Dihedral）
    """

    def __init__(self, topology_path: str):
        """
        Args:
            topology_path: PDB或PRMTOP文件路径
        """
        import os

        # 加载拓扑
        traj = None

        # 处理AMBER拓扑文件
        if topology_path.endswith('.prmtop'):
            # 查找对应的坐标文件
            base_path = topology_path.replace('.prmtop', '')
            coord_file = None

            # 尝试常见的坐标文件扩展名（优先PDB，然后AMBER格式）
            for ext in ['.pdb', '.inpcrd', '.rst7', '.ncrst', '.restrt']:
                candidate = base_path + ext
                if os.path.exists(candidate):
                    coord_file = candidate
                    break

            if coord_file is None:
                raise FileNotFoundError(
                    f"找不到与 {topology_path} 对应的坐标文件。"
                    f"请确保存在 {base_path}.pdb 或 {base_path}.inpcrd"
                )

            # 加载AMBER文件
            traj = md.load(coord_file, top=topology_path)
            print(f"[Physics] 加载AMBER拓扑: {topology_path}")
            print(f"[Physics] 加载坐标文件: {coord_file}")
        else:
            # 处理PDB或其他格式
            try:
                traj = md.load(topology_path)
            except Exception:
                traj = md.load_pdb(topology_path)

        self.topology = traj.topology
        pos = traj.xyz[0]  # 参考坐标 (n_atoms, 3) in nm

        # 1. 提取键（Bonds）
        bonds = [(b[0].index, b[1].index) for b in self.topology.bonds]
        if len(bonds) == 0:
            raise ValueError("No bonds found in topology")

        self.bonds = np.array(bonds, dtype=np.int64)  # (M_bond, 2)

        # 计算参考键长
        p0 = pos[self.bonds[:, 0], :]
        p1 = pos[self.bonds[:, 1], :]
        ref_bond_lengths = np.linalg.norm(p0 - p1, axis=-1)
        self.ref_bond_lengths = torch.from_numpy(ref_bond_lengths).float()

        print(f"[Physics] 加载 {len(self.bonds)} 个键")

        # 2. 提取键角（Angles）
        # 键角定义为三个连续原子：i-j-k
        angles = self._extract_angles()
        self.angles = np.array(angles, dtype=np.int64)  # (M_angle, 3)

        if len(self.angles) > 0:
            # 计算参考键角
            ref_angle_values = self._compute_angles(pos[None, :, :])
            self.ref_angles = torch.from_numpy(ref_angle_values[0]).float()
            print(f"[Physics] 加载 {len(self.angles)} 个键角")
        else:
            self.ref_angles = None
            print(f"[Physics] 未找到键角")

        # 3. 提取二面角（Dihedrals）
        # 二面角定义为四个连续原子：i-j-k-l
        dihedrals = self._extract_dihedrals()
        self.dihedrals = np.array(dihedrals, dtype=np.int64) if len(dihedrals) > 0 else None

        if self.dihedrals is not None and len(self.dihedrals) > 0:
            # 计算参考二面角
            ref_dihedral_values = self._compute_dihedrals(pos[None, :, :])
            self.ref_dihedrals = torch.from_numpy(ref_dihedral_values[0]).float()
            print(f"[Physics] 加载 {len(self.dihedrals)} 个二面角")
        else:
            self.ref_dihedrals = None
            print(f"[Physics] 未找到二面角")

    def _extract_angles(self):
        """
        从拓扑中提取键角
        键角 = 三个连续原子 i-j-k，其中 i-j 和 j-k 都是键
        """
        angles = []

        # 构建邻接表
        adjacency = {}
        for i, j in self.bonds:
            if i not in adjacency:
                adjacency[i] = []
            if j not in adjacency:
                adjacency[j] = []
            adjacency[i].append(j)
            adjacency[j].append(i)

        # 对每个原子j，找到所有连接的原子对 (i, k)
        for j in adjacency:
            neighbors = adjacency[j]
            if len(neighbors) >= 2:
                # 所有邻居对形成键角
                for idx1 in range(len(neighbors)):
                    for idx2 in range(idx1 + 1, len(neighbors)):
                        i = neighbors[idx1]
                        k = neighbors[idx2]
                        angles.append((i, j, k))

        return angles

    def _extract_dihedrals(self):
        """
        从拓扑中提取二面角
        二面角 = 四个连续原子 i-j-k-l
        """
        dihedrals = []

        # 构建邻接表
        adjacency = {}
        for i, j in self.bonds:
            if i not in adjacency:
                adjacency[i] = []
            if j not in adjacency:
                adjacency[j] = []
            adjacency[i].append(j)
            adjacency[j].append(i)

        # 对每条键 j-k，找到连接的原子 i 和 l
        for j, k in self.bonds:
            # 找到连接到j的原子（除了k）
            i_candidates = [x for x in adjacency.get(j, []) if x != k]
            # 找到连接到k的原子（除了j）
            l_candidates = [x for x in adjacency.get(k, []) if x != j]

            # 组合形成二面角
            for i in i_candidates:
                for l in l_candidates:
                    dihedrals.append((i, j, k, l))

        return dihedrals

    def _compute_angles(self, coords):
        """
        计算键角

        Args:
            coords: (B, N, 3) 坐标

        Returns:
            angles: (B, M_angle) 键角值（弧度）
        """
        if self.angles is None or len(self.angles) == 0:
            return None

        i, j, k = self.angles[:, 0], self.angles[:, 1], self.angles[:, 2]

        # 获取坐标
        pos_i = coords[:, i, :]  # (B, M, 3)
        pos_j = coords[:, j, :]
        pos_k = coords[:, k, :]

        # 计算向量
        vec_ji = pos_i - pos_j  # j -> i
        vec_jk = pos_k - pos_j  # j -> k

        # 归一化
        vec_ji_norm = vec_ji / (np.linalg.norm(vec_ji, axis=-1, keepdims=True) + 1e-8)
        vec_jk_norm = vec_jk / (np.linalg.norm(vec_jk, axis=-1, keepdims=True) + 1e-8)

        # 计算夹角
        cos_angle = np.sum(vec_ji_norm * vec_jk_norm, axis=-1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.arccos(cos_angle)

        return angles

    def _compute_dihedrals(self, coords):
        """
        计算二面角

        Args:
            coords: (B, N, 3) 坐标

        Returns:
            dihedrals: (B, M_dihedral) 二面角值（弧度）
        """
        if self.dihedrals is None or len(self.dihedrals) == 0:
            return None

        i, j, k, l = (self.dihedrals[:, 0], self.dihedrals[:, 1],
                      self.dihedrals[:, 2], self.dihedrals[:, 3])

        # 获取坐标
        pos_i = coords[:, i, :]
        pos_j = coords[:, j, :]
        pos_k = coords[:, k, :]
        pos_l = coords[:, l, :]

        # 计算向量
        b1 = pos_j - pos_i
        b2 = pos_k - pos_j
        b3 = pos_l - pos_k

        # 计算法向量
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # 归一化
        n1_norm = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
        n2_norm = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)

        # 计算二面角
        cos_dihedral = np.sum(n1_norm * n2_norm, axis=-1)
        cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)

        # 计算符号
        m1 = np.cross(n1_norm, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8))
        sign = np.sign(np.sum(m1 * n2_norm, axis=-1))

        dihedrals = sign * np.arccos(cos_dihedral)

        return dihedrals

    def compute_bond_loss(self, coords: torch.Tensor, use_huber: bool = True, delta: float = 0.5):
        """
        计算键长约束损失

        Args:
            coords: (B, N, 3) 坐标（nm）
            use_huber: 是否使用Huber loss
            delta: Huber loss阈值

        Returns:
            loss: 标量损失
        """
        device = coords.device
        bonds = torch.from_numpy(self.bonds).long().to(device)

        idx0 = bonds[:, 0]
        idx1 = bonds[:, 1]

        pos0 = coords[:, idx0, :]  # (B, M, 3)
        pos1 = coords[:, idx1, :]

        cur_lens = torch.norm(pos0 - pos1, dim=-1)  # (B, M)
        ref = self.ref_bond_lengths.to(device).unsqueeze(0)  # (1, M)

        if use_huber:
            diff = torch.abs(cur_lens - ref)
            loss = torch.where(
                diff < delta,
                0.5 * (cur_lens - ref) ** 2,
                delta * (diff - 0.5 * delta)
            )
            loss = torch.mean(loss)
        else:
            loss = torch.mean((cur_lens - ref) ** 2)

        return loss

    def compute_angle_loss(self, coords: torch.Tensor):
        """
        计算键角约束损失

        Args:
            coords: (B, N, 3) 坐标（nm）

        Returns:
            loss: 标量损失
        """
        if self.angles is None or len(self.angles) == 0:
            return torch.tensor(0.0, device=coords.device)

        device = coords.device
        angles = torch.from_numpy(self.angles).long().to(device)

        i, j, k = angles[:, 0], angles[:, 1], angles[:, 2]

        pos_i = coords[:, i, :]
        pos_j = coords[:, j, :]
        pos_k = coords[:, k, :]

        # 计算向量
        vec_ji = pos_i - pos_j
        vec_jk = pos_k - pos_j

        # 归一化
        vec_ji_norm = vec_ji / (torch.norm(vec_ji, dim=-1, keepdim=True) + 1e-8)
        vec_jk_norm = vec_jk / (torch.norm(vec_jk, dim=-1, keepdim=True) + 1e-8)

        # 计算夹角
        cos_angle = torch.sum(vec_ji_norm * vec_jk_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        cur_angles = torch.acos(cos_angle)

        # 参考角度
        ref = self.ref_angles.to(device).unsqueeze(0)

        # MSE loss
        loss = torch.mean((cur_angles - ref) ** 2)

        return loss

    def compute_dihedral_loss(self, coords: torch.Tensor):
        """
        计算二面角约束损失

        Args:
            coords: (B, N, 3) 坐标（nm）

        Returns:
            loss: 标量损失
        """
        if self.dihedrals is None or len(self.dihedrals) == 0:
            return torch.tensor(0.0, device=coords.device)

        device = coords.device
        dihedrals = torch.from_numpy(self.dihedrals).long().to(device)

        i, j, k, l = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]

        pos_i = coords[:, i, :]
        pos_j = coords[:, j, :]
        pos_k = coords[:, k, :]
        pos_l = coords[:, l, :]

        # 计算向量
        b1 = pos_j - pos_i
        b2 = pos_k - pos_j
        b3 = pos_l - pos_k

        # 计算法向量
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        # 归一化
        n1_norm = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
        n2_norm = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)

        # 计算二面角
        cos_dihedral = torch.sum(n1_norm * n2_norm, dim=-1)
        cos_dihedral = torch.clamp(cos_dihedral, -1.0, 1.0)

        # 计算符号
        b2_norm = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        m1 = torch.cross(n1_norm, b2_norm, dim=-1)
        sign = torch.sign(torch.sum(m1 * n2_norm, dim=-1))

        cur_dihedrals = sign * torch.acos(cos_dihedral)

        # 参考二面角
        ref = self.ref_dihedrals.to(device).unsqueeze(0)

        # 周期性MSE loss（考虑2π周期性）
        diff = cur_dihedrals - ref
        # 将差值映射到 [-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        loss = torch.mean(diff ** 2)

        return loss
