import networkx as nx
import math
import torch
from ALT import apply_ligand_transform_batch_fast

def so3_exp_map(rot_vec):
    theta = torch.norm(rot_vec, dim=-1, keepdim=True)  # (...,1)
    theta = theta + 1e-8
    k = rot_vec / theta

    K = torch.zeros(*rot_vec.shape[:-1], 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
    K[...,0,1] = -k[...,2]; K[...,0,2] =  k[...,1]
    K[...,1,0] =  k[...,2]; K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]; K[...,2,1] =  k[...,0]

    I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)
    I = I.expand_as(K)

    R = I + torch.sin(theta)[...,None]*K + (1-torch.cos(theta))[...,None]*(K@K)
    return R


def to_edge_list(lig_edges):
    if isinstance(lig_edges, list) and len(lig_edges) == 1 and isinstance(lig_edges[0], torch.Tensor):
        edge_tensor = lig_edges[0]
        if edge_tensor.ndim == 2 and edge_tensor.shape[0] == 2:
            src, dst = edge_tensor[0].tolist(), edge_tensor[1].tolist()
            return [(int(i), int(j)) for i, j in zip(src, dst)]
        else:
            raise ValueError(f"Expected 2×N tensor inside list, got shape {edge_tensor.shape}")

    if isinstance(lig_edges, list) and all(
        isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == 2 for x in lig_edges):
        edges_all = []
        for edge_tensor in lig_edges:
            src, dst = edge_tensor[0].tolist(), edge_tensor[1].tolist()
            edges_all.append([(int(i), int(j)) for i, j in zip(src, dst)])
        return edges_all

    if isinstance(lig_edges, torch.Tensor):
        if lig_edges.ndim == 2 and lig_edges.shape[0] == 2:
            src, dst = lig_edges[0].tolist(), lig_edges[1].tolist()
            return [(int(i), int(j)) for i, j in zip(src, dst)]
        elif lig_edges.ndim == 2 and lig_edges.shape[1] == 2:
            return [(int(i), int(j)) for i, j in lig_edges.cpu().numpy()]
        else:
            raise ValueError(f"Unsupported tensor shape: {lig_edges.shape}")

    if isinstance(lig_edges, list):
        edges_out = []
        for e in lig_edges:
            if isinstance(e, torch.Tensor):
                e = e.tolist()
            if isinstance(e, (list, tuple)) and len(e) == 2:
                edges_out.append((int(e[0]), int(e[1])))
        return edges_out

    raise TypeError(f"Unsupported lig_edges type: {type(lig_edges)}")


def find_rotatable_bonds(edges, lig_edge_types):
    if isinstance(lig_edge_types, list):
        if len(lig_edge_types) == 1 and isinstance(lig_edge_types[0], torch.Tensor):
            lig_edge_types = lig_edge_types[0]
        else:
            lig_edge_types = torch.tensor(lig_edge_types)

    if isinstance(lig_edge_types, torch.Tensor):
        lig_edge_types = lig_edge_types.detach().cpu().numpy().tolist()
    lig_edge_types = [int(round(t)) for t in lig_edge_types]

    rotatable_bonds = [(i, j) for (i, j), t in zip(edges, lig_edge_types) if t == 1]
    return rotatable_bonds

def build_rotatable_region_mask(num_atoms, lig_edges, lig_edge_types):
    edges = to_edge_list(lig_edges)
    if isinstance(edges, list) and len(edges) > 0 and isinstance(edges[0], list):
        edges = [e for sub in edges for e in sub]
    G = nx.Graph()
    G.add_nodes_from(range(num_atoms))
    G.add_edges_from(edges)
    rings = nx.cycle_basis(G)
    ring_atoms = set(a for ring in rings for a in ring)

    rotatable_bonds = find_rotatable_bonds(edges, lig_edge_types)

    movable_mask = torch.zeros(num_atoms, dtype=torch.bool)
    for i, j in rotatable_bonds:
        if i not in ring_atoms and j not in ring_atoms:
            movable_mask[i] = True
            movable_mask[j] = True
    return movable_mask

def apply_ligand_transform(pred_pos, ligand_mask, rot_vec, t_pred,
                           delta_lig_res_pos, batch, lig_edges, lig_edge_types,
                           d_min_dict={1: 1.2, 2: 1.2, 3: 1.1},
                           d_max_dict={1: 1.7, 2: 1.5, 3: 1.4},
                           torsion_max=10.0):
    pos_lig = pred_pos[ligand_mask].clone()
    delta = delta_lig_res_pos[ligand_mask].clone() * 0.1
    batch_lig = batch[ligand_mask]
    num_atoms = pos_lig.shape[0]

    R_batch = torch.stack([so3_exp_map(r) for r in rot_vec], dim=0)
    pos_rot = torch.einsum("nd,ndm->nm", pos_lig, R_batch[batch_lig]) + t_pred[batch_lig]
    pos_lig_new = pos_rot.clone()

    movable_mask = build_rotatable_region_mask(num_atoms, lig_edges, lig_edge_types)
    edges = to_edge_list(lig_edges)

    num_groups = 0
    group_labels = -1 * torch.ones(num_atoms, dtype=torch.long, device=pos_lig.device)
    G = [[] for _ in range(num_atoms)]
    for i, j in edges:
        G[i].append(j)
        G[j].append(i)

    def dfs(u, label):
        group_labels[u] = label
        for v in G[u]:
            if movable_mask[v] and group_labels[v] == -1:
                dfs(v, label)

    for i in range(num_atoms):
        if movable_mask[i] and group_labels[i] == -1:
            dfs(i, num_groups)
            num_groups += 1

    G_nx = nx.Graph()
    for i, j in edges:
        G_nx.add_edge(int(i), int(j))
    rings = nx.cycle_basis(G_nx)
    ring_atoms_set = set()
    for ring in rings:
        if len(ring) in [5, 6]:
            ring_atoms_set.update(ring)

    delta_proj = torch.zeros_like(delta)
    for g in range(num_groups):
        idxs = torch.nonzero(group_labels == g).squeeze(-1)
        if len(idxs) == 0:
            continue
        delta_mean = delta[idxs].mean(dim=0)
        max_disp = 0.3 * math.sin(torsion_max / 180 * math.pi)
        if torch.norm(delta_mean) > max_disp:
            delta_mean = delta_mean * (max_disp / torch.norm(delta_mean))
        idxs_ring = [i.item() for i in idxs if i.item() in ring_atoms_set]
        idxs_nonring = [i.item() for i in idxs if i.item() not in ring_atoms_set]
        if len(idxs_ring) > 0:
            delta_ring = delta_mean * 0.2
            pos_lig_new[idxs_ring] += delta_ring
        if len(idxs_nonring) > 0:
            idxs_nonring_tensor = torch.tensor(idxs_nonring, dtype=torch.long, device=delta_proj.device)
            delta_proj[idxs_nonring_tensor] = delta_mean

    pos_lig_new[movable_mask] += delta_proj[movable_mask]
    for idx, (i, j) in enumerate(edges):
        bond_type = int(round(lig_edge_types[0][idx].item()))
        d_min = d_min_dict.get(bond_type, 1.0)
        d_max = d_max_dict.get(bond_type, 2.0)
        vec = pos_lig_new[j] - pos_lig_new[i]
        d = torch.norm(vec)
        if d > d_max:
            vec = vec * (d_max / d)
            pos_lig_new[j] = pos_lig_new[i] + vec
        elif d < d_min:
            vec = vec * (d_min / d)
            pos_lig_new[j] = pos_lig_new[i] + vec

    min_nonbond_dist = 2.0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if (i, j) in edges or (j, i) in edges:
                continue
            if i in ring_atoms_set and j in ring_atoms_set:
                continue
            vec = pos_lig_new[j] - pos_lig_new[i]
            dist = torch.norm(vec)
            if dist < min_nonbond_dist:
                scale = min_nonbond_dist / (dist + 1e-8)
                delta_adjust = vec * (scale - 1.0)
                if movable_mask[j]:
                    pos_lig_new[j] += delta_adjust / 2
                if movable_mask[i]:
                    pos_lig_new[i] -= delta_adjust / 2

    return pos_lig_new


import torch


def compute_dynamic_directional_scales(pos_lig, pos_pocket, t_pred, base_motion=2.0):
    """
    根据运动方向和口袋空间的相对几何关系，动态决定整体运动的缩放与正负号。
    """
    if pos_pocket.shape[0] == 0:
        return base_motion if base_motion is not None else 2.0

    # 如果外部没有指定 base_motion，给一个合理的默认基准
    if base_motion is None:
        base_motion = 2.0

    # 1. 计算当前配体中心
    lig_center = pos_lig.mean(dim=0)

    # 2. 找到最靠近配体中心的口袋局部区域（避免全局平均被远处原子稀释）
    dist_to_center = torch.norm(pos_pocket - lig_center, dim=-1)
    close_pocket_mask = dist_to_center < (dist_to_center.min() + 4.0)
    relevant_pocket = pos_pocket[close_pocket_mask] if close_pocket_mask.sum() > 0 else pos_pocket

    # 3. 计算口袋环境对配体中心的排斥矢量（从口袋指向配体 = 开放方向）
    vec_pocket_to_lig = lig_center - relevant_pocket
    dists = torch.norm(vec_pocket_to_lig, dim=-1, keepdim=True) + 1e-5
    repulsion_dir = (vec_pocket_to_lig / (dists ** 3)).sum(dim=0)
    repulsion_dir = repulsion_dir / (torch.norm(repulsion_dir) + 1e-6)

    # 4. 判断预测的平移方向 t_pred 是否顺应开放空间
    t_dir = t_pred.mean(dim=0) if t_pred.dim() > 1 else t_pred
    t_dir_norm = t_dir / (torch.norm(t_dir) + 1e-6)

    # 计算方向对齐度（余弦相似度）
    alignment = torch.dot(t_dir_norm, repulsion_dir).item()
    # 5. 动态决定平移步长与方向
    min_dist = dists.min().item()

    if min_dist < 2.2:  # 极度危险距离（发生严重碰撞）
        # 强制反向拉回，且加大力度逃离碰撞区
        motion_scale = -1.2 * base_motion if alignment < 0 else 0.2 * base_motion
    else:
        # 正常区域：方向对，加大步长；方向错，反向或微调
        if alignment >= 0:
            motion_scale = base_motion * (0.5 + 1.2 * alignment)  # 顺着方向，最多放大到1.7倍
        else:
            motion_scale = base_motion * 0.5 * alignment  # 逆着方向，变成负数拉回
    return motion_scale


def rodrigues_rotation_batched(coords, axes, thetas):
    """ 高效的批处理罗德里格斯旋转 """
    B = axes.size(0)
    N = coords.size(0)
    axes = axes / (torch.norm(axes, dim=-1, keepdim=True) + 1e-8)
    cos_t = torch.cos(thetas).view(B, 1, 1)
    sin_t = torch.sin(thetas).view(B, 1, 1)

    coords_exp = coords.unsqueeze(0).expand(B, N, 3)
    axes_exp = axes.unsqueeze(1).expand(B, N, 3)

    dot_prod = torch.sum(coords_exp * axes_exp, dim=-1, keepdim=True)
    cross_prod = torch.linalg.cross(axes_exp, coords_exp)

    rotated = coords_exp * cos_t + cross_prod * sin_t + axes_exp * dot_prod * (1.0 - cos_t)
    return rotated


def evaluate_poses_batched(moved_coords, p_u, axes, thetas, pos_pocket):
    """ 并行评估所有候选路径的得分 """
    B = axes.size(0)
    rotated_coords_batch = rodrigues_rotation_batched(moved_coords, axes, thetas) + p_u

    # 矩阵乘法加速距离计算
    a_sq = rotated_coords_batch.pow(2).sum(dim=-1, keepdim=True)
    b_sq = pos_pocket.pow(2).sum(dim=-1).view(1, 1, -1)
    ab = torch.matmul(rotated_coords_batch, pos_pocket.t())

    dist_matrix = torch.sqrt(torch.clamp(a_sq + b_sq - 2 * ab, min=1e-6))
    min_dists, _ = torch.min(dist_matrix, dim=-1)

    collision_atoms = torch.sum(min_dists < 2.0, dim=-1)
    # 维持较宽松的黄金接触区间 (2.2Å ~ 4.5Å)
    contact_atoms = torch.sum((min_dists >= 2.2) & (min_dists <= 4.5), dim=-1)

    scores = contact_atoms.float() - (collision_atoms.float() * 100.0)
    return scores, collision_atoms


def compute_bond_torsion_scale_vectorized(current_pos, downstream_idxs, pos_pocket,
                                          axis, p_u, raw_delta_theta, base_torsion=0.035):
    """
    完全消除 CPU 同步的 GPU 版本
    """
    if pos_pocket.shape[0] == 0 or downstream_idxs.shape[0] == 0:
        # 返回 Tensor，避免破坏 GPU 流
        return (torch.tensor(base_torsion, device=current_pos.device),
                raw_delta_theta * base_torsion, axis)

    device = current_pos.device
    moved_coords = current_pos[downstream_idxs] - p_u
    axis = axis / (torch.norm(axis) + 1e-8)

    max_multiplier = 6.0
    aggressive_scale = base_torsion * max_multiplier

    # ================= 阶段 1 =================
    # 使用 torch.tensor 预设常数，或者在外部定义好
    fast_scales = torch.tensor([aggressive_scale, base_torsion, -base_torsion, -aggressive_scale], device=device)
    fast_axes = axis.unsqueeze(0).expand(4, 3)
    # 替换原本的 .item() 循环创建：使用张量外积
    fast_thetas = (raw_delta_theta * fast_scales).unsqueeze(-1)

    fast_scores, fast_collisions = evaluate_poses_batched(moved_coords, p_u, fast_axes, fast_thetas, pos_pocket)

    # 用 torch.argmax 配合条件掩码替代 Python 的 if-else 提前退出
    best_fast_idx = torch.argmax(fast_scores)
    # 检查是否满足提前退出条件 (Tensor 操作)
    early_stop_mask = (fast_collisions[best_fast_idx] == 0) & (fast_scores[best_fast_idx] > 0)

    # ================= 阶段 2 & 3 =================
    # 构造正交轴
    ref_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
    # 避免使用 .item() 的条件判断，改用 torch.where
    cond = torch.abs(torch.dot(axis, ref_vec)) > 0.9
    ref_vec = torch.where(cond, torch.tensor([0.0, 1.0, 0.0], device=device), ref_vec)

    ortho_v1 = torch.linalg.cross(axis, ref_vec)
    ortho_v1 = ortho_v1 / (torch.norm(ortho_v1) + 1e-8)
    ortho_v2 = torch.linalg.cross(axis, ortho_v1)

    tilt_angle = 0.35
    cos_t, sin_t = math.cos(tilt_angle), math.sin(tilt_angle)  # 用标准库或提前算好

    perturbed_axes = torch.stack([
        axis * cos_t + ortho_v1 * sin_t,
        axis * cos_t - ortho_v1 * sin_t,
        axis * cos_t + ortho_v2 * sin_t,
        axis * cos_t - ortho_v2 * sin_t
    ])

    search_scales = torch.tensor([aggressive_scale, -aggressive_scale, base_torsion, -base_torsion],
                                 device=device).repeat(4)
    search_axes = perturbed_axes.repeat_interleave(4, dim=0)
    search_thetas = (raw_delta_theta * search_scales).unsqueeze(-1)

    side_scores, side_collisions = evaluate_poses_batched(moved_coords, p_u, search_axes, search_thetas, pos_pocket)

    # 拼装结果
    all_scores = torch.cat([fast_scores, side_scores])
    all_collisions = torch.cat([fast_collisions, side_collisions])
    all_axes = torch.cat([fast_axes, search_axes])
    all_scales = torch.cat([fast_scales, search_scales])

    best_idx = torch.argmax(all_scores)

    # 最终决策同样使用 Tensor 条件选择
    collision_threshold = downstream_idxs.shape[0] * 0.2
    fail_mask = (all_collisions[best_idx] > collision_threshold) & (all_scores[best_idx] < -5)

    chosen_scale = torch.where(fail_mask, torch.tensor(0.0, device=device), all_scales[best_idx])
    delta_theta_update = raw_delta_theta * chosen_scale
    chosen_axis = torch.where(fail_mask.unsqueeze(-1), axis, all_axes[best_idx])

    # 如果触发了第一阶段的提前退出，则覆盖
    if early_stop_mask:
        return fast_scales[best_fast_idx], fast_thetas[best_fast_idx], axis

    return chosen_scale, delta_theta_update, chosen_axis

def compute_bond_torsion_scale(current_pos, downstream_idxs, pos_pocket,
                               axis, p_u, delta_theta, base_torsion=0.035):
    """
    大步长全速寻路旋转机制
    同时对“新方向”应用“大振幅”，大幅增加滑出口袋死角的概率。
    """
    if base_torsion is None:
        base_torsion = 0.035

    if pos_pocket.shape[0] == 0 or len(downstream_idxs) == 0:
        return base_torsion, delta_theta * base_torsion, axis

    device = current_pos.device
    moved_coords = current_pos[downstream_idxs] - p_u
    axis = axis / (torch.norm(axis) + 1e-8)

    # 定义激进放大系数
    max_multiplier = 6.0
    aggressive_scale = base_torsion * max_multiplier

    # ================= 阶段 1: 极速一维通道（保留原轴的高低档测试） =================
    fast_scales = [aggressive_scale, base_torsion, -base_torsion, -aggressive_scale]
    fast_axes = axis.unsqueeze(0).expand(4, 3)
    fast_thetas = torch.tensor([delta_theta * s for s in fast_scales], device=device).unsqueeze(-1)

    fast_scores, fast_collisions = evaluate_poses_batched(moved_coords, p_u, fast_axes, fast_thetas, pos_pocket)

    # 如果原轴的大步长或常规步长已经非常完美（不撞且有接触），直接提前退出
    best_fast_idx = torch.argmax(fast_scores).item()
    if fast_collisions[best_fast_idx] == 0 and fast_scores[best_fast_idx] > 0:
        chosen_scale = fast_scales[best_fast_idx]
        return chosen_scale, delta_theta * chosen_scale, axis

    # ================= 阶段 2: 向量化三维“大跨步”寻路 =================
    # 构造正交扰动轴（左、右、上、下）
    ref_vec = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.abs(torch.dot(axis, ref_vec)) > 0.9:
        ref_vec = torch.tensor([0.0, 1.0, 0.0], device=device)
    ortho_v1 = torch.linalg.cross(axis, ref_vec)
    ortho_v1 = ortho_v1 / (torch.norm(ortho_v1) + 1e-8)
    ortho_v2 = torch.linalg.cross(axis, ortho_v1)

    # 稍微加大轴倾斜角度（从 12.5 度提高到 20 度，让左右摆动的幅度更明显）
    tilt_angle = 0.35
    cos_t, sin_t = torch.cos(torch.tensor(tilt_angle)), torch.sin(torch.tensor(tilt_angle))

    perturbed_axes = torch.stack([
        axis * cos_t + ortho_v1 * sin_t,  # 左偏
        axis * cos_t - ortho_v1 * sin_t,  # 右偏
        axis * cos_t + ortho_v2 * sin_t,  # 上偏
        axis * cos_t - ortho_v2 * sin_t  # 下偏
    ])

    # 【核心修改】：每个新方向，同时测试“常规步长”和“大步长”（正反大跨步交替）
    # 这样既保证了探索的广度，又赋予了新方向突破屏障的动力
    search_scales = [aggressive_scale, -aggressive_scale, base_torsion, -base_torsion] * 4  # 每个轴测试 4 档
    search_axes = perturbed_axes.repeat_interleave(4, dim=0)  # 4 * 4 = 16 种组合
    search_thetas = torch.tensor([delta_theta * s for s in search_scales], device=device).unsqueeze(-1)

    # 并行计算这 16 种大跨步姿态
    side_scores, side_collisions = evaluate_poses_batched(moved_coords, p_u, search_axes, search_thetas, pos_pocket)

    # ================= 阶段 3: 综合决策 =================
    all_scores = torch.cat([fast_scores, side_scores])
    all_collisions = torch.cat([fast_collisions, side_collisions])
    all_axes = torch.cat([fast_axes, search_axes])
    all_scales = fast_scales + search_scales

    best_idx = torch.argmax(all_scores).item()

    # 容忍度适当放宽：只要大跨步能显著减少碰撞（比如碰撞原子数少于 20%），我们依然允许它运动
    if all_collisions[best_idx] > (len(downstream_idxs) * 0.2) and all_scores[best_idx] < -5:
        # 如果无论怎么大跨步、怎么变向都撞得一团糟，再原地锁死
        return 0.0, 0.0, axis

    chosen_scale = all_scales[best_idx]
    chosen_axis = all_axes[best_idx]
    # print(chosen_scale)
    return chosen_scale, delta_theta * chosen_scale, chosen_axis

def apply_ligand_transform_batch(pred_pos, ligand_mask, rot_vec, t_pred,
                                 delta_lig_res_pos, batch,
                                 lig_edges_list, lig_edge_types_list):  # 0.025 , motion_scale=2, # 2.5 torsion_scale=0.035

    device = pred_pos.device
    pos_lig = pred_pos[ligand_mask].clone()
    pos_pocket = pred_pos[~ligand_mask].clone()
    delta = delta_lig_res_pos[ligand_mask].clone()
    batch_lig = batch[ligand_mask]
    motion_scale = compute_dynamic_directional_scales(pos_lig, pos_pocket, t_pred, base_motion=2.0)
    R_batch = torch.stack([so3_exp_map(motion_scale * r) for r in rot_vec], dim=0)
    pos_rot = torch.einsum("nd,ndm->nm", pos_lig, R_batch[batch_lig]) + motion_scale * t_pred[batch_lig]
    pos_lig_new = pos_rot.clone()
    # torsion_scale
    num_atoms_list = [e.max().item()+1 for e in lig_edges_list]
    offsets = torch.cumsum(torch.tensor([0] + num_atoms_list[:-1], device=device), dim=0)
    # movable_mask = torch.zeros(pos_lig.shape[0], dtype=torch.bool, device=device)

    for offset, n_atoms, edges_tensor, edge_types in zip(offsets, num_atoms_list, lig_edges_list, lig_edge_types_list):
        edges = edges_tensor.t().to(device)                    # [E,2]
        if isinstance(edge_types, torch.Tensor):
            edge_types = edge_types.detach().cpu().numpy().ravel().tolist()
        edge_types = [int(round(t)) for t in edge_types]
        current_pos = pos_lig_new[offset: offset + n_atoms].clone()

        # 2. 建立 NetworkX 图
        G = nx.Graph()
        G.add_nodes_from(range(n_atoms))
        edges_list = edges.tolist()
        G.add_edges_from(edges_list)

        # 3. 筛选可旋转键（且不在环内的单键）
        rings = nx.cycle_basis(G)
        ring_edges = set()
        for ring in rings:
            for i in range(len(ring)):
                u, v = ring[i], ring[(i + 1) % len(ring)]
                ring_edges.add(tuple(sorted((u, v))))

        rotatable_bonds = []
        for idx, (e, t) in enumerate(zip(edges_list, edge_types)):
            if t == 1 and tuple(sorted(e)) not in ring_edges:
                rotatable_bonds.append((e[0], e[1], idx))  # 记录两端原子和键的索引

        for u, v, bond_idx in rotatable_bonds:
            raw_delta_theta = delta[offset + u][0].item()
            p_u = current_pos[u]
            p_v = current_pos[v]
            axis = p_v - p_u
            axis_len = torch.norm(axis)
            if axis_len < 1e-6:
                continue
            axis = axis / axis_len  # 归一化旋转轴
            G_split = G.copy()
            G_split.remove_edge(u, v)
            downstream_atoms = list(nx.node_connected_component(G_split, v))
            downstream_tensor_idxs = torch.tensor(downstream_atoms, dtype=torch.long, device=device)
            moved_coords = current_pos[downstream_tensor_idxs] - p_u
            chosen_scale, delta_theta_update, axis_update = compute_bond_torsion_scale_vectorized(
                current_pos=current_pos,
                downstream_idxs=downstream_tensor_idxs,
                pos_pocket=pos_pocket,
                axis=axis,
                p_u=p_u,
                raw_delta_theta=raw_delta_theta
            )
            # print(dynamic_torsion_scale)
            # theta = final_delta_theta * dynamic_torsion_scale
            if torch.abs(delta_theta_update) < 1e-5:
                continue
            rotated_coords = rodrigues_rotation(moved_coords, axis_update, delta_theta_update)

            current_pos[downstream_tensor_idxs] = rotated_coords + p_u

        pos_lig_new[offset: offset + n_atoms] = current_pos
    return pos_lig_new


def rodrigues_rotation(coords, axis, theta):
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # R = cos_t * I + (1 - cos_t) * (u x u^T) + sin_t * [u]_x
    u_x, u_y, u_z = axis[0], axis[1], axis[2]
    K = torch.tensor([
        [0, -u_z, u_y],
        [u_z, 0, -u_x],
        [-u_y, u_x, 0]
    ], device=coords.device, dtype=coords.dtype)

    UU_T = torch.outer(axis, axis)
    I = torch.eye(3, device=coords.device, dtype=coords.dtype)
    R = cos_t * I + (1 - cos_t) * UU_T + sin_t * K  # [3, 3]
    return torch.matmul(coords, R.t())