import networkx as nx
import math
import torch

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

def apply_ligand_transform_batch_fast(pred_pos, ligand_mask, rot_vec, t_pred,
                                 delta_lig_res_pos, batch,
                                 lig_edges_list, lig_edge_types_list, motion_scale=3,
                                 torsion_scale=1):
    device = pred_pos.device
    pos_lig = pred_pos[ligand_mask].clone()  # [N_lig, 3]
    delta = delta_lig_res_pos[ligand_mask].clone() * torsion_scale
    batch_lig = batch[ligand_mask]

    R_batch = torch.stack([so3_exp_map(motion_scale * r) for r in rot_vec], dim=0)
    pos_rot = torch.einsum("nd,ndm->nm", pos_lig, R_batch[batch_lig]) + motion_scale * t_pred[batch_lig]
    pos_lig_new = pos_rot.clone()
    # torsion_scale
    num_atoms_list = [e.max().item()+1 for e in lig_edges_list]
    offsets = torch.cumsum(torch.tensor([0] + num_atoms_list[:-1], device=device), dim=0)
    movable_mask = torch.zeros(pos_lig.shape[0], dtype=torch.bool, device=device)
    for offset, n_atoms, edges_tensor, edge_types in zip(offsets, num_atoms_list, lig_edges_list, lig_edge_types_list):
        edges = edges_tensor.t().to(device)                    # [E,2]
        if isinstance(edge_types, torch.Tensor):
            edge_types = edge_types.detach().cpu().numpy().ravel().tolist()
        edge_types = [int(round(t)) for t in edge_types]
        rotatable_bonds = [e for e, t in zip(edges.tolist(), edge_types) if t == 1]
        G = nx.Graph()
        G.add_nodes_from(range(n_atoms))
        G.add_edges_from(edges.tolist())
        rings = nx.cycle_basis(G)
        ring_atoms = set(a for ring in rings for a in ring)

        # movable_mask
        local_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        for i,j in rotatable_bonds:
            if i not in ring_atoms and j not in ring_atoms:
                local_mask[i] = True
                local_mask[j] = True
        movable_mask[offset:offset+n_atoms] = local_mask

    # torsion delta
    for offset, n_atoms in zip(offsets, num_atoms_list):
        mask_local = movable_mask[offset:offset+n_atoms]
        idxs_local = torch.nonzero(mask_local, as_tuple=False).squeeze(-1)
        if idxs_local.numel() > 0:
            delta_mean = delta[offset:offset+n_atoms][idxs_local].mean(dim=0)
            pos_lig_new[offset:offset+n_atoms][idxs_local] += delta_mean

    return pos_lig_new