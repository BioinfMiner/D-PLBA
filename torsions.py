import torch

def so3_exp_map(rot_vec):
    theta = torch.norm(rot_vec, dim=-1, keepdim=True)  # (...,1)
    theta = theta + 1e-8
    k = rot_vec / theta  #

    K = torch.zeros(*rot_vec.shape[:-1], 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
    K[...,0,1] = -k[...,2]; K[...,0,2] =  k[...,1]
    K[...,1,0] =  k[...,2]; K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]; K[...,2,1] =  k[...,0]

    I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)
    I = I.expand_as(K)

    R = I + torch.sin(theta)[...,None]*K + (1-torch.cos(theta))[...,None]*(K@K)
    return R

def dihedral_angle(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (torch.norm(b1, dim=-1, keepdim=True) + 1e-8)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = torch.cross(b1, v, dim=-1).sum(-1) * w.sum(-1)
    return torch.atan2(y, x)

import networkx as nx
def find_ring_bonds_fast(lig_edges, max_ring_size=6):
    edges = lig_edges.t().tolist()
    G = nx.Graph()
    G.add_edges_from(edges)

    ring_bonds = set()
    for cycle in nx.cycle_basis(G):
        if len(cycle) <= max_ring_size:
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                ring_bonds.add(tuple(sorted((a, b))))
    return ring_bonds

def extract_torsions_from_topology(
    pos_lig, lig_edges_local, lig_edge_types,
    skip_ring=True, max_ring_size=6, max_torsions=10000
):
    N = pos_lig.shape[0]
    device = pos_lig.device

    src, dst = lig_edges_local
    neighbors = [[] for _ in range(N)]
    for i, j in zip(src.tolist(), dst.tolist()):
        neighbors[i].append(j)
        neighbors[j].append(i)

    ring_bonds = set()
    if skip_ring:
        ring_bonds = find_ring_bonds_fast(lig_edges_local, max_ring_size=max_ring_size)

    torsions = []
    edge_info = list(zip(src.tolist(), dst.tolist(), lig_edge_types.tolist()))

    for i, j, btype in edge_info:
        if int(btype) != 1:
            continue
        if tuple(sorted((i, j))) in ring_bonds:
            continue

        A_cand = [a for a in neighbors[i] if a != j]
        D_cand = [d for d in neighbors[j] if d != i]

        for a in A_cand[:2]:
            for d in D_cand[:2]:
                torsions.append((a, i, j, d))
                if len(torsions) >= max_torsions:
                    break
            if len(torsions) >= max_torsions:
                break
        if len(torsions) >= max_torsions:
            break

    if len(torsions) == 0:
        return None, None

    torsion_idx = torch.tensor(torsions, dtype=torch.long, device=device)
    p0, p1, p2, p3 = [pos_lig[torsion_idx[:, k]] for k in range(4)]
    torsion_true = dihedral_angle(p0, p1, p2, p3)

    return torsion_idx, torsion_true

def torsion_loss_vectorized(pos_pred, torsion_idx, torsion_true):
    if torsion_idx is None or torsion_idx.shape[0] == 0:
        return torch.tensor(0.0, device=pos_pred.device)

    p0, p1, p2, p3 = [pos_pred[torsion_idx[:, k]] for k in range(4)]  # (M,3)

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = torch.norm(b1, dim=-1, keepdim=True) + 1e-8
    b1_unit = b1 / b1_norm

    v = b0 - (b0 * b1_unit).sum(-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(-1, keepdim=True) * b1_unit

    x = (v * w).sum(-1)
    y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)

    torsion_pred = torch.atan2(y, x)

    diff = torch.atan2(torch.sin(torsion_pred - torsion_true.to(pos_pred.device)),
                       torch.cos(torsion_pred - torsion_true.to(pos_pred.device)))

    return (diff ** 2).mean()