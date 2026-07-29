import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn import LayerNorm
from networks.EGNN import Feature_Refine, Lig_feature, Lig_div_feature
import torch_scatter
from ApplyLigandTrans import apply_ligand_transform, apply_ligand_transform_batch


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_channels, hidden_channels):
        super().__init__(aggr='add')
        self.phi_e = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.phi_x = nn.Linear(hidden_channels, 1)
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        self.scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x, pos, edge_index, edge_attr):

        check_index_safety(edge_index, x)
        row, col = edge_index
        # [E, 1] pairwise distance
        dist_raw = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        dist = torch.clamp(dist_raw, min=1e-4, max=1e4)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        edge_input = torch.cat([x[row], x[col], edge_attr, dist], dim=-1)
        if torch.isnan(edge_input).any() or torch.isinf(edge_input).any():
            print("NaN or Inf detected in edge_input", flush=True)
            print(edge_input)
        m_ij = self.phi_e(edge_input)
        dx = (pos[row] - pos[col]) * self.phi_x(m_ij)
        dx = torch.tanh(dx) * self.scale
        delta_pos = scatter(dx, row, dim=0, dim_size=x.size(0))
        deg = scatter(torch.ones_like(m_ij), row, dim=0, dim_size=x.size(0))
        deg = deg.clamp(min=1.0)
        agg_m = scatter(m_ij, row, dim=0, dim_size=x.size(0)) / deg

        h = self.phi_h(torch.cat([x, agg_m], dim=-1))
        return h, delta_pos


def check_index_safety(edge_index, x, name=""):
    ei = edge_index.cpu()
    if ei.max().item() >= x.shape[0] or ei.min().item() < 0:
        print(
            f"[{name}] edge_index over edge_index?[{ei.min().item()}, {ei.max().item()}], x.shape[0] = {x.shape[0]}")
        raise RuntimeError("Invalid edge_index before CUDA")


def pl_subgraph_batch(x, pos, edge_index, edge_attr, protein_lens, batch):
    device = x.device
    num_graphs = batch.max().item() + 1

    x_sub_all, pos_sub_all = [], []
    edge_index_all, edge_attr_all = [], []
    batch_sub_all = []

    node_offset = 0  #
    for i in range(num_graphs):
        node_mask = batch == i
        node_idx = node_mask.nonzero(as_tuple=False).view(-1)

        x_i = x[node_idx]
        pos_i = pos[node_idx]

        #
        idx_map = -torch.ones(x.size(0), dtype=torch.long, device=device)
        idx_map[node_idx] = torch.arange(node_idx.size(0), device=device)

        #
        edge_mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
        edge_index_i = edge_index[:, edge_mask]
        edge_attr_i = edge_attr[edge_mask]

        #
        edge_index_i = idx_map[edge_index_i]

        N_i = x_i.size(0)
        protein_len = protein_lens[i]
        ligand_idx = torch.arange(protein_len, N_i, device=device)

        row, col = edge_index_i
        mask1 = (row < protein_len) & (col >= protein_len)
        mask2 = (col < protein_len) & (row >= protein_len)
        mask = mask1 | mask2

        protein_connected_idx = torch.unique(torch.cat([
            row[mask][row[mask] < protein_len],
            col[mask][col[mask] < protein_len]
        ]))

        sub_nodes = torch.cat([ligand_idx, protein_connected_idx]).unique()
        idx_mapping = -torch.ones(N_i, dtype=torch.long, device=device)
        idx_mapping[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)

        x_sub = x_i[sub_nodes]
        pos_sub = pos_i[sub_nodes]

        edge_mask_sub = (idx_mapping[row] >= 0) & (idx_mapping[col] >= 0)
        edge_index_sub = torch.stack([
            idx_mapping[row[edge_mask_sub]],
            idx_mapping[col[edge_mask_sub]]
        ], dim=0)
        edge_attr_sub = edge_attr_i[edge_mask_sub]

        #
        edge_index_sub = edge_index_sub + node_offset

        x_sub_all.append(x_sub)
        pos_sub_all.append(pos_sub)
        edge_index_all.append(edge_index_sub)
        edge_attr_all.append(edge_attr_sub)
        batch_sub_all.append(torch.full((x_sub.size(0),), i, dtype=torch.long, device=device))
        node_offset += x_sub.size(0)

    x_sub = torch.cat(x_sub_all, dim=0)
    pos_sub = torch.cat(pos_sub_all, dim=0)
    edge_index_sub = torch.cat(edge_index_all, dim=1)
    edge_attr_sub = torch.cat(edge_attr_all, dim=0)
    subgraph_batch = torch.cat(batch_sub_all, dim=0)
    return x_sub, pos_sub, edge_index_sub, edge_attr_sub, subgraph_batch


class AffinityPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super(AffinityPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=0.2)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  #
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_pool = global_add_pool(x, batch)  # [num_graphs, hidden_dim]
        out = self.mlp(x_pool)  # [num_graphs, 1]
        return out.view(-1)  # [num_graphs]


def dynamic_pocket_mask_radius(pos, protein_lens, batch, cutoff=8.0):
    device = pos.device
    pocket_mask = torch.zeros(pos.size(0), dtype=torch.bool, device=device)
    num_graphs = batch.max().item() + 1
    for i in range(num_graphs):
        node_mask = batch == i
        node_idx = node_mask.nonzero(as_tuple=False).view(-1)
        pos_i = pos[node_idx]
        protein_len = protein_lens[i]
        if protein_len == 0:
            continue
        pos_protein = pos_i[:protein_len]
        pos_ligand = pos_i[protein_len:]
        if pos_ligand.size(0) == 0:
            continue
        dists = torch.cdist(pos_protein, pos_ligand)
        protein_in_pocket = (dists <= cutoff).any(dim=1)
        protein_pocket_idx = node_idx[:protein_len][protein_in_pocket]
        pocket_mask[protein_pocket_idx] = True
    return pocket_mask


def dynamic_pocket_mask(pos, protein_lens, batch, k=5):
    device = pos.device
    pocket_mask = torch.zeros(pos.size(0), dtype=torch.bool, device=device)
    num_graphs = batch.max().item() + 1
    for i in range(num_graphs):
        node_mask = batch == i
        node_idx = node_mask.nonzero(as_tuple=False).view(-1)
        pos_i = pos[node_idx]
        protein_len = protein_lens[i]
        ligand_idx_local = torch.arange(protein_len, pos_i.size(0), device=device)
        if len(ligand_idx_local) == 0:
            continue
        pos_protein = pos_i[:protein_len]
        pos_ligand = pos_i[ligand_idx_local]
        dists = torch.cdist(pos_protein, pos_ligand)
        k_select = min(k, protein_len)
        _, topk_idx = torch.topk(dists, k_select, largest=False, dim=0)
        protein_pocket_idx = node_idx[:protein_len][topk_idx.flatten()]
        pocket_mask[protein_pocket_idx] = True
        pocket_mask[node_idx[ligand_idx_local]] = True
    # num_true = pocket_mask.sum().item()
    # print(f"Number of True values: {num_true}")
    return pocket_mask


from torch_geometric.nn import knn_graph
import time


def pocket_subgraph(x, pos, batch, pocket_mask, k=10):
    node_idx = pocket_mask.nonzero(as_tuple=False).view(-1)
    x_sub = x[node_idx]
    pos_sub = pos[node_idx]
    batch_sub = batch[node_idx]
    edge_index = knn_graph(pos_sub, k=k, batch=batch_sub, loop=False, flow='source_to_target')
    return x_sub, pos_sub, edge_index, batch_sub


class DualDeltaPredictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_channels, 3)
        )

    def forward(self, node_rep, delta_pos_total, mask):
        delta = self.linear(node_rep)
        return delta * mask


from torch_scatter import scatter_softmax, scatter_sum, scatter_mean


class LigandRigidPredictor(nn.Module):
    def __init__(self, in_dim, hidden_channels=128):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim + 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 6)
        )

    def forward(self, x, pos, batch, ligand_mask):
        x_lig = x[ligand_mask]
        pos_lig = pos[ligand_mask]
        batch_lig = batch[ligand_mask]

        if x_lig.size(0) == 0:
            num_batches = batch.max().item() + 1
            return (torch.zeros((num_batches, 3), device=x.device),
                    torch.zeros((num_batches, 3), device=x.device))

        feats = torch.cat([x_lig, pos_lig], dim=-1)  # [N_lig, in_dim+3]
        attn_score = self.attn_mlp(feats).squeeze(-1)  # [N_lig]
        attn_weight = scatter_softmax(attn_score, batch_lig, dim=0)  # [N_lig]
        pooled = scatter_sum(attn_weight.unsqueeze(-1) * feats, batch_lig, dim=0)  # [num_batches, in_dim+3]
        out = self.mlp(pooled)  # [num_batches, 6]
        rot_vec, t_pred = out[:, :3], out[:, 3:]  #
        return rot_vec, t_pred


def so3_exp_map(rot_vec):
    theta = torch.norm(rot_vec, dim=-1, keepdim=True)  # (...,1)
    theta = theta + 1e-8
    k = rot_vec / theta
    K = torch.zeros(*rot_vec.shape[:-1], 3, 3, device=rot_vec.device, dtype=rot_vec.dtype)
    K[..., 0, 1] = -k[..., 2];
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1];
    K[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=rot_vec.device, dtype=rot_vec.dtype)
    I = I.expand_as(K)
    R = I + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)
    return R


class Predictor(nn.Module):
    def __init__(self, hidden_channels, out_channels=1, dropout=0.15):
        super(Predictor, self).__init__()
        self.mlp_complex = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.mlp_pl = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2)
        )
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 4),
            nn.Tanh(),
            nn.Linear(hidden_channels // 4, 1)
        )
        self.temp_linear = nn.Linear(5, 1)

        self.energy_scale = nn.Parameter(torch.tensor(0.5))

        self.mlp_out = nn.Sequential(
            nn.LayerNorm(hidden_channels * 3),
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),  # nn.SiLU()
            nn.Linear(hidden_channels, out_channels)
        )
        self.aff_out1 = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),  # nn.SiLU()
            nn.Linear(hidden_channels, out_channels)
        )
        self.aff_out2 = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, global_graph_fea, dynamic_complex, ene_list):
        attn_logits = self.attn_score(dynamic_complex)
        enes = ene_list.unsqueeze(-1)
        attn_logits = attn_logits + self.energy_scale * enes
        attn = torch.softmax(attn_logits, dim=0)

        fea_complex = self.temp_linear((attn * dynamic_complex).permute(1, 2, 0)).squeeze(-1)
        fea_complex = self.mlp_complex(fea_complex).squeeze(-1)
        fea_pl = self.mlp_pl(global_graph_fea).squeeze(-1)
        fea_all = torch.cat((fea_pl, fea_complex), dim=-1)
        C = fea_pl.shape[1]
        # fea_all = self.laynorm(fea_all)
        out = self.mlp_out(fea_all).squeeze(-1)
        aff_out1 = self.aff_out1(fea_pl).squeeze(-1)
        aff_out2 = self.aff_out2(fea_complex).squeeze(-1)
        return out, aff_out1, aff_out2


def weighted_scatter_mean(node_rep, pos, pocket_poss, pocket_masks, batch, num_graphs):
    pocket_centers = []
    for i in range(len(pocket_poss)):
        pocket_center = pocket_poss[i].mean(dim=0)  # [3]
        pocket_centers.append(pocket_center)
    pocket_centers = torch.stack(pocket_centers, dim=0)  # [N_pocket, 3]

    node_center = torch.zeros_like(pos)  # [num_nodes, 3]
    for i, mask in enumerate(pocket_masks):
        node_center[mask] = pocket_centers[i]

    dist = torch.norm(pos - node_center, dim=-1)  # [num_nodes]

    weight = 1.0 / (1.0 + dist)

    weighted_node = node_rep * weight.unsqueeze(-1)  # [num_nodes, hidden_dim]
    weighted_sum = torch_scatter.scatter_add(weighted_node, batch, dim=0, dim_size=num_graphs)
    weight_sum = torch_scatter.scatter_add(weight, batch, dim=0, dim_size=num_graphs).unsqueeze(-1)
    global_graph_fea = weighted_sum / weight_sum  # [num_graphs, hidden_dim]
    return global_graph_fea


def weighted_scatter_mean_fast(node_rep, pos, pocket_poss, pocket_masks, batch, num_graphs):
    device = node_rep.device
    num_nodes = pos.size(0)
    pocket_centers = torch.stack([p.mean(dim=0) for p in pocket_poss], dim=0).to(device)
    node_to_pocket_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for i, mask in enumerate(pocket_masks):
        node_to_pocket_idx[mask] = i
    node_center = pocket_centers[node_to_pocket_idx]
    dist = torch.norm(pos - node_center, dim=-1)
    weight = 1.0 / (1.0 + dist)
    weighted_node = node_rep * weight.unsqueeze(-1)
    weighted_sum = torch_scatter.scatter_add(weighted_node, batch, dim=0, dim_size=num_graphs)
    weight_sum = torch_scatter.scatter_add(weight, batch, dim=0, dim_size=num_graphs).unsqueeze(-1)
    global_graph_fea = weighted_sum / weight_sum
    return global_graph_fea


class GatedFusion(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gate = nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, f_finetune, f_pretrain):
        gate = torch.sigmoid(self.gate(torch.cat((f_finetune, f_pretrain), dim=-1)))
        return gate * f_finetune + (1 - gate) * f_pretrain


class MultiHeadScatterAttentionPool(nn.Module):
    def __init__(self, in_dim, out_dim=32, num_heads=4, attn_hidden=32):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.attn_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1)
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_heads)
        ])

    def forward(self, node_feats, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1

        graph_embs = []

        for attn_mlp, proj in zip(self.attn_mlps, self.proj):
            attn_scores = attn_mlp(node_feats).squeeze(-1)

            max_per_graph = torch_scatter.scatter_max(
                attn_scores, batch, dim=0, dim_size=num_graphs
            )[0]
            attn_scores = attn_scores - max_per_graph[batch]

            exp_scores = torch.exp(attn_scores)
            sum_per_graph = torch_scatter.scatter_add(
                exp_scores, batch, dim=0, dim_size=num_graphs
            )

            attn_weights = exp_scores / (sum_per_graph[batch] + 1e-9)
            weighted_feats = proj(node_feats) * attn_weights.unsqueeze(-1)

            graph_emb = torch_scatter.scatter_add(
                weighted_feats, batch, dim=0, dim_size=num_graphs
            )

            graph_embs.append(graph_emb)

        return torch.cat(graph_embs, dim=-1)


class ScatterAttentionPool(nn.Module):
    def __init__(self, in_dim, attn_hidden=32):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, node_feats, batch, num_graphs=None):
        if num_graphs is None:
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
        # (1) Compute unnormalized attention scores for each node
        attn_scores = self.attn_mlp(node_feats).squeeze(-1)  # [N]
        # (2) Compute per-graph softmax using scatter
        max_per_graph = torch_scatter.scatter_max(attn_scores, batch, dim=0, dim_size=num_graphs)[0]
        attn_scores = attn_scores - max_per_graph[batch]  # numerical stability
        exp_scores = torch.exp(attn_scores)
        sum_per_graph = torch_scatter.scatter_add(exp_scores, batch, dim=0, dim_size=num_graphs)
        attn_weights = exp_scores / (sum_per_graph[batch] + 1e-9)  # normalized per graph
        # (3) Weighted node sum
        weighted_feats = node_feats * attn_weights.unsqueeze(-1)
        graph_emb = torch_scatter.scatter_add(weighted_feats, batch, dim=0, dim_size=num_graphs)
        return graph_emb


def focus_protein_radius(pos, protein_lens, batch, r=15.0):
    device = pos.device
    focus_region = torch.zeros(pos.size(0), dtype=torch.bool, device=device)
    num_graphs = batch.max().item() + 1
    for i in range(num_graphs):
        node_mask = batch == i
        node_idx = node_mask.nonzero(as_tuple=False).view(-1)
        pos_i = pos[node_idx]
        protein_len = protein_lens[i]
        ligand_idx_local = torch.arange(protein_len, pos_i.size(0), device=device)
        if len(ligand_idx_local) == 0:
            continue
        pos_protein = pos_i[:protein_len]
        pos_ligand = pos_i[ligand_idx_local]
        ligand_center = pos_ligand.mean(dim=0, keepdim=True)  # (1,3)
        dists = torch.cdist(pos_protein, ligand_center).squeeze(-1)  # (protein_len,)
        pocket_mask_local = dists <= r
        pocket_idx_global = node_idx[:protein_len][pocket_mask_local]
        focus_region[pocket_idx_global] = True
    return focus_region


class ProLigCrossAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        inner_dim = self.dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.proj = nn.Linear(inner_dim, dim)

    def forward(self, pro_rep, pro_batch, lig_rep, lig_batch):
        Np, d = pro_rep.shape
        Nl = lig_rep.size(0)
        h = self.heads
        dh = self.dim_head

        Q = self.to_q(pro_rep).view(Np, h, dh)  # [Np, h, dh]
        K = self.to_k(lig_rep).view(Nl, h, dh)  # [Nl, h, dh]
        V = self.to_v(lig_rep).view(Nl, h, dh)  # [Nl, h, dh]

        att_logits = torch.einsum("n h d, m h d -> n h m", Q, K) / (dh ** 0.5)
        mask = (pro_batch[:, None] == lig_batch[None, :])  # [Np, Nl]
        att_logits = att_logits.masked_fill(~mask[:, None, :], float('-inf'))
        att = torch.softmax(att_logits, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        out_heads = torch.einsum("n h m, m h d -> n h d", att, V)

        out = out_heads.reshape(Np, h * dh)
        out = self.proj(out) + pro_rep  # [Np, d]
        return out


class AtomToResidueAggregator(nn.Module):
    def __init__(self, atom_dim, hidden_dim=196):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(atom_dim, atom_dim),
                                  nn.GELU(),
                                  nn.Linear(atom_dim, atom_dim))
        self.score_net = nn.Sequential(nn.Linear(atom_dim, atom_dim), nn.GELU(), nn.Linear(atom_dim, 1))

    def forward(self, atom_rep, atom_pos, atom_batch, residue_global_idx):
        """
        atom_rep: [N_atom, F]
        atom_pos: [N_atom, 3]
        atom_batch: [N_atom]
        residue_global_idx: [N_atom]
        """

        unique_residue_idx, res_inverse = torch.unique(
            residue_global_idx,
            return_inverse=True
        )
        num_res = unique_residue_idx.size(0)

        atom_rep_proj = self.proj(atom_rep)
        score = self.score_net(atom_rep_proj).squeeze(-1)  # [N_atom]
        alpha = scatter_softmax(score, res_inverse)  # [N_atom]

        res_rep = scatter_sum(
            atom_rep_proj * alpha.unsqueeze(-1),
            res_inverse,
            dim=0,
            dim_size=num_res
        )  # [N_res, F]

        res_pos = scatter_mean(
            atom_pos,
            res_inverse,
            dim=0,
            dim_size=num_res
        )  # [N_res, 3]

        res_batch = torch.zeros(
            num_res,
            dtype=atom_batch.dtype,
            device=atom_batch.device
        )
        res_batch.scatter_(0, res_inverse, atom_batch)

        return res_rep, res_pos, res_batch


import torch.nn.init as init


class REGNN(nn.Module):
    def __init__(self, traj_channel=128, edge_channels=1, hidden_channels=96, n_layers=3):
        super().__init__()
        self.embedding1 = nn.Embedding(23, traj_channel // 2)
        self.embedding2 = nn.Embedding(11, traj_channel // 2)
        self.gnn_layers = nn.ModuleList([
            EGNNLayer(traj_channel, edge_channels, traj_channel) for _ in range(n_layers)
        ])
        self.pocket_predictor = DualDeltaPredictor(hidden_channels=traj_channel)
        self.lig_res_predictor = DualDeltaPredictor(hidden_channels=traj_channel)
        self.affinity_prediction = AffinityPredictor(in_channels=traj_channel, hidden_channels=128)
        self.out = nn.Linear(traj_channel, 3)  # predict position delta
        self.LigandRigidPredictor = LigandRigidPredictor(traj_channel, traj_channel)
        self.hidden_size = traj_channel
        self.motion_scale = 5

        for p in self.parameters():
            p.requires_grad = False
        self.linear_dta1 = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 64)
        )
        self.linear_dta2 = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.5), # 0.5
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.embedding_dta2 = nn.Embedding(11, 60)
        self.norm = nn.LayerNorm(hidden_channels)
        # self.embedding_dta3 = nn.Embedding(92, hidden_channels // 3)
        self.time_embedding = nn.Embedding(num_embeddings=5, embedding_dim=hidden_channels)

        # self.embedding2 = nn.Embedding(23, hidden_channels // 2)
        self.atom2res = AtomToResidueAggregator(atom_dim=hidden_channels, hidden_dim=hidden_channels)
        self.FeatureRefine = Feature_Refine(n_layers, hidden_channels, 6)
        self.lig_transform = nn.Linear((hidden_channels // 2), hidden_channels, bias=False)
        self.feature_lig = Lig_feature(n_layers, hidden_channels, 32)
        self.attention = ProLigCrossAttention(dim=hidden_channels)
        self.feature_fusion = GatedFusion(hidden_channels)
        # self.PocketFeatureRefine = Feature_Refine(n_layers, hidden_channels, 6)
        self.PocketFeatureRefine = nn.ModuleList(
            [Feature_Refine(n_layers, hidden_channels, 10) for _ in range(5)])
        self.protein_readout = ScatterAttentionPool(in_dim=hidden_channels)
        self.lig_readout = ScatterAttentionPool(in_dim=hidden_channels)
        self.complex_readout = nn.ModuleList(
            [ScatterAttentionPool(in_dim=hidden_channels) for _ in range(5)])
        self.reshape = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.predictor = Predictor(hidden_channels)
        self.channel = hidden_channels

    def representation(self, atom_elements, atom_residue):
        x1 = self.embedding1(atom_residue)
        x2 = self.embedding2(atom_elements)
        x = torch.cat((x1, x2), dim=-1)
        return x

    def DTA_rep(self, atom_elements, atom_details, residue_embeds, mol_embeds, residue_global_idx, ligand_mask, batch):
        atom_len = len(atom_elements)
        Res_embed = torch.zeros(atom_len, 64, device=atom_elements.device)
        Res_embed[~ligand_mask] = self.linear_dta1(residue_embeds[residue_global_idx])
        Res_embed[ligand_mask] = self.linear_dta2(mol_embeds[batch[ligand_mask]])
        x1 = Res_embed
        x2 = self.embedding_dta2(atom_elements)
        x3 = atom_details  # self.embedding_dta3(atom_details)
        x = torch.cat((x1, x2, x3), dim=-1)  # , x2
        return x

    def DTA_fea(self, x, pos, edge_index, edge_attr, protein_len, batch, lig_edges, lig_edge_types, ligand_mask):
        pos_ori = pos
        node_rep = x
        delta_pos_total = 0

        for layer in self.gnn_layers:
            node_rep, delta_pos = layer(node_rep, pos, edge_index, edge_attr)
            delta_pos_total += delta_pos

        pocket_mask = dynamic_pocket_mask(pos, protein_len, batch)
        # print(len(pocket_mask))
        pocket_mask_wo_lig = pocket_mask & ~ligand_mask
        pocket_mask_wo_lig = pocket_mask_wo_lig.unsqueeze(1)
        rot_vec, t_pred = self.LigandRigidPredictor(node_rep, pos, batch, ligand_mask)
        DELTA_pocket_pos = self.pocket_predictor(node_rep, delta_pos_total, pocket_mask_wo_lig)
        x_sub, pos_sub, edge_index_sub, subgraph_batch = pocket_subgraph(node_rep, pos, batch, pocket_mask)
        Energy = self.affinity_prediction(x_sub, edge_index_sub, subgraph_batch)

        DELTA_lig_res_pos = self.lig_res_predictor(node_rep, delta_pos_total, ligand_mask.unsqueeze(1))
        pos_new = pos_ori + DELTA_pocket_pos

        pos_lig_new = apply_ligand_transform_batch(pos_new, ligand_mask, rot_vec, t_pred, DELTA_lig_res_pos, batch,
                                                   lig_edges, lig_edge_types)
        pos_new[ligand_mask] = pos_lig_new
        # print(f"Energy values: {Energy}")
        return pos_new.detach(), pocket_mask.detach(), Energy.detach(), node_rep.detach()

    def DTA_prediction(self, x, pocket_reps, protein_len, pos, pocket_poss, pocket_masks, ENEs, batch, lig_edges,
                       lig_edge_types, ligand_mask, residue_global_idx, Training=True):
        N = len(pocket_poss)
        dynamic_complexes = []
        node_rep = x.detach()  # self.feature_fusion(x, node_reps_pretrain)
        num_graphs = batch.max().item() + 1
        # protein_mask = focus_protein_radius(pos, protein_len, batch, r=15.0)
        pro_rep = node_rep[~ligand_mask]  # [:,0:96]
        pro_pos = pos[~ligand_mask]
        pro_batch = batch[~ligand_mask]
        # protein_mask = dynamic_pocket_mask_radius(pos, protein_len, batch, cutoff=12.0)
        # pro_rep = node_rep[protein_mask]
        # pro_pos = pos[protein_mask]
        # pro_batch = batch[protein_mask]
        res_rep, res_pos, res_batch = self.atom2res(pro_rep, pro_pos, pro_batch, residue_global_idx)
        res_ext, _, _, _ = self.FeatureRefine(res_rep, res_pos, res_batch)
        # pro_rep_ext, _, _, _ = self.FeatureRefine(pro_rep, pro_pos, pro_batch)
        protein_fea = self.protein_readout(res_ext, res_batch, num_graphs)

        # node_rep[~ligand_mask] = pro_rep_ext
        lig_rep = node_rep[ligand_mask]  # self.lig_transform(node_rep[ligand_mask][:,64:])# [:,48:]
        lig_batch = batch[ligand_mask]
        lig_rep_ext = self.feature_lig(lig_rep, lig_edges, lig_edge_types, batch, ligand_mask)
        # pro_lig_rela = self.attention(pro_rep_ext, pro_batch, lig_rep, lig_batch)
        lig_fea = self.lig_readout(lig_rep_ext, lig_batch, num_graphs)

        global_graph_fea = torch.cat([protein_fea, lig_fea], dim=-1)

        # node_rep[~ligand_mask] = pro_rep_ext
        # node_rep[ligand_mask] = lig_rep_ext
        # global_graph_fea = self.protein_readout(node_rep, batch, num_graphs)# weighted_scatter_mean_fast(node_rep, pos, pocket_poss, pocket_masks, batch, num_graphs)
        ene_list = torch.stack(ENEs, dim=0)

        for i in range(N):
            pocket_rep = x  # (pocket_reps[i] + x) / 2
            pos_pocket = pos.clone()
            pos_pocket[pocket_masks[i]] = pocket_poss[i]
            pocket_mask = dynamic_pocket_mask(pos, protein_len, batch, k=10)
            pocket_rep_input = pocket_rep[pocket_mask]
            pos_pocket_input = pos_pocket[pocket_mask]
            batch_input = batch[pocket_mask]
            pocket_node_fea, x_pos, all_st, edge_index = self.PocketFeatureRefine[i](pocket_rep_input, pos_pocket_input,
                                                                                     batch_input)
            dynamic_complexes.append(self.complex_readout[i](pocket_node_fea, batch_input, num_graphs))

        dynamic_complexes = torch.stack(dynamic_complexes, dim=0)  # .transpose(0, 1)  # [N, hidden_dim]
        # ene_list = torch.stack(ENEs, dim=0)  # [N, 1]
        aff, aff_out1, aff_out2 = self.predictor(global_graph_fea, dynamic_complexes, ene_list)
        return aff, aff_out1, aff_out2

    '''
    def DTA_prediction(self, x, node_reps_pretrain, pos, pocket_poss, pocket_masks, ENEs, batch, lig_edges, lig_edge_types, ligand_mask):
        N = len(pocket_poss)
        fea_list = []
        node_rep = self.feature_fusion(x, node_reps_pretrain)
        pro_rep = node_rep[~ligand_mask]
        pro_pos = pos[~ligand_mask]
        pro_batch = batch[~ligand_mask]
        pro_rep_ext, _, _, _ = self.FeatureRefine(pro_rep, pro_pos, pro_batch)
        node_rep[~ligand_mask] = pro_rep_ext
        lig_rep = node_rep[ligand_mask]
        lig_rep_ext = self.feature_lig(lig_rep, lig_edges, lig_edge_types, batch, ligand_mask)
        node_rep[ligand_mask] = lig_rep_ext
        num_graphs = batch.max().item() + 1
        for i in range(N):
            pocket_node_rep = node_rep[pocket_masks[i]]
            pocket_batchs = batch[pocket_masks[i]]
            pocket_node_fea, x_pos, all_st, edge_index = self.PocketFeatureRefine(pocket_node_rep, pocket_poss[i], pocket_batchs)
            graph_sum = torch_scatter.scatter_add(pocket_node_fea, pocket_batchs, dim=0, dim_size=num_graphs)
            fea_list.append(graph_sum)
        global_graph_fea = weighted_scatter_mean_fast(node_rep, pos, pocket_poss, pocket_masks, batch, num_graphs)
        global_graph_fea_expand = global_graph_fea.repeat(N, 1, 2)  # [N, hidden_dim]
        fea_list = torch.stack(fea_list, dim=0)  # [N, hidden_dim]
        ene_list = torch.stack(ENEs, dim=0)  # [N, 1]
        final_fea = torch.cat([fea_list, global_graph_fea_expand], dim=-1)  # [N, 2*hidden_dim]
        aff = self.predictor(final_fea, ene_list)
        return aff
    '''

    def forward(self, h, x, pos, edge_index, edge_attr, protein_len, batch, ligand_mask=None):
        pos_ori = pos
        node_rep = x
        delta_pos_total = 0
        for layer in self.gnn_layers:
            node_rep, delta_pos = layer(node_rep, pos, edge_index, edge_attr)
            delta_pos_total += delta_pos

        pocket_mask = dynamic_pocket_mask(pos, protein_len, batch)
        # print(pocket_mask.shape)
        pocket_mask_wo_lig = pocket_mask & ~ligand_mask
        pocket_mask_wo_lig = pocket_mask_wo_lig.unsqueeze(1)
        rot_vec, t_pred = self.LigandRigidPredictor(node_rep, pos, ligand_mask)

        DELTA_pocket_pos = self.pocket_predictor(node_rep, delta_pos_total, pocket_mask_wo_lig)
        DELTA_lig_res_pos = self.lig_res_predictor(node_rep, delta_pos_total, ligand_mask.unsqueeze(1))

        x_sub, pos_sub, edge_index_sub, subgraph_batch = pocket_subgraph(node_rep, pos, batch, pocket_mask)
        aff = self.affinity_prediction(x_sub, edge_index_sub, subgraph_batch)

        pos_new = pos_ori + DELTA_pocket_pos

        return h, pos_new, DELTA_pocket_pos, aff, pocket_mask_wo_lig, rot_vec, t_pred, DELTA_lig_res_pos


class AffinityP(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.pred = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fea, attn):
        B, T, D = fea.size()
        scores = self.attn(fea)  # [B, T, 1]
        scores = scores.squeeze(-1)  # [B, T]
        attn_weights = F.softmax(scores, dim=1)  # [B, T]
        fea_weighted = torch.sum(fea * attn_weights.unsqueeze(-1), dim=1)  # [B, dim]
        aff = self.pred(fea_weighted)  # [B, 1]
        return aff, attn_weights