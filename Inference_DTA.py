import argparse
import logging
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Model_down import REGNN, dynamic_pocket_mask
from components.datasets import ProtDataset, trajectory_edge, protein_ligand_edge, H5ProtDataset
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np

def so3_exp_map(rot_vec):
    """
    so(3) (axis-angle)  R
    rot_vec: (...,3)
    return: (...,3,3)
    """
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

def apply_ligand_transform(pred_pos, ligand_mask, rot_vec, t_pred, delta_lig_res_pos, inf_lig_res=True):
    pos_lig = pred_pos[ligand_mask]
    R = so3_exp_map(rot_vec)
    pos_rot = pos_lig @ R.T + t_pred

    if inf_lig_res:
        pos_lig_new = pos_rot + delta_lig_res_pos[ligand_mask]
    else:
        pos_lig_new = pos_rot
    return pos_lig_new

def get_ligand_mask(protein_len, batch):
    device = batch.device
    ligand_mask = torch.zeros(batch.shape, dtype=torch.bool, device=device)
    num_graphs = protein_len.shape[0]
    for g in range(num_graphs):
        mask_g = batch == g
        nodes_g = mask_g.nonzero(as_tuple=True)[0]
        n_protein = protein_len[g].item()
        ligand_nodes = nodes_g[n_protein:]
        ligand_mask[ligand_nodes] = True
    return ligand_mask

def weighted_pocket_loss(delta_pos, delta_target, ligand_mask, N, ligand_weight=0.0):
    per_atom_loss = F.smooth_l1_loss(delta_pos, delta_target, reduction="none").sum(dim=-1)  # (N,)
    weights = torch.ones_like(per_atom_loss)
    weights[ligand_mask] = ligand_weight
    loss = (per_atom_loss * weights).sum() / (N + 1e-8)
    return loss

from torch_geometric.nn import radius_graph, knn_graph

def k_graph_with_inv_distance(x, k, batch):
    edge_index = knn_graph(x, k=k, batch=batch, loop=False, flow='source_to_target')  # [2, num_edges]
    src, dst = edge_index
    dist = torch.norm(x[src] - x[dst], dim=-1)  # [num_edges]
    edge_weight = 1.0 / (dist + 1e-5)  # [num_edges]
    return edge_index, edge_weight

def radius_graph_with_inv_distance(x, r, batch):
    edge_index = radius_graph(x, r=r, batch=batch, loop=False, flow='source_to_target')  # [2, num_edges]
    src, dst = edge_index
    dist = torch.norm(x[src] - x[dst], dim=-1)
    edge_weight = 1.0 / (dist + 1e-5)
    return edge_index, edge_weight

def average_pairwise_distance(pos):
    N = pos.shape[0]
    if N < 2:
        return 0.0, 0.0, 0.0

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N,N,3]
    dist_matrix = torch.norm(diff, dim=2)  # [N,N]

    dist_matrix.fill_diagonal_(float('inf'))
    nn_distances = dist_matrix.min(dim=1).values
    avg_nn_dist = nn_distances.mean().item()
    min_nn_dist = nn_distances.min().item()
    max_nn_dist = nn_distances.max().item()
    print(f"平均原子距离: {avg_nn_dist:.3f} Å, 最大距离: {min_nn_dist:.3f}, 最小距离: {max_nn_dist:.3f}")

def analyze_molecule_geometry(pos):
    N = pos.shape[0]
    center = pos.mean(dim=0)
    print(f"质心位置: {center.cpu().numpy()}")
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N,N,3]
    dist_matrix = torch.norm(diff, dim=2)
    dist_matrix.fill_diagonal_(float('inf'))
    nn_distances = dist_matrix.min(dim=1).values
    print(f"最近邻平均距离: {nn_distances.mean().item():.3f} Å")
    print(f"最近邻最小距离: {nn_distances.min().item():.3f} Å")
    print(f"最近邻最大距离: {nn_distances.max().item():.3f} Å")
    pos_centered = pos - center
    cov = pos_centered.T @ pos_centered / N
    eigenvalues, _ = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.flip(0)  # 从大到小排序
    print(f"PCA 主轴长度（方差平方根）: {eigenvalues.sqrt().cpu().numpy()}")


from metrics import get_sbap_regression_metric_dict

@torch.no_grad()
def test(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true_aff = []
    y_pred_aff = []
    result_dict = {}
    with torch.no_grad():
        for data in loader:
            # print(data[f"samples"])
            atom_element = data[f"atom_ids"].to(device)
            atom_details = data[f"atom_features"].to(device)
            atom_residue = data[f"residue_ids"].to(device)
            pos_ori = data[f"coords"].to(device)
            lig_edges = data[f"lig_edges"]
            lig_edge_types = data[f"lig_edge_weights"]
            residue_global_idx = data["residue_global_idx"].to(device)
            residue_embeds = data["residue_embeds"].to(device)
            mol_embeds = data["mol_embeds"].to(device)
            x = model.representation(atom_element, atom_residue)
            batch = data[f"batch"].to(device)
            label = data[f"label"].to(device)
            pos_t = pos_ori.clone()
            pocket_poss = []
            pocket_masks = []
            pocket_reps = []
            ENEs = []
            protein_len = data[f"protein_len"].to(device)
            ligand_mask = get_ligand_mask(protein_len, batch)
            for t in range(0, 6):  #
                edge_index_t, edge_attr_t = k_graph_with_inv_distance(pos_t, 6, batch)
                pred_pos, pocket_mask, Energy, node_rep = model.DTA_fea(x, pos_t, edge_index_t, edge_attr_t,
                                                                        protein_len, batch, lig_edges, lig_edge_types, ligand_mask)
                if t == 5:
                    ENEs.append(Energy)
                else:
                    pocket_poss.append(pred_pos[pocket_mask])
                    pocket_masks.append(pocket_mask)
                    pocket_reps.append(node_rep)
                    ENEs.append(Energy)
                pos_t = pred_pos
            # node_reps_pretrain = torch.mean(torch.stack(node_reps, dim=0), dim=0)
            x_dta = model.DTA_rep(atom_element, atom_details, residue_embeds, mol_embeds, residue_global_idx, ligand_mask, batch)
            aff, _, _ = model.DTA_prediction(x_dta, pocket_reps, protein_len, pos_ori, pocket_poss, pocket_masks, ENEs[1:], batch, lig_edges, lig_edge_types, ligand_mask, residue_global_idx, Training=False)
            # print(label) , aff_out1, aff_out2
            y_true_aff.append(label)
            y_pred_aff.append(aff)
            dataname = data[f"samples"]
            ENEs_ = torch.stack(ENEs, dim=1).detach().cpu().numpy()
            ENEs_ = ENEs_
            for name, ene in zip(dataname, ENEs_):
                result_dict[name] = ene
            # result_dict[dataname] = ENEs_

    y_true_aff = torch.cat(y_true_aff, dim=0).detach().cpu().numpy()
    y_pred_aff = torch.cat(y_pred_aff, dim=0).detach().cpu().numpy()
    metrics = get_sbap_regression_metric_dict(y_true_aff, y_pred_aff)
    return metrics, y_true_aff, y_pred_aff, result_dict

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def collate_multi_frame(batch):
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        list_of_items = [item[key] for item in batch]
        if key in ['mol_embeds'] + ['residue_ids'] + ['atom_ids'] + ['coords'] + ['main_chains'] + ['atom_features']:
            batch_out[key] = torch.cat(list_of_items, dim=0)
        elif key in ['residue_embeds'] + ['residue_indices']:
            batch_out[key] = torch.cat(list_of_items, dim=0)
            batch_out[key+"_list"] = list_of_items # torch.tensor(list_of_items, dtype=torch.float)
        elif key in ['label'] + [f'IE_frame_{i}' for i in range(10)]:
            batch_out[key] = torch.tensor(list_of_items, dtype=torch.float)
        elif key == 'id':
            batch_out[key] = list_of_items
        elif key in ["samples"]:
            batch_out[key] = list_of_items
        elif key in ['lig_edges']:
            new_edges = []
            for i, edge_index in enumerate(list_of_items):
                offset_edge = edge_index.clone()
                new_edges.append(offset_edge)
            batch_out[key] = new_edges

        elif key in ['lig_edge_weights']:
            batch_out[key] = list_of_items
        elif key == 'affinity':
            batch_out[key] = list_of_items
        elif key == 'protein_len':
            batch_out[key] = torch.tensor(list_of_items, dtype=torch.long)
        else:
            try:
                batch_out[key] = torch.stack(list_of_items)
            except Exception as e:
                print(f"Warning: could not stack key {key} due to {e}, keep as list")
                batch_out[key] = list_of_items
    if 'atom_ids' in batch_out:
        node_counts = [item['atom_ids'].size(0) for item in batch]
        batch_out['batch'] = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(node_counts)
        ])
    if "residue_indices_list" in batch_out and "residue_embeds_list" in batch_out:
        residue_global_idx = []
        offset = 0
        for i in range (len(batch_out["residue_indices_list"])):
            residue_global_idx.append(batch_out["residue_indices_list"][i]+offset)
            offset += len(batch_out["residue_embeds_list"][i])
        batch_out["residue_global_idx"] = torch.cat(residue_global_idx, dim=0)
    return batch_out


def aggregate_metrics(metrics_list):
    result_dict = {}
    if not metrics_list:
        return result_dict
    metric_names = metrics_list[0].keys()

    for name in metric_names:
        values = [m[name] for m in metrics_list]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        result_dict[name] = {"mean": mean, "std": std}
    return result_dict


def infer(args, device, log_dir, name, rep=None):
    log_file = os.path.join(log_dir, str(name) + '_infer.log')
    if not logging.getLogger('lba').handlers:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger('lba')

    test_dataset = H5ProtDataset(args.test_set)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multi_frame)

    model = REGNN(traj_channel=128, edge_channels=1, hidden_channels=args.hidden_dim, n_layers=3).to(device)

    # best_model_path = os.path.join(log_dir, name)
    checkpoint = torch.load(name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from {name}")

    model.eval()
    start_time = time.time()
    metrics, y_true, y_pred, result_dict = test(model, test_loader, device)
    elapsed = time.time() - start_time
    print(f"Test set1 metrics: {metrics}, time: {elapsed:.2f}s")
    logger.info(f"Test set1 metrics: {metrics}, time: {elapsed:.2f}s")

    return metrics, y_true, y_pred, result_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default="E:/lesson8_dataset/github_upload/dataset/OpenBind.h5") #OpenBind csar_2020_ESM3
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default="E:/lesson8_dataset/github_upload/result/DTA_test_82/")
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()
    import pickle

    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True
    import glob
    model_paths = glob.glob(os.path.join(log_dir, '**', '*_best_weights_rep{rep}.pt'))
    print(model_paths)
    metrics_list = []
    for model_path in model_paths:
        rep_str = model_path.split('rep')[-1].split('. pt')[0]
        metrics, y_true, y_pred, result_dict = infer(args, device, log_dir, name=model_path, rep=rep_str)
        metrics_list.append(metrics)
    agg_results = aggregate_metrics(metrics_list)
    print(agg_results)
