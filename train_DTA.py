import sys
import argparse
import logging
import os
import time
import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Model_down import REGNN, dynamic_pocket_mask
import utils
from components.datasets import ProtDataset, trajectory_edge, protein_ligand_edge, H5ProtDataset
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from torch_scatter import scatter_max

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

import torch.nn as nn
class ResidualDecouplingLoss(nn.Module):
    def __init__(self, lambda_div=0.1, eps=1e-7, margin_tau=0.6):
        super().__init__()
        self.lambda_div = lambda_div
        self.eps = eps
        self.margin_tau = margin_tau

    def forward(self, out1, out2, target):
        criterion_sml = torch.nn.SmoothL1Loss()
        if out1.shape != out2.shape or out1.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: out1 {out1.shape}, out2 {out2.shape}, target {target.shape}"
            )

        loss_reg = criterion_sml(out1, target) + criterion_sml(out2, target)

        e1 = out1 - target
        e2 = out2 - target

        e1_norm = torch.norm(e1, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        e2_norm = torch.norm(e2, p=2, dim=-1, keepdim=True).clamp_min(self.eps)

        e1_hat = e1 / e1_norm
        e2_hat = e2 / e2_norm

        cos_sim = (e1_hat * e2_hat).sum(dim=-1)

        if self.margin_tau > 0:
            mean_err = 5e-3 * (e1_norm.squeeze(-1) + e2_norm.squeeze(-1))
            gate = (mean_err > self.margin_tau).float().detach()
        else:
            gate = 1.0

        loss_div = (gate * (cos_sim ** 2)).mean()

        loss = loss_reg + self.lambda_div * loss_div
        return  loss

def DTA_train(model, loader, optimizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    loss_all = 0.0
    criterion = torch.nn.MSELoss()
    criterion_RDL = ResidualDecouplingLoss()
    for data in loader:
        optimizer.zero_grad()
        atom_element = data[f"atom_ids"].to(device)
        atom_details = data[f"atom_features"].to(device)
        atom_residue = data[f"residue_ids"].to(device)
        pos_ori = data[f"coords"].to(device)
        # analyze_molecule_geometry(pos_ori)
        lig_edges = data[f"lig_edges"]
        lig_edge_types = data[f"lig_edge_weights"]
        residue_global_idx = data["residue_global_idx"].to(device)
        residue_embeds = data["residue_embeds"].to(device)
        mol_embeds = data["mol_embeds"].to(device)
        x = model.representation(atom_element, atom_residue)
        batch = data[f"batch"].to(device)
        label = data[f"label"].to(device)
        # edge_index, edge_attr = k_graph_with_inv_distance(pos_ori, 6, batch)
        pos_t = pos_ori.clone()
        pocket_poss = []
        pocket_masks = []
        pocket_reps = []
        ENEs = []
        protein_len = data[f"protein_len"].to(device)
        ligand_mask = get_ligand_mask(protein_len, batch)
        for t in range(0, 5):  #
            edge_index, edge_attr = k_graph_with_inv_distance(pos_t, 6, batch)
            pred_pos, pocket_mask, Energy, pocket_rep = model.DTA_fea(x, pos_t, edge_index, edge_attr, protein_len, batch, lig_edges, lig_edge_types, ligand_mask)
            if t == 5:
                ENEs.append(Energy)
            else:
                pocket_poss.append(pred_pos[pocket_mask])
                pocket_masks.append(pocket_mask)
                pocket_reps.append(pocket_rep)
                ENEs.append(Energy)
            pos_t = pred_pos

        # node_reps_pretrain = torch.mean(torch.stack(node_reps, dim=0), dim=0)
        x_dta = model.DTA_rep(atom_element, atom_details, residue_embeds, mol_embeds, residue_global_idx, ligand_mask, batch)
        aff, aff_out1, aff_out2 = model.DTA_prediction(x_dta, pocket_reps, protein_len, pos_ori, pocket_poss, pocket_masks, ENEs, batch, lig_edges, lig_edge_types, ligand_mask, residue_global_idx, Training=True)
        aff_loss = criterion_RDL(aff_out1, aff_out2, label)
        # print("dta_loss:", criterion(aff, label))
        # print("aff_loss:", aff_loss)
        loss = criterion(aff, label) + aff_loss
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all

from metrics import get_sbap_regression_metric_dict

@torch.no_grad()
def test(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true_aff = []
    y_pred_aff = []
    with torch.no_grad():
        for data in loader:
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
            # edge_index, edge_attr = k_graph_with_inv_distance(pos_ori, 6, batch)
            pocket_poss = []
            pocket_masks = []
            pocket_reps = []
            ENEs = []
            protein_len = data[f"protein_len"].to(device)
            ligand_mask = get_ligand_mask(protein_len, batch)
            for t in range(0, 5):  #
                edge_index, edge_attr = k_graph_with_inv_distance(pos_t, 6, batch)
                pred_pos, pocket_mask, Energy, pocket_rep = model.DTA_fea(x, pos_t, edge_index, edge_attr,
                                                                        protein_len, batch, lig_edges, lig_edge_types, ligand_mask)
                if t == 5:
                    ENEs.append(Energy)
                else:
                    pocket_poss.append(pred_pos[pocket_mask])
                    pocket_masks.append(pocket_mask)
                    pocket_reps.append(pocket_rep)
                    ENEs.append(Energy)
                pos_t = pred_pos
            # node_reps_pretrain = torch.mean(torch.stack(node_reps, dim=0), dim=0)
            x_dta = model.DTA_rep(atom_element, atom_details, residue_embeds, mol_embeds, residue_global_idx, ligand_mask, batch)
            aff, aff_out1, aff_out2 = model.DTA_prediction(x_dta, pocket_reps, protein_len, pos_ori, pocket_poss, pocket_masks, ENEs, batch, lig_edges, lig_edge_types, ligand_mask, residue_global_idx, Training=False)
            y_true_aff.append(label)
            y_pred_aff.append(aff)

    y_true_aff = torch.cat(y_true_aff, dim=0).detach().cpu().numpy()
    y_pred_aff = torch.cat(y_pred_aff, dim=0).detach().cpu().numpy()
    metrics = get_sbap_regression_metric_dict(y_true_aff, y_pred_aff)
    return metrics, y_true_aff, y_pred_aff

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
        # print(batch_out["residue_indices_list"])
        # print(batch_out["residue_embeds_list"])
        residue_global_idx = []
        offset = 0
        for i in range (len(batch_out["residue_indices_list"])):
            # print(max(batch_out["residue_indices_list"][i]))
            # print(len(batch_out["residue_embeds_list"][i]))
            residue_global_idx.append(batch_out["residue_indices_list"][i]+offset)
            offset += len(batch_out["residue_embeds_list"][i])
        batch_out["residue_global_idx"] = torch.cat(residue_global_idx, dim=0)
    return batch_out

def train(args, device, log_dir, name="DTA_0", rep=None, test_mode=False):
    log_file = os.path.join(log_dir, str(name) + '.log')
    if not logging.getLogger('lba').handlers:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger('lba')

    train_dataset = H5ProtDataset(args.train_set)
    val_dataset = H5ProtDataset(args.val_set)
    test_dataset = H5ProtDataset(args.test_set)
    test_dataset2 = H5ProtDataset(args.test_set2)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multi_frame)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multi_frame)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multi_frame)
    test_loader2 = DataLoader(test_dataset2, args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multi_frame)

    model = REGNN(traj_channel=128, edge_channels=1, hidden_channels=args.hidden_dim, n_layers=3).to(device)
    target_model_state_dict = model.state_dict()
    # source_model_state_dict = torch.load("E:/lesson8_dataset/github_upload/pretrain/pretrain_align_82.pt", weights_only=True, map_location=device)
    source_model_state_dict = torch.load(
        "E:/lesson8_dataset/github_upload/pretrain/pretrain_align_82.pt"
    )
    prefixes = [
        "embedding1", "embedding2",
        "gnn_layers",
        "pocket_predictor",
        "lig_res_predictor",
        "affinity_prediction",
        "out",
        "LigandRigidPredictor"
    ]
    partial_dict = {
        k: v for k, v in source_model_state_dict["model_state_dict"].items()
        if any(k.startswith(prefix) for prefix in prefixes)
    }
    target_model_state_dict.update(partial_dict)
    model.load_state_dict(target_model_state_dict)
    # model.load_state_dict(checkpoint["model_state_dict"])
    best_aff_mse = 999
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, min_lr=5e-7)
    for epoch in range(1, args.num_epochs + 1):
        print('Start training with epoch_' + str(epoch))
        start = time.time()
        # aff = DTA_eva(model, val_loader)
        # metrics, y_true_aff, y_pred_aff = test(model, val_loader, device)
        train_loss = DTA_train(model, train_loader, optimizer, device)
        if epoch > 5 and epoch % 5 == 0:
            metrics, y_true_aff, y_pred_aff = test(model, val_loader)
            aff_rmse = metrics['RMSE']
            scheduler.step(aff_rmse)
            if utils.is_main_process():
                if aff_rmse < best_aff_mse and epoch > 35:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                    }, os.path.join(log_dir, f'_'+str(name)+'_best_weights_rep{rep}.pt'))
                    best_aff_rmse = aff_rmse
            elapsed = (time.time() - start)
            print('\t Train_loss: {:.7f}, aff_mse: {:.7f}, time: {:.7f}'.format(
                train_loss, aff_rmse, elapsed))
            logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(
                epoch, elapsed, train_loss, aff_rmse))
        else:
            elapsed = (time.time() - start)
            print('\t Train_loss: {:.7f}, time: {:.7f}'.format(train_loss, elapsed))
            logger.info('{:03d}\t{:.7f}\t{:.7f}\t No validation\n'.format(
                epoch, elapsed, train_loss))
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\t Current learning rate: {current_lr:.8f}")
        logger.info('{:.7f}\t Current learning rate\n'.format(current_lr))

    if test_mode:
        test_file = os.path.join(log_dir, f'lba-rep.best.test.pt')
        cpt = torch.load(os.path.join(log_dir, f'_'+str(name)+'_best_weights_rep{rep}.pt'))
        model.load_state_dict(cpt['model_state_dict'])
        metrics1, y_true_aff, y_pred_aff = test(model, test_loader)
        metrics2, y_true_aff2, y_pred_aff2 = test(model, test_loader2)
        aff_rmse = metrics1['RMSE']
        torch.save({'targets': y_true_aff, 'predictions': y_pred_aff}, test_file)
        print(f'\tTest Test aff_rmse {aff_rmse}')

    return best_aff_rmse, metrics1, metrics2

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', type=str, default="E:/lesson8_dataset/github_upload/dataset/train.h5")
    parser.add_argument('--val_set', type=str, default="E:/lesson8_dataset/github_upload/dataset/valid.h5")
    parser.add_argument('--test_set', type=str, default="E:/lesson8_dataset/github_upload/dataset/test.h5")
    parser.add_argument('--test_set2', type=str,default="E:/lesson8_dataset/github_upload/dataset/csar_2020.h5")
    parser.add_argument('--master_port', type=int, default=12354)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--log_dir', type=str, default="E:/lesson8_dataset/github_upload/result/DTA_test_85/")
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if utils.is_main_process():
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        train(args, device, log_dir)

    elif args.mode == 'test':
        metrics_list1 = []
        metrics_list2 = []
        timseline = []
        for rep, seed in enumerate(np.random.randint(0, 100, size=1)):
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dirs = os.path.join(log_dir, f'MD_{now}')
            if utils.is_main_process():
                if not os.path.exists(log_dirs):
                    os.makedirs(log_dirs)
            np.random.seed(seed)
            torch.manual_seed(seed)
            best_aff_rmse, metrics1, metrics2 = train(args, device, log_dirs, rep, test_mode=True)
            # if metrics1['RMSE'] < 1.18 and len(timseline) < 6:
            metrics_list1.append(metrics1)
            metrics_list2.append(metrics2)
            # timseline.append(now)
            # if len(timseline) >= 6:
            #     break
        agg_results1 = aggregate_metrics(metrics_list1)
        agg_results2 = aggregate_metrics(metrics_list2)
        print(agg_results1)
        print(agg_results2)
        import pandas as pd
        df1 = pd.DataFrame(agg_results1).T
        df1["Dataset"] = "Test1_pdbbind2020"
        df2 = pd.DataFrame(agg_results2).T
        df2["Dataset"] = "Test2_csar"
        df = pd.concat([df1, df2])
        df.to_csv("/home/yh/lesson8/Downstream/output/results_test_85.csv")


# nohup python train_DTA.py > output.log 2>&1 &