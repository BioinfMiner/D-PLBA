import scipy.spatial as ss
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from scipy.spatial import cKDTree as KDTree
ligand_atoms_mapping = {8: 0, 16: 1, 6: 2, 7: 3, 1: 4, 15: 5, 17: 6, 9: 7, 53: 8, 35: 9, 5: 10, 33: 11, 26: 12, 14: 13, 34: 14, 44: 15, 12: 16, 23: 17, 77: 18, 27: 19, 52: 20, 30: 21, 4: 22, 45: 23}


class ProtDataset(Dataset):

    def __init__(self, md_data_file, qm_data_file, idx_file):

        self.md_data_file = Path(md_data_file).absolute()
        self.qm_data_file = Path(qm_data_file).absolute()

        with open(idx_file, 'r') as f:
            self.ids = f.read().splitlines()

        self.f = h5py.File(self.md_data_file, 'r')
        self.f2 = h5py.File(self.qm_data_file, 'r')


    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        try:
            if not 0 <= (index) < len(self.ids):
                raise IndexError(index)

            item = {}

            column_names = ["x", "y", "z", "element"]

            pitem = self.f[self.ids[index]]
            pitem2 = self.f2[self.ids[index]]

            # protein-ligand feature
            # print(len(pitem["atoms_element"][:][:]))
            item["pl_element"] = node_feat_transform(pitem["atoms_element"][:][:])

            # protein edge && protein-ligand trajectory pos
            cutoff = pitem["molecules_begin_atom_index"][:][-1]
            # mol_start = pitem["molecules_begin_atom_index"][:][-1]
            item["id"] = self.ids[index]
            trajectory = pitem["trajectory_coordinates"][:, :, :]
            protein_trajectory = pitem["trajectory_coordinates"][:, :cutoff, :]
            protein_indices = list(range(0, min(protein_trajectory.shape[0], 100), 10))
            item["protein_len"] = cutoff
            # ligand bond
            ligand_trajectory = pitem["trajectory_coordinates"][:, cutoff:, :]

            ligand = pitem2["atom_properties/atom_names"]
            bonds = pitem2["atom_properties/bonds"][:]
            # 获取氢原子的位置（atom name == b'1'）

            hydrogen_indices = {i for i, atom in enumerate(ligand) if atom == b'1'}
            # 获取非氢原子的索引
            non_h_indices = sorted(set(range(len(ligand))) - hydrogen_indices)
            # 建立 old_index -> new_index 的映射
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
            # 重新映射 bonds 中的 i, j，如果 i, j 都在非氢原子集合中才保留
            filtered_bonds = []
            for i, j, c in bonds:
                if i in index_map and j in index_map:
                    new_i = index_map[i]
                    new_j = index_map[j]
                    filtered_bonds.append((new_i, new_j, c))
            # 添加 cutoff 偏移
            filtered_bonds2 = [(i + cutoff, j + cutoff, t) for (i, j, t) in filtered_bonds]
            lig_edges = torch.tensor([[i for (i, j, t) in filtered_bonds2],
                                      [j for (i, j, t) in filtered_bonds2]], dtype=torch.long)
            lig_edge_weights = torch.tensor([t for (i, j, t) in filtered_bonds2], dtype=torch.float)
            ligand_cleaned = [ligand[i] for i in non_h_indices]
            # ligand = pitem2["atom_properties/atom_names"] 4UUB
            # bonds = pitem2["atom_properties/bonds"][:]
            # ligand_cleaned = [x for x in ligand if x != b'1']
            # hydrogen_indices = {i for i, atom in enumerate(ligand) if atom == b'1'}
            # filtered_bonds = [
            #     (i, j, c) for (i, j, c) in bonds
            #     if i not in hydrogen_indices and j not in hydrogen_indices
            # ] # remove H
            # filtered_bonds2 = [(i + cutoff, j + cutoff, t) for (i, j, t) in filtered_bonds]
            # lig_edges = torch.tensor([[i for (i, j, t) in filtered_bonds2], [j for (i, j, t) in filtered_bonds2]], dtype=torch.long)
            #
            # lig_edge_weights = torch.tensor([t for (i, j, t) in filtered_bonds2], dtype=torch.float)
            item["lig_edges"] = lig_edges
            item["lig_edge_weights"] = lig_edge_weights

            ligand_len = len(ligand_cleaned)
            # print(len(ligand_cleaned)+cutoff)
            # print(len(pitem["atoms_element"][:]))
            if cutoff + ligand_len != trajectory.shape[1]:
                print(f"[⚠skip wrong data] id = {self.ids[index]}")
                print(f"cutoff: {cutoff}, raw_ligand_len: {ligand_len}, total: {trajectory.shape[1]}")
                raise ValueError("cutoff + ligand_len != total atoms in trajectory")
            i = 0
            for frame_idx in protein_indices:
                # protein-ligand trajectory pos
                item["pl_trajectory_pos_frame_" + str(i)] = torch.FloatTensor(trajectory[frame_idx])
                # protein trajectory edge
                coords = protein_trajectory[frame_idx]  # shape: [num_atoms, 3]
                prot_df = pd.DataFrame(columns=column_names)
                prot_df[['x', 'y', 'z']] = coords
                prot_edges, prot_edge_weights = trajectory_edge(prot_df, edge_dist_cutoff=3.2)
                # torch.set_printoptions(threshold=float('inf'))
                # item["prot_trajectory_edge_index_frame_" + str(i)] = edges
                # item["prot_trajectory_edge_attr_frame_" + str(i)] = edge_weights
                # protein-ligand trajectory edge
                ligand_coords = ligand_trajectory[frame_idx]  # shape: [num_atoms, 3]
                lig_df = pd.DataFrame(columns=column_names)
                lig_df[['x', 'y', 'z']] = ligand_coords
                edges, edge_weights = protein_ligand_edge(prot_df, lig_df, edge_dist_cutoff=4.5)
                item["pl_trajectory_edge_index_frame_" + str(i)] = torch.cat([prot_edges, edges, lig_edges], dim=1)
                item["pl_trajectory_edge_attr_frame_" + str(i)] = torch.cat([prot_edge_weights, edge_weights, lig_edge_weights], dim=0)
                # labels rmsd
                # item["bSASA_frame_" + str(i)] = pitem["frames_bSASA"][frame_idx]
                energy = torch.tensor(pitem["frames_interaction_energy"][frame_idx])
                IE = -energy / 10
                item["IE_frame_" + str(i)] = IE
                i += 1
            return item

        except Exception as e:
            print(f"[‼ skip wrong data] index = {index}, id = {self.ids[index] if hasattr(self, 'ids') else 'N/A'}")
            print(f"Reasons: {e}")
            return self.__getitem__((index + 1) % len(self))


def node_feat_transform(df):
    node_feats = torch.LongTensor([e for e in df])
    return node_feats

atom_mapping = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'P', 6: 'S', 7: 'CL', 8: 'BR', 9: 'I', 10: 'UNK'}

def trajectory_edge(df, edge_dist_cutoff):
    try:
        node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
        kd_tree = ss.KDTree(node_pos)
        edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
        edges = torch.LongTensor(edge_tuples).t().contiguous()
        edges = to_undirected(edges)
    except:
        print(f"Problem with the data")
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)
    return edges, edge_weights

def protein_ligand_edge(protein_df, ligand_df, edge_dist_cutoff):
    try:
        ligand_pos = ligand_df[['x', 'y', 'z']].to_numpy()
        protein_pos = protein_df[['x', 'y', 'z']].to_numpy()
        ligand_offset = len(protein_pos)  # 配体索引在总图中的起点偏移量
        protein_tree = KDTree(protein_pos)

        neighbors = protein_tree.query_ball_point(ligand_pos, edge_dist_cutoff)

        edges = []
        for ligand_idx, prot_indices in enumerate(neighbors):
            ligand_global_idx = ligand_idx + ligand_offset  # 配体节点全局索引
            for prot_idx in prot_indices:
                # print((ligand_global_idx, prot_idx))
                edges.append((ligand_global_idx, prot_idx))  # ligand → protein
                edges.append((prot_idx, ligand_global_idx))  # protein → ligand

        edge_index = torch.LongTensor(edges).t().contiguous()
        all_pos = np.vstack([protein_pos, ligand_pos])

        edge_weights = torch.FloatTensor([
            1.0 / (np.linalg.norm(all_pos[i] - all_pos[j]) + 1e-5) for i, j in edges
        ])
        return edge_index, edge_weights

    except Exception as e:
        print(f"Edge construction failed: {e}")
        return None, None


class H5ProtDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.keys = []
        self.transform = transform

        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[key]

            item = {}

            for ds_key in grp.keys():
                data = grp[ds_key][()]
                if isinstance(data, bytes):
                    item[ds_key] = data.decode('utf-8')

                elif isinstance(data, str):
                    item[ds_key] = data

                elif isinstance(data, np.ndarray) and data.dtype == np.object_:
                    if all(isinstance(x, (bytes, str)) for x in data):
                        item[ds_key] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in data]
                    else:
                        raise TypeError(f"Unsupported object array in {ds_key}: {data}")

                else:
                    item[ds_key] = torch.as_tensor(data)

            for attr_key in grp.attrs:
                item[attr_key] = grp.attrs[attr_key]

            if self.transform:
                item = self.transform(item)

            return item