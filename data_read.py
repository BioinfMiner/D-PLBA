import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
class H5ProtDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.keys = []
        self.transform = transform

        # 预读取所有样本 id（group 名称）
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        print(key)
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



if __name__ == "__main__":

    h5_path = 'F:/lesson8_dataset/val_set3.h5'
    dataset = H5ProtDataset(h5_path)

    print(f"🧬 Total samples in dataset: {len(dataset)}")

    # 遍历并打印每个样本信息
    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(f"\n🔹 Sample #{idx} - ID: {sample.get('id', 'N/A')}")
        for key, val in sample.items():

            if isinstance(val, torch.Tensor):
                print(f"  • {key}: Tensor, shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  • {key}: {val} ({type(val).__name__})")
        break
