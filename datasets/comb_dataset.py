from torch.utils.data import Dataset
import numpy as np
from datasets.dataset_utils import datasets


class CombinedDataset(Dataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.debug = cfg.debug_datasets() or cfg.debug()

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")
        self.split = split

        # Initialize datasets based on configuration
        super(CombinedDataset, self).__init__()
        if list(datasets.keys()) != self.cfg.datasets.list_combine():
            raise ValueError(f"Invalid dataset list: {self.cfg.datasets.list_combine()}. Available datasets: {list(datasets.keys())}")   
        self.datasets = [datasets[k](self.cfg)[self.split] for k in self.cfg.datasets.list_combine()]    
            
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i in range(len(self.lengths)):
            if idx < sum(self.lengths[:i + 1]):
                dataset_idx = idx - sum(self.lengths[:i])
                if self.debug: print(f"Accessing dataset {i} at index {dataset_idx} for combined index {idx}")
                row = self.datasets[i].iloc[dataset_idx]

                x = row[:-1].to_numpy(dtype=np.float32)
                y = np.int64(row[-1])

                return {
                    "input": x,
                    "label": y
                }
        raise IndexError("Index out of range for combined dataset.")
