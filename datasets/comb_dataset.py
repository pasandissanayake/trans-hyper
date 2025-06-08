from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets.dataset_utils import datasets


class CombinedDataset(Dataset):
    def __init__(self, cfg, split: str):
        super(CombinedDataset, self).__init__()

        self.cfg = cfg
        self.debug = cfg.debug_datasets() or cfg.debug()

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")
        self.split = split

        if self.split == 'train':
            self.dataset_list = self.cfg.datasets.list_combine_train()
        elif self.split == 'val':
            self.dataset_list = self.cfg.datasets.list_combine_val()
        elif self.split == 'test':
            self.dataset_list = self.cfg.datasets.list_combine_test()
        else:
            self.dataset_list = []

        # Initialize datasets based on configuration
        if not set(self.dataset_list).issubset(list(datasets.keys())):
            raise ValueError(f"Invalid dataset list: {self.dataset_list}. Available datasets: {list(datasets.keys())}")   
        self.datasets = [datasets[k](self.cfg) for k in self.dataset_list]  
        self.datapoints = [dataset[self.split] for dataset in self.datasets]
            
        self.lengths = [len(ds) for ds in self.datapoints]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        i, ds_idx = self._get_dataset_idx(idx)
        row = self.datapoints[i].iloc[ds_idx]

        x = row[:-1].to_numpy(dtype=np.float32)
        y = np.int64(row.iloc[-1])

        return {
            "input": x,
            "label": y
        }
    
    def _get_dataset_idx(self, idx):
        """
        Helper function to get the dataset index for a given combined index.
        """
        for i in range(len(self.lengths)):
            if idx < sum(self.lengths[:i + 1]):
                return i, idx - sum(self.lengths[:i])
        raise IndexError("Index out of range for combined dataset.")


class CombinedTextDataset(CombinedDataset):
    def __init__(self, cfg, split):
        super(CombinedTextDataset, self).__init__(cfg, split)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            idx = index[0]
            get_text = index[1]
        else:
            idx = index
            get_text = False
        item = super(CombinedTextDataset, self).__getitem__(idx)
        if get_text:
            i, ds_idx = self._get_dataset_idx(idx)
            template = self.datasets[i].dataset_cfg.str_template()  # Get the text template for this dataset
            text = template.format(*item['input'], item['label'])
            return text
        else:
            return item  # Return the raw input and label without text


class FewshotDataset(Dataset):
    def __init__(self, cfg, split: str, n_shots: int, n_queries: int):
        super(FewshotDataset, self).__init__()

        self.cfg = cfg
        self.split = split
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.block_len = self.n_shots + self.n_queries
        

        self.combined_dataset = CombinedTextDataset(cfg, split)
        self.combds_length = len(self.combined_dataset)
        if self.combds_length < self.block_len:
            raise ValueError(f"Not enough samples in combined dataset for few-shot learning. Required: {n_shots}, Available: {self.combds_length}")

        # Randomly sample from the combined dataset
        self.length = self.combds_length // self.block_len
        indices = np.random.choice(self.combds_length, self.combds_length, replace=False)
        self.grouped_indices = self._group_items(data=indices, n_items=self.block_len, drop_last=True)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        group = {
            'shots': [self.combined_dataset[i, True] for i in self.grouped_indices[idx][:self.n_shots]],
            'queries': [self.combined_dataset[i, False] for i in self.grouped_indices[idx][self.n_shots:]]
        }
        return group
    
    def _group_items(self, data, n_items, drop_last):
        """
        Groups a list of items into sublists of size n_items.
        
        Args:
            data (List[Any]): The list to group.
            n_items (int): Size of each group.
            drop_last (bool): If True, discard the final group if it's smaller than n_items.
                            If False, include the final smaller group.
        
        Returns:
            List[List[Any]]: A list of grouped sublists.
        """
        grouped = [data[i:i + n_items] for i in range(0, len(data), n_items)]
        
        if drop_last and grouped and len(grouped[-1]) < n_items:
            grouped.pop()  # Remove the last incomplete group

        return grouped