from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import numpy as np

from datahandles.datahandler import DataHandler

TEXT_COL_NAME = "text"
TARGET_COL_NAME = "label"


class FewShotDataset(Dataset):
    def __init__(
        self,
        dataset_names: list[str],
        data_root: str,
        split: str,
        split_size: int,
        n_shots: int,
        n_queries: int,
        queries_same_as_shots: bool,
        max_n_features: int,
        balance_labels: bool,
        col_permutation: list,
        shuffle: bool,
        debug: bool
    ):
        super().__init__()
        self.dataset_names = dataset_names
        self.split = split
        self.split_size = split_size
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.queries_same_as_shots = queries_same_as_shots
        self.balance_labels = balance_labels
        self.data_root = Path(data_root)
        self.shuffle = shuffle

        self.col_permutation = col_permutation

        # Load all datasets
        self.datasets = {}
        self.handlers = {}
        self.prompts = {}
        self.max_features = 0
        self.class_indices = {}

        for name in dataset_names:
            handler = DataHandler(self.data_root / name)
            if self.split == "test":
                df = handler.test_df
            elif self.split in ["train", "val"]:
                df = handler.train_df
                if self.split == "val":
                    df = df[::-1].reset_index(drop=True) # reverse the order, so that validation examples are chosen from the bottom of the dataframe
            else:
                raise ValueError(f"Split should be one of train, val or test. Received {self.split}")
                
            self.datasets[name] = df
            self.handlers[name] = handler
            self.max_features = max(self.max_features, handler.n_features_preproc)

            # Precompute class indices for balancing
            target_col = handler.target_name
            self.class_indices[name] = {}
            for label in df[target_col].unique():
                self.class_indices[name][label] = df.index[df[target_col] == label].tolist()

        # Decide pad dimension
        self.pad_to = max_n_features if max_n_features is not None else self.max_features

        # Build assignments
        self.assignments = []
        self.ds_counts = {}
        rng = np.random.RandomState(42)
        for i in range(split_size):
            ds_name = rng.choice(dataset_names)
            # keep track of the number of datapoints coming from each dataset, for debugging purposes
            if ds_name in self.ds_counts.keys():
                self.ds_counts[str(ds_name)] += 1
            else:
                self.ds_counts[str(ds_name)] = 1

            if balance_labels:
                self.assignments.append(
                    self._assign_balanced(ds_name, i)
                )
            else:
                self.assignments.append(
                    self._assign_sequential(ds_name, i)
                )

        if debug: 
            print(f"Split: {split}, number of data point: {self.ds_counts}, balanced: {self.balance_labels}")
            print(f"Total number of datapoints: {sum(self.ds_counts.values())}")


    def _assign_sequential(self, ds_name, i):
        df = self.datasets[ds_name]
        start = (i * (self.n_shots + self.n_queries)) % len(df)
        shot_idx = list(range(start, start + self.n_shots))
        query_idx = (
            shot_idx
            if self.queries_same_as_shots
            else list(range(start + self.n_shots, start + self.n_shots + self.n_queries))
        )
        shot_idx = [idx % len(df) for idx in shot_idx]
        query_idx = [idx % len(df) for idx in query_idx]
        return (ds_name, shot_idx, query_idx)

    def _assign_balanced(self, ds_name, i):
        """Return balanced label-wise indices."""
        class_indices = self.class_indices[ds_name]
        labels = list(class_indices.keys())
        n_classes = len(labels)

        # Shots: distribute equally across labels
        shots_per_class = max(1, self.n_shots // n_classes)
        shot_idx = []
        for j, label in enumerate(labels):
            indices = class_indices[label]
            for k in range(shots_per_class):
                idx = (i * shots_per_class + k) % len(indices)
                shot_idx.append(indices[idx])
        # Pad in case of remainder
        while len(shot_idx) < self.n_shots:
            shot_idx.append(class_indices[labels[0]][i % len(class_indices[labels[0]])])

        # Queries
        if self.queries_same_as_shots:
            query_idx = shot_idx.copy()
        else:
            queries_per_class = max(1, self.n_queries // n_classes)
            query_idx = []
            for j, label in enumerate(labels):
                indices = class_indices[label]
                for k in range(queries_per_class):
                    idx = (i * queries_per_class + k + 1000) % len(indices)  # offset so queries differ
                    query_idx.append(indices[idx])
            while len(query_idx) < self.n_queries:
                query_idx.append(class_indices[labels[0]][(i + 500) % len(class_indices[labels[0]])])

        return (ds_name, shot_idx, query_idx)

    def __len__(self):
        return self.split_size

    def __getitem__(self, index):
        print(index)
        index, permutation = index

        ds_name, shot_idx, query_idx = self.assignments[index]
        df = self.datasets[ds_name]
        handler = self.handlers[ds_name]

        if permutation == True:
            permutation = list(np.random.permutation(handler.n_features))
        elif permutation == False:
            permutation = []

        # Shots = text+label strings
        shots_df = df.iloc[shot_idx]
        shots_df = handler.apply_permutation(shots_df, permutation)
        prompts = handler.apply_template(shots_df)
        shots = "".join([f"Example {i}: {shot[TEXT_COL_NAME]} {shot[TARGET_COL_NAME]}\n\n" for i, shot in prompts.iterrows()])
        
        # Queries
        query_df = df.iloc[query_idx]
        query_df = handler.apply_permutation(query_df, permutation)
        query_df = handler.preprocess(query_df)
        queries_x = query_df.iloc[:, :-1].to_numpy(dtype=np.float32)
        queries_y = query_df.iloc[:, -1].to_numpy(dtype=np.int64)

        # Pad/truncate queries_x
        if queries_x.shape[1] <= self.pad_to:
            pad = np.zeros((queries_x.shape[0], self.pad_to - queries_x.shape[1]), dtype=np.float32)
            queries_x = np.concatenate([queries_x, pad], axis=1)
        else:
            raise ValueError(f"Padding error: Dataset: {ds_name}, max no. of features: {self.pad_to}, available no. of features: {queries_x.shape[1]}")

        return {
            "dataset": ds_name,
            "shots": shots,
            "queries_x": torch.tensor(queries_x),
            "queries_y": torch.tensor(queries_y),
        }



class MetaDatasetBuilder:
    """
    Builds train/val/test datasets of FewShotDataset objects.
    """

    def __init__(
        self,
        data_root: str,
        train_datasets: list[str],
        val_datasets: list[str],
        test_datasets: list[str],
        train_size: int,
        val_size: int,
        test_size: int,
        train_permutation: list,
        val_permutation: list,
        test_permutation: list,
        train_balance: bool,
        val_balance: bool,
        test_balance: bool,
        n_shots: int,
        n_queries: int,
        max_n_features: int,
        shuffle: bool,
        queries_same_as_shots: bool,
        debug: bool,
    ):
        self.datasets = {
            "train": FewShotDataset(
                dataset_names=train_datasets,
                data_root=data_root,
                split="train",
                split_size=train_size,
                n_shots=n_shots,
                n_queries=n_queries,
                max_n_features=max_n_features,
                col_permutation=train_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                balance_labels=train_balance,
                debug=debug
            ),
            "val": FewShotDataset(
                dataset_names=val_datasets,
                data_root=data_root,
                split="val",
                split_size=val_size,
                n_shots=n_shots,
                n_queries=n_queries,
                col_permutation=val_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                max_n_features=max_n_features,
                balance_labels=val_balance,
                debug=debug
            ),
            "test": FewShotDataset(
                dataset_names=test_datasets,
                data_root=data_root,
                split="test",
                split_size=test_size,
                n_shots=n_shots,
                n_queries=n_queries,
                col_permutation=test_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                max_n_features=max_n_features,
                balance_labels=test_balance,
                debug=debug
            ),
        }

    def get_datasets(self):
        return self.datasets
