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
        col_permutation: bool,
        shuffle: bool,
        debug: bool,
        random_seed: int | bool,
        shots_with_labels: bool = True,
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
        self.shots_with_labels = shots_with_labels

        if type(random_seed) is int:
            self.random_seed = random_seed
        elif random_seed:
            self.random_seed = np.random.randint(low=1, high=1000)
        else:
            self.random_seed = 42

        if debug:
            print(f"Few-shot dataset random seed = {self.random_seed}")

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
            
            if self.shuffle:
                df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
                
            self.datasets[name] = df
            self.handlers[name] = handler
            self.max_features = max(self.max_features, handler.n_features_preproc)

            # Precompute class indices for balancing
            target_col = handler.target_name
            self.class_indices[name] = {}
            for label in df[target_col].unique():
                self.class_indices[name][label] = df.index[df[target_col] == label].tolist()

        self.class_ptrs = {
            name: {label: 0 for label in self.class_indices[name]}
            for name in dataset_names
        }

        # Decide pad dimension
        self.pad_to = max_n_features if max_n_features is not None else self.max_features
        
        # Build assignments
        self.assignments = []
        self.ds_counts = {}
        block_size = self.n_queries if self.queries_same_as_shots else (self.n_shots + self.n_queries)
        i = 0
        rng = np.random.RandomState(self.random_seed)
        while len(self.assignments) < self.split_size:
            ds_name = rng.choice(dataset_names)
            # keep track of the number of datapoints coming from each dataset, for debugging purposes
            if ds_name in self.ds_counts.keys():
                self.ds_counts[str(ds_name)] += block_size
            else:
                self.ds_counts[str(ds_name)] = block_size

            # check whether the dataset has enough examples
            if len(self.datasets[ds_name]) < (i+1)*block_size:
                self.ds_counts[str(ds_name)] -= block_size
                dataset_names.remove(ds_name)
                continue

            if balance_labels:
                self.assignments.append(
                    self._assign_balanced(ds_name, i)
                )
            else:
                self.assignments.append(
                    self._assign_sequential(ds_name, i)
                )
            
            i += 1

        if debug: 
            print(f"Split: {split}, number of data point: {self.ds_counts}, balanced: {self.balance_labels}")
            print(f"Total number of datapoints: {sum(self.ds_counts.values())}")


    def _assign_sequential(self, ds_name, i):
        start = (i * self.n_queries
                 if self.queries_same_as_shots
                 else i * (self.n_shots + self.n_queries)
        )
        shot_idx = list(range(start, start + self.n_shots))
        query_idx = (
            list(range(start, start + self.n_queries))
            if self.queries_same_as_shots
            else list(range(start + self.n_shots, start + self.n_shots + self.n_queries))
        )
        # print(f"Dataset: {ds_name}, number of examples: {len(self.datasets[ds_name])}, current end index: {query_idx[-1]}")
        return (ds_name, shot_idx, query_idx)

    # def _assign_balanced(self, ds_name, i):
    #     """Return balanced label-wise indices."""
    #     class_indices = self.class_indices[ds_name]
    #     labels = list(class_indices.keys())
    #     n_classes = len(labels)

    #     block_size = self.n_queries if self.queries_same_as_shots else (self.n_shots + self.n_queries)
    #     queries_per_class = max(1, self.n_queries // n_classes)
    #     shots_per_class = max(1, self.n_shots // n_classes)
    #     query_idx = []
    #     shot_idx = []
    #     for label in labels:
    #         indices = class_indices[label]
    #         shot_start = i * block_size
    #         shot_end = shot_start + shots_per_class
    #         query_start = shot_start if self.queries_same_as_shots else shot_end
    #         query_end = query_start + queries_per_class
    #         shot_idx.extend(indices[shot_start:shot_end])
    #         query_idx.extend(indices[query_start:query_end])

    #     # print(f"Dataset: {ds_name}, number of examples: {len(self.datasets[ds_name])}, current end index: {query_idx[-1]}")   
    #     return (ds_name, shot_idx, query_idx)

    def _assign_balanced(self, ds_name, i):
        class_indices = self.class_indices[ds_name]
        ptrs = self.class_ptrs[ds_name]
        labels = list(class_indices.keys())
        n_classes = len(labels)

        # Distribute remainder fairly
        def split_counts(total):
            base = total // n_classes
            rem = total % n_classes
            return [base + (j < rem) for j in range(n_classes)]

        shots_counts   = split_counts(self.n_shots)
        queries_counts = split_counts(self.n_queries)

        shot_idx, query_idx = [], []

        for j, label in enumerate(labels):
            indices = class_indices[label]
            ptr = ptrs[label]

            s_cnt = shots_counts[j]
            q_cnt = queries_counts[j]

            # Wrap-around if needed
            if ptr + s_cnt + q_cnt > len(indices):
                np.random.shuffle(indices)
                ptr = 0

            shot_idx.extend(indices[ptr : ptr + s_cnt])
            query_start = ptr if self.queries_same_as_shots else ptr + s_cnt
            query_idx.extend(indices[query_start : query_start + q_cnt])

            ptrs[label] = ptr + s_cnt + q_cnt

        return (ds_name, shot_idx, query_idx)

    def __len__(self):
        return self.split_size

    def __getitem__(self, index):
        if type(index) is tuple:
            index, permutation = index
        else:
            index = index
            permutation = self.col_permutation

        ds_name, shot_idx, query_idx = self.assignments[index]
        df = self.datasets[ds_name]
        handler = self.handlers[ds_name]

        if permutation == True:
            permutation = list(np.random.permutation(handler.n_features))
        elif permutation == False:
            permutation = []

        # Shots = text+label strings
        shots_df = df.iloc[shot_idx]
        if self.split == "train":
            shots_df = shots_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        shots_df = handler.apply_permutation(shots_df, permutation)
        prompts = handler.apply_template(shots_df)
        if self.shots_with_labels:
            shots = "".join([f"Example {i}: {shot[TEXT_COL_NAME]} {shot[TARGET_COL_NAME]}\n\n" for i, shot in prompts.iterrows()])
        else:
            shots = "".join([f"Example {i}: {shot[TEXT_COL_NAME]}\n\n" for i, shot in prompts.iterrows()])
        
        # Queries
        query_df = df.iloc[query_idx]
        if self.split == "train":
            query_df = query_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
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
    
    def set_pad_length(self, pad_length):
        self.pad_to = pad_length



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
        train_permutation: bool,
        val_permutation: bool,
        test_permutation: bool,
        train_balance: bool,
        val_balance: bool,
        test_balance: bool,
        n_shots: int,
        n_queries: int,
        max_n_features: int,
        shuffle: bool,
        queries_same_as_shots: bool,
        debug: bool,
        random_seed: int | bool,
        shots_with_labels: dict[str, bool] = {"train": True, "val": True, "test": False},
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
                debug=debug,
                shots_with_labels=shots_with_labels["train"],
                random_seed=random_seed
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
                debug=debug,
                shots_with_labels=shots_with_labels["val"],
                random_seed=random_seed
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
                debug=debug,
                shots_with_labels=shots_with_labels["test"],
                random_seed=random_seed
            ),
        }

        if max_n_features is None:
            # Set pad length to the max no. of features across datasets
            pad_length = max(
                [
                    self.datasets[split].max_features
                    for split in ["train", "val", "test"]
                ]
            )
            for split in ["train", "val", "test"]:
                self.datasets[split].set_pad_length(pad_length)
            self.max_n_features = pad_length
        else:
            self.max_n_features = max_n_features

    def get_datasets(self):
        return self.datasets
