from torch.utils.data import Dataset, DataLoader
import numpy as np
from datahandles.datahandler import DataHandler
from tabllm import load_and_preprocess_dataset, balance_dataset
from datasets import load_from_disk
from pathlib import Path
from datahandles import CombinedDataset, CombinedTextDataset, FewshotDataset
from utils import Config

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

TEXT_COL_NAME = "text"
TARGET_COL_NAME = "label"

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# class FewShotDataset(Dataset):
#     """
#     Few-shot dataset that samples tasks from multiple datasets
#     using DataHandler. Supports separate train/val/test splits,
#     where validation comes from the held-out part of train.csv.
#     """

#     def __init__(
#         self,
#         dataset_names: list[str],
#         data_root: str,
#         split: str,
#         split_size: int,
#         n_shots: int,
#         n_queries: int,
#         max_n_features: int,
#         col_permutation: list,
#         shuffle: bool,
#         queries_same_as_shots: bool,
#         val_fraction: float = 0.2,
#     ):
#         """
#         Parameters
#         ----------
#         dataset_names : list[str]
#             Names of datasets available in data_root/[dataset_name].
#         data_root : str
#             Root directory containing dataset folders.
#         split : str
#             One of 'train', 'val', 'test'.
#         split_size : int
#             Number of items (few-shot tasks) to generate for this split.
#         n_shots : int
#             Number of support examples in each task.
#         n_queries : int
#             Number of query examples in each task.
#         queries_same_as_shots : bool
#             If True, queries_x/queries_y are the same rows as shots.
#         val_fraction : float
#             Fraction of train.csv reserved for validation (only used if split='val').
#         max_n_features : int | None
#             If None, queries_x padded to maximum feature count across datasets.
#             If int, pad queries_x to that many features.
#         """
#         super().__init__()
#         self.dataset_names = dataset_names
#         self.data_root = Path(data_root)
#         self.split = split
#         self.split_size = split_size
#         self.n_shots = n_shots
#         self.n_queries = n_queries
#         self.queries_same_as_shots = queries_same_as_shots
#         self.val_fraction = val_fraction
#         self.max_n_features = max_n_features
#         self.col_permutation = col_permutation
#         self.shuffle = shuffle

#         # Pre-load DataHandlers for all datasets in this split
#         self.handlers = {}
#         self.feature_counts = {}
#         for name in self.dataset_names:
#             if split in ["train", "val"]:
#                 handler = DataHandler(self.data_root / name, "train")
#                 df, prompts = handler.get_split(col_permutation=self.col_permutation,
#                                                 shuffle=self.shuffle, 
#                                                 preprocess=True)

#                 # Partition train into train/val subsets
#                 n_val = int(len(df) * val_fraction)
#                 if split == "train":
#                     handler.df = df.iloc[:-n_val].reset_index(drop=True)
#                     handler.prompts = prompts.iloc[:-n_val].reset_index(drop=True)
#                 else:  # split == "val"
#                     handler.df = df.iloc[-n_val:].reset_index(drop=True)
#                     handler.prompts = prompts.iloc[-n_val:].reset_index(drop=True)

#                 self.handlers[name] = handler
#                 self.feature_counts[name] = handler.df.shape[1] - 1  # exclude target col
#             else:  # split == "test"
#                 handler = DataHandler(self.data_root / name, "test")
#                 df, prompts = handler.get_split(col_permutation=self.col_permutation,
#                                                 shuffle=self.shuffle, 
#                                                 preprocess=True)
#                 handler.df = df
#                 handler.prompts = prompts
#                 self.handlers[name] = handler
#                 self.feature_counts[name] = handler.df.shape[1] - 1

#         # Decide padding length
#         self.pad_to = self.max_n_features

#         # Assign datasets to items
#         self.max_per_dataset = max(1, split_size // len(dataset_names))
#         self.assignments = []
#         for name in dataset_names:
#             self.assignments.extend([name] * self.max_per_dataset)
#         random.shuffle(self.assignments)
#         self.assignments = self.assignments[:split_size]

#     def __len__(self):
#         return self.split_size

#     def __getitem__(self, idx):
#         dataset_name = self.assignments[idx]
#         handler = self.handlers[dataset_name]

#         df = handler.df
#         prompts = handler.prompts

#         # Combine prompt + label into one string for shots
#         combined_prompts = pd.Series(
#             [f"Example {i}: {prompts.iloc[i]['text']} {prompts.iloc[i]['label']}\n\n" for i in range(len(df))]
#         )

#         if len(df) < max(self.n_shots, self.n_queries):
#             raise ValueError(
#                 f"Dataset {dataset_name} too small "
#                 f"(needed {max(self.n_shots, self.n_queries)}, got {len(df)})"
#             )

#         chosen_idx = random.sample(range(len(df)), self.n_shots)
#         if self.queries_same_as_shots:
#             query_idx = chosen_idx
#         else:
#             query_idx = random.sample(range(len(df)), self.n_queries)

#         shots = combined_prompts.iloc[chosen_idx].tolist()
#         separator = ''
#         shots = separator.join(shots)

#         # queries_x = features (padded), not text
#         feature_cols = [c for c in df.columns if c != handler.target_name]
#         queries_x_raw = df.iloc[query_idx][feature_cols].to_numpy(dtype=np.float32)
#         queries_y = df[handler.target_name].iloc[query_idx].tolist()

#         # Pad queries_x
#         n_features = queries_x_raw.shape[1]
#         if n_features < self.pad_to:
#             pad_width = self.pad_to - n_features
#             queries_x = np.pad(queries_x_raw, ((0, 0), (0, pad_width)))
#         else:
#             queries_x = queries_x_raw[:, : self.pad_to]

#         return {
#             "dataset": dataset_name,
#             "shots": shots,                 # list[str] (prompt+label)
#             "queries_x": torch.tensor(queries_x, dtype=torch.float32),
#             "queries_y": torch.tensor(queries_y, dtype=torch.long),
#         }


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
        balance_labels: bool = True,
        col_permutation: list = [],
        shuffle: bool = True
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

        # Load all datasets
        self.datasets = {}
        self.prompts = {}
        self.max_features = 0
        self.class_indices = {}

        for name in dataset_names:
            if split == "val":
                handler = DataHandler(self.data_root / name, split="train")
                full_df, full_prompts = handler.get_split(shuffle=False)
                # Use only the leftover rows as validation
                df = full_df.iloc[split_size:].reset_index(drop=True)
                prompts = full_prompts.iloc[split_size:].reset_index(drop=True)
            else:
                handler = DataHandler(self.data_root / name, split=split)
                df, prompts = handler.get_split(shuffle=False)

            self.datasets[name] = df
            self.prompts[name] = prompts
            self.max_features = max(self.max_features, df.shape[1])

            # Precompute class indices for balancing
            target_col = handler.target_name
            self.class_indices[name] = {}
            for label in df[target_col].unique():
                self.class_indices[name][label] = df.index[df[target_col] == label].tolist()

        # Decide pad dimension
        self.pad_to = max_n_features if max_n_features is not None else self.max_features

        # Build assignments
        self.assignments = []
        rng = np.random.RandomState(42)
        for i in range(split_size):
            ds_name = rng.choice(dataset_names)
            if balance_labels:
                self.assignments.append(
                    self._assign_balanced(ds_name, i)
                )
            else:
                self.assignments.append(
                    self._assign_sequential(ds_name, i)
                )

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
        ds_name, shot_idx, query_idx = self.assignments[index]
        df = self.datasets[ds_name]
        prompts = self.prompts[ds_name]

        # Shots = text+label strings
        shots = [prompts.iloc[i] for i in shot_idx]
        shots = "".join([f"Example: {shot['text']} {shot['label']}\n\n" for shot in shots])

        # Queries
        queries_x = df.iloc[query_idx, :-1].to_numpy(dtype=np.float32)
        queries_y = df.iloc[query_idx, -1].to_numpy(dtype=np.int64)

        # Pad/truncate queries_x
        if queries_x.shape[1] < self.pad_to:
            pad = np.zeros((queries_x.shape[0], self.pad_to - queries_x.shape[1]), dtype=np.float32)
            queries_x = np.concatenate([queries_x, pad], axis=1)
        else:
            queries_x = queries_x[:, :self.pad_to]

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
        n_shots: int,
        n_queries: int,
        max_n_features: int,
        col_permutation: list,
        shuffle: bool,
        queries_same_as_shots: bool = False,
        val_fraction: float = 0.2,
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
                col_permutation=col_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                # val_fraction=val_fraction,
            ),
            "val": FewShotDataset(
                dataset_names=val_datasets,
                data_root=data_root,
                split="val",
                split_size=val_size,
                n_shots=n_shots,
                n_queries=n_queries,
                col_permutation=col_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                # val_fraction=val_fraction,
                max_n_features=max_n_features,
            ),
            "test": FewShotDataset(
                dataset_names=test_datasets,
                data_root=data_root,
                split="test",
                split_size=test_size,
                n_shots=n_shots,
                n_queries=n_queries,
                col_permutation=col_permutation,
                shuffle=shuffle,
                queries_same_as_shots=queries_same_as_shots,
                max_n_features=max_n_features,
            ),
        }

    def get_datasets(self):
        return self.datasets
