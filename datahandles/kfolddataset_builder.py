from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import pandas as pd

from datahandles.datahandler import DataHandler
from datahandles.dataset_utils import balance_dataframe, sample_dataframe, make_kfolds

TEXT_COL_NAME = "text"
TARGET_COL_NAME = "label"

class KFoldDatasetBuilder():
    def __init__(self,
                 dataset_name: str,
                 data_root: str | Path,
                 col_permutations: dict[str, bool],
                 train_size: int,

                 n_folds: int,
                 n_shots: int, # n_queries will be remaining samples in training fold
                 max_n_features: int | None,
                 
                 balanced: dict[str, bool],
                 overlap_shots_queries: bool,
                 eval_shots_from_train: bool,
                 eval_shots_with_labels: bool,
                 ) -> None:
                
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.col_permutations = col_permutations

        self.train_size = train_size
        self.n_folds = n_folds
        self.n_shots = n_shots
        self.max_n_features = max_n_features
        self.balanced = balanced
        self.eval_shots_with_labels = eval_shots_with_labels
        self.overlap_shots_queries = overlap_shots_queries
        self.eval_shots_from_train = eval_shots_from_train
        
        self.datahandler = DataHandler(data_path=self.data_root / self.dataset_name)
        if self.max_n_features is None:
            self.max_n_features = self.datahandler.n_features_preproc
        
        # balance / sample training and testing data
        self.train_df = sample_dataframe(df=self.datahandler.train_df,
                                         target_col=self.datahandler.target_name,
                                         n_samples=self.train_size,
                                         method="balanced" if self.balanced["train"] else "stratified")
        if self.balanced["test"]:
            self.test_df = balance_dataframe(df=self.datahandler.test_df,
                                            target_col=self.datahandler.target_name,
                                            method="balanced")
        else:
            self.test_df = self.datahandler.test_df.copy()

        self.folds = make_kfolds(df=self.train_df, k=self.n_folds)
        dataset_folds = []
        for fold in self.folds:
            train_df, val_df = fold

            train_shots = train_df[:self.n_shots]
            train_queries = train_df if self.overlap_shots_queries else train_df[self.n_shots:]
            
            val_shots = train_df[:self.n_shots] if self.eval_shots_from_train else val_df[:self.n_shots]
            val_queries = val_df[self.n_shots:] if not self.overlap_shots_queries and not self.eval_shots_from_train else val_df 
            
            train_ds = self._build_fold_df(fold_shots=train_shots, 
                                           fold_queries=train_queries,
                                           handler=self.datahandler,
                                           max_n_features=self.max_n_features,
                                           col_permutation=self.col_permutations.get("train", False),
                                           shots_with_labels=True,)
            val_ds = self._build_fold_df(fold_shots=val_shots, 
                                         fold_queries=val_queries,
                                         handler=self.datahandler,
                                         max_n_features=self.max_n_features,
                                         col_permutation=self.col_permutations.get("val", False),
                                         shots_with_labels=self.eval_shots_with_labels,)
            dataset_folds.append({'train': train_ds, 'val': val_ds})

        if eval_shots_from_train:
            self.test_ds = self._build_fold_df(fold_shots=self.train_df[:self.n_shots], 
                                               fold_queries=self.test_df,
                                               handler=self.datahandler,
                                               max_n_features=self.max_n_features,
                                               col_permutation=self.col_permutations.get("test", False),
                                               shots_with_labels=self.eval_shots_with_labels,)
        else:
            self.test_ds = self._build_fold_df(fold_shots=self.test_df[:self.n_shots], 
                                               fold_queries=self.test_df,
                                               handler=self.datahandler,
                                               max_n_features=self.max_n_features,
                                               col_permutation=self.col_permutations.get("test", False),
                                               shots_with_labels=self.eval_shots_with_labels,)
        
        self.train_folds = dataset_folds
        self.test_ds = self.test_ds

    def get_datasets(self):
        return {
            "train": self.train_folds,
            "test": self.test_ds,
        }

    
    def _build_fold_df(self, 
                       fold_shots: pd.DataFrame, 
                       fold_queries: pd.DataFrame,
                       handler: DataHandler,
                       max_n_features: int,
                       col_permutation: bool,
                       shots_with_labels: bool,) -> Dataset:
        class FoldDataset(Dataset):
            def __init__(self) -> None:
                super().__init__()
                                
            def __len__(self):
                return 1
            
            def __getitem__(self, index):
                if type(index) is tuple:
                    index, permutation = index
                else:
                    index = index 
                    permutation = col_permutation

                if index != 0:
                    raise IndexError("FoldDataset only has one item at index 0")
                
                if type(permutation) is bool and permutation == True:
                    permutation = list(np.random.permutation(handler.n_features))
                elif permutation == False:
                    permutation = []

                # Shots = text+label strings
                shots_df = handler.apply_permutation(fold_shots, permutation)
                prompts = handler.apply_template(shots_df)
                if shots_with_labels:
                    shots = "".join([f"Example {i}: {shot[TEXT_COL_NAME]} {shot[TARGET_COL_NAME]}\n\n" for i, shot in prompts.iterrows()])
                else:
                    shots = "".join([f"Example {i}: {shot[TEXT_COL_NAME]}\n\n" for i, shot in prompts.iterrows()])
                
                # Queries
                query_df = handler.apply_permutation(fold_queries, permutation)
                query_df = handler.preprocess(query_df)
                queries_x = query_df.iloc[:, :-1].to_numpy(dtype=np.float32)
                queries_y = query_df.iloc[:, -1].to_numpy(dtype=np.int64)

                # Pad/truncate queries_x
                if queries_x.shape[1] <= max_n_features:
                    pad = np.zeros((queries_x.shape[0], max_n_features - queries_x.shape[1]), dtype=np.float32)
                    queries_x = np.concatenate([queries_x, pad], axis=1)
                else:
                    raise ValueError(f"Padding error: Max no. of features: {max_n_features}, available no. of features: {queries_x.shape[1]}")

                return {
                    "shots": shots,
                    "queries_x": torch.tensor(queries_x),
                    "queries_y": torch.tensor(queries_y),
                }
        return FoldDataset()
            
            

             



                

        


        

        

        
