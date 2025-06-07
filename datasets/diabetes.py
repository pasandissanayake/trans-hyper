import os
import pandas as pd

from sklearn.datasets import load_diabetes

from .dataset_utils import *

DATASET_NAME = 'diabetes'

@register(DATASET_NAME)
class DiabetesDataset:
    def __init__(self, cfg):
        dataset_cfg = getattr(cfg.datasets, DATASET_NAME)
        save_dir = os.path.join(cfg.datasets.data_root(), dataset_cfg.save_dir())
        n_features = dataset_cfg.n_features()
        debug = cfg.debug_datasets() or cfg.debug()

        if debug: print(f"Initializing {DATASET_NAME} dataset with save_dir: {save_dir} and n_features: {n_features}")

        if save_dir is None:
            raise ValueError("Save directory must be specified in the configuration.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df["target"] = (df["target"] > df["target"].median()).astype(int)  # convert to binary
        data = preprocess_numeric(cfg, df, target_col="target", n_features=n_features)
        
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']

        if debug: print(f"Train shape: {self.train.shape}, Val shape: {self.val.shape}, Test shape: {self.test.shape}")