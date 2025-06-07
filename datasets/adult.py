import os
import pandas as pd
import openml
from .dataset_utils import *

DATASET_NAME = 'adult'

@register(DATASET_NAME)
class AdultDataset:
    def __init__(self, cfg):
        dataset_cfg = getattr(cfg.datasets, DATASET_NAME)
        save_dir = os.path.join(cfg.datasets.data_root(), dataset_cfg.save_dir())
        n_features = dataset_cfg.n_features()
        debug = cfg.debug_datasets() or cfg.debug()

        if debug: print(f"Initializing {DATASET_NAME} dataset with save_dir: {save_dir} and n_features: {n_features}")

        if save_dir is None:
            raise ValueError("Save directory must be specified in the configuration.")
        
        os.makedirs(save_dir, exist_ok=True)
        # Configure the OpenML cache directory to your desired folder
        openml.config.set_root_cache_directory(os.path.expanduser(save_dir))

        if debug: print("Downloading Adult dataset from OpenML...")
        dataset = openml.datasets.get_dataset(1590, download_data=True)
        df, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        df['label'] = y    
        if debug: print(f"Saved dataset to {save_dir}")
        
        data = preprocess_numeric(df, target_col='label', n_features=n_features)
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']