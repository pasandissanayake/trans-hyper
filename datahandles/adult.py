import os
import pandas as pd

import openml

from .dataset_utils import *

DATASET_NAME = 'adult'

@register(DATASET_NAME)
class AdultDataset(RawDataset):
    def __init__(self, cfg):
        self.name = DATASET_NAME
        super(AdultDataset, self).__init__(cfg)

    def _fetch_data(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # Configure the OpenML cache directory to your desired folder
        openml.config.set_root_cache_directory(os.path.expanduser(self.save_dir))

        if self.debug: print("Downloading Adult dataset from OpenML...")
        dataset = openml.datasets.get_dataset(1590, download_data=True)
        df, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        df['label'] = y.cat.rename_categories({'>50K': 1, '<=50K': 0})
        if self.debug: print(f"Saved dataset to {self.save_dir}")
        self.data = preprocess_numeric(self.cfg, df, target_col='label', n_features=self.n_features)

    