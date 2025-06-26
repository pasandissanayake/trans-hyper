import os
import pandas as pd

from sklearn.datasets import load_diabetes

from .dataset_utils import *

DATASET_NAME = 'diabetes'

@register(DATASET_NAME)
class DiabetesDataset(RawDataset):
    def __init__(self, cfg):
        self.name = DATASET_NAME
        super(DiabetesDataset, self).__init__(cfg)
        
    def _fetch_data(self):        
        os.makedirs(self.save_dir, exist_ok=True)
        data = load_diabetes(as_frame=True)
        df = data.frame.copy() # type: ignore
        df["target"] = (df["target"] > df["target"].median()).astype(int)  # convert to binary
        self.data = preprocess_numeric(self.cfg, df, target_col="target", n_features=self.n_features)