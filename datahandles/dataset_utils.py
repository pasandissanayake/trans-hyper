import copy
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


datahandles = {}


def register(name):
    def decorator(cls):
        datahandles[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datahandles[dataset_spec['name']](**dataset_args)
    return dataset


def preprocess_numeric(cfg, df, target_col, n_features, random_state=42):
    # Select only numerical features
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) < n_features:
        raise ValueError(f"Not enough numeric features in the dataset. Found {len(numeric_cols)}, required {n_features}.")
    selected_cols = list(numeric_cols[:n_features])
    
    # Normalize and center
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_cols])
    
    X = pd.DataFrame(X_scaled, columns=selected_cols)
    y = df[target_col].reset_index(drop=True)
    
    # Merge and shuffle
    data = pd.concat([X, y], axis=1)
    
    # Split
    train, temp = train_test_split(data, test_size=0.3, stratify=y, random_state=random_state)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[target_col], random_state=random_state)
    
    return {
        'train': train.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'test': test.reset_index(drop=True)
    }


class RawDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_cfg = getattr(self.cfg.datasets, self.name)
        self.save_dir = os.path.join(self.cfg.datasets.data_root(), self.dataset_cfg.save_dir())
        self.n_features = self.dataset_cfg.n_features()
        self.debug = self.cfg.debug_datasets() or self.cfg.debug()

        if self.debug: print(f"Initializing {self.name} dataset with save_dir: {self.save_dir} and n_features: {self.n_features}")

        if self.save_dir is None:
            raise ValueError("Save directory must be specified in the configuration.")
        
        self._fetch_data()
                
        if self.debug: print(f"Train shape: {self.data['train'].shape}, Val shape: {self.data['val'].shape}, Test shape: {self.data['test'].shape}")

    def __getitem__(self, split):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test'].")
        return self.data[split]