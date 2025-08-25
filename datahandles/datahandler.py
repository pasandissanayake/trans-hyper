import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from string import Template


def preprocess_numerical(ref_df, df, num_cols, method="standard"):
        """
        Preprocess numerical columns of test_df based on statistics from train_df.
        
        Parameters
        ----------
        ref_df : pd.DataFrame
            Training split (used to compute statistics).
        df : pd.DataFrame
            Df to be transformed (transformed using ref_df stats).
        num_cols : list of str
            List of numerical column names to preprocess.
        method : str, optional (default="standard")
            Preprocessing method. Options:
            - "standard": (x - mean) / std
            - "minmax": (x - min) / (max - min)
        
        Returns
        -------
        train_processed, test_processed : pd.DataFrame
            Copies of train_df and test_df with transformed numerical columns.
        """
        
        df_proc = df.copy()
        
        if method == "standard":
            stats = ref_df[num_cols].agg(["mean", "std"]).T
            for col in num_cols:
                mean, std = stats.loc[col, "mean"], stats.loc[col, "std"]
                df_proc[col] = (df_proc[col] - mean) / (std if std > 0 else 1)
                
        elif method == "minmax":
            stats = ref_df[num_cols].agg(["min", "max"]).T
            for col in num_cols:
                min_val, max_val = stats.loc[col, "min"], stats.loc[col, "max"]
                denom = (max_val - min_val) if (max_val > min_val) else 1
                df_proc[col] = (df_proc[col] - min_val) / denom
        
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        return df_proc


def onehot_encode_multiple(df, cat_dict, drop_unknown=False):
    """
    One-hot encode multiple categorical columns based on given categories,
    inserting the encoded columns in place of the originals (preserving order).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cat_dict : dict
        Dictionary of {col_name: [categories]} defining the order of categories for each column.
    drop_unknown : bool, optional (default=False)
        If True, unknown categories get all zeros.
        If False, raises an error on unknown categories.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns replacing the original categorical columns,
        with the new columns inserted in the same order as the original columns.
    """
    df_encoded = df.copy()
    new_cols = []
    for col in df.columns:
        if col in cat_dict:  # categorical column to encode
            categories = cat_dict[col]
            # Initialize all columns with 0
            onehot_df = pd.DataFrame(
                0, index=df_encoded.index, 
                columns=[f"{col}-{cat}" for cat in categories]
            )
            # Fill corresponding one-hot positions
            for i, val in df_encoded[col].items():
                if val in categories:
                    onehot_df.loc[i, f"{col}-{val}"] = 1
                elif not drop_unknown:
                    raise ValueError(f"Unknown category '{val}' in column '{col}', i={i}")
                # else: keep row all zeros

            new_cols.append(onehot_df)
        else:  # non-categorical column
            new_cols.append(df_encoded[[col]])
    
    # Concatenate back in the correct order
    df_encoded = pd.concat(new_cols, axis=1)
    return df_encoded


def df_to_text(df: pd.DataFrame, task_template_str: str, label_template_str: str, label_col: str):
    """
    Convert a DataFrame into TabLLM-style text passages for LLMs 
    using task + label templates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    task_template_str : str
        Template string for task, must include ${row_text}.
    label_template_str : str
        Template string for label, must include ${label}.
    label_col : str
        Column name for label.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - 'text': task + row converted into text (TabLLM style)
        - 'label': formatted label
    """
    task_template = Template(task_template_str)
    label_template = Template(label_template_str)

    cols = [c for c in df.columns if c != label_col]
    col_indices = [df.columns.get_loc(c) for c in cols]
    label_idx = df.columns.get_loc(label_col)

    # Build row_text for each row safely with tuple indexing
    row_texts = [
        ". ".join(f"{col} is {row[i]}" for col, i in zip(cols, col_indices))
        for row in df.itertuples(index=False, name=None)  # name=None gives plain tuples
    ]

    texts = [task_template.safe_substitute({"row_text": rt}) for rt in row_texts]
    labels = [label_template.safe_substitute({"label": row[label_idx]}) 
              for row in df.itertuples(index=False, name=None)]

    return pd.DataFrame({"text": texts, "label": labels})


class DataHandler:
    def __init__(self, data_path:str|Path, split: str):
        self.data_path = Path(data_path)

        self.raw_train_df = pd.read_csv(self.data_path / 'train.csv') # training split is always loaded to extract stats for normalizing
        if split == 'train':
                self.df = pd.read_csv(self.data_path / 'train.csv')
        elif split == 'test':
                self.df = pd.read_csv(self.data_path / 'test.csv')
        else:
             raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

        with open(self.data_path / 'columns.json', 'r') as file:
            self.metadata = json.load(file)

        # Load templates from YAML
        with open(self.data_path / "template.yaml", "r") as f:
            templates = yaml.safe_load(f)
        
        if "task_template" not in templates or "label_template" not in templates:
            raise ValueError("YAML file must contain 'task_template' and 'label_template'.")
        
        self.task_template = templates["task_template"]
        self.label_template = templates.get("label_template", "")

        self.columns = self.raw_train_df.columns.tolist()
        self.target_name = self.columns[-1] # Assuming the last column is the target
        self.columns = self.columns[:-1] # Exclude target column

        self.num_cols = [col for col in self.columns if self.metadata[col]["type"] == 'numerical']
        self.cat_cols = [col for col in self.columns if self.metadata[col]["type"] == 'categorical']

    def get_split(self, col_permutation: list[int] = [], shuffle: bool = True, preprocess: bool = True):        
        df = self.df        
        if len(col_permutation) == len(self.columns):
            col_permutation.append(len(self.columns))  # Ensure target column is included
            df = df.iloc[:, col_permutation]
        elif len(col_permutation) > 0:
            raise ValueError(f"Permutation length {len(col_permutation)} does not match number of columns {len(self.columns)}.")
        
        prompts = df_to_text(
            df=df,
            task_template_str=self.task_template,
            label_template_str=self.label_template,
            label_col=self.target_name
        )

        if preprocess:
            df = preprocess_numerical(
                ref_df=self.raw_train_df,
                df=df,
                num_cols=self.num_cols
            )

            cat_dict = {
                col: self.metadata[col]["values"] for col in self.cat_cols
            }
            df = onehot_encode_multiple(
                df=df,
                cat_dict=cat_dict,
                drop_unknown=False
            )

            target_enc_dict = {
                val: i
                for i, val in enumerate(self.metadata[self.metadata["target_name"]]["values"])
            }
            df[self.target_name] = df[self.target_name].map(target_enc_dict)
            
        if shuffle:
            random_state = 42
            perm = np.random.RandomState(seed=random_state).permutation(len(df))
            df = df.iloc[perm].reset_index(drop=True)
            prompts = prompts.iloc[perm].reset_index(drop=True)

        return df, prompts