import copy
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import KFold


def make_kfolds(df: pd.DataFrame, k: int, random_state: int = 42, shuffle: bool = True):
    """
    Split a dataframe into k folds (non-stratified).
    
    Returns a list of (train_df, val_df) tuples.
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    folds = []
    for train_idx, val_idx in kf.split(df):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        folds.append((train_df.reset_index(drop=True), val_df.reset_index(drop=True)))
    return folds


def balance_dataframe(df: pd.DataFrame, target_col: str, method: str = "undersample", random_state: int = 42) -> pd.DataFrame:
    """
    Balance a DataFrame by oversampling or undersampling the target column.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        method (str): "undersample" | "oversample"
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Balanced dataframe
    """
    # Split by class
    classes = df[target_col].unique()
    dfs = [df[df[target_col] == c] for c in classes]

    if method == "undersample":
        min_size = min(len(d) for d in dfs)
        dfs_balanced = [resample(d, replace=False, n_samples=min_size, random_state=random_state) for d in dfs]

    elif method == "oversample":
        max_size = max(len(d) for d in dfs)
        dfs_balanced = [resample(d, replace=True, n_samples=max_size, random_state=random_state) for d in dfs]

    else:
        raise ValueError("method must be 'undersample' or 'oversample'")

    return pd.concat(dfs_balanced).sample(frac=1, random_state=random_state).reset_index(drop=True)


def sample_dataframe(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    method: str = "stratified",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Deterministically sample from a dataframe in either stratified or balanced manner.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        n_samples (int): Total number of samples desired
        method (str): "stratified" or "balanced"
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Sampled dataframe (deterministic order)
    """
    classes = sorted(df[target_col].unique())  # ensure fixed order
    dfs = [df[df[target_col] == c].sort_values(by=df.columns.tolist()).reset_index(drop=True) for c in classes]

    if method == "stratified":
        total = len(df)
        sampled_dfs = []
        for d in dfs:
            frac = len(d) / total
            n_class_samples = int(round(n_samples * frac))
            sampled_dfs.append(
                resample(
                    d, replace=False,
                    n_samples=min(n_class_samples, len(d)),
                    random_state=random_state
                )
            )

    elif method == "balanced":
        per_class = n_samples // len(classes)
        sampled_dfs = []
        for d in dfs:
            if len(d) < per_class:
                sampled_dfs.append(
                    resample(
                        d, replace=True,
                        n_samples=per_class,
                        random_state=random_state
                    )
                )
            else:
                sampled_dfs.append(
                    resample(
                        d, replace=False,
                        n_samples=per_class,
                        random_state=random_state
                    )
                )

    else:
        raise ValueError("method must be 'stratified' or 'balanced'")

    # Concatenate and shuffle deterministically
    result = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result