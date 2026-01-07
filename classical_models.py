
import os
import argparse
import random
import joblib
import json
from pathlib import Path
from typing import Tuple, Any, Dict, List
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import loguniform, uniform, randint

# Try imports that may not be installed in all environments
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None

from datahandles import FewShotDataset


def get_training_data(ds_name, n_shots, random_seed) -> Dict[str, np.ndarray]:
    """Return X_train, y_train only. We will handle cross-validation ourselves."""
    train_ds = FewShotDataset(
        dataset_names=[ds_name],
        data_root="./data",
        split="train",
        split_size=1,
        n_shots=n_shots,
        n_queries=n_shots,
        queries_same_as_shots=True,
        max_n_features=None,
        balance_labels=True,
        col_permutation=False,
        shuffle=True,
        debug=False,
        random_seed=random_seed,
        shots_with_labels=False
    )
    return {
        "X_train": train_ds[0]["queries_x"].numpy(),
        "y_train": train_ds[0]["queries_y"].to(torch.long).numpy()
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, width: int, output_dim: int = 1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_mlp(X_train, y_train, X_val, y_val, mlp_config, device='cpu', epochs=30, batch_size=32, lr=1e-3, seed=0, ckpt_path=None):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install torch to train the MLP.")
    set_global_seed(seed)
    model = TorchMLP(input_dim=X_train.shape[1], n_layers=mlp_config[0], width=mlp_config[1]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_auc = -1.0
    best_epoch = -1
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        # Validate
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                ys.append(yb.numpy())
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        try:
            val_auc = roc_auc_score(ys, preds)
        except Exception:
            val_auc = float('nan')
        if ckpt_path is not None:
            # Save checkpoint each epoch (or only best if you prefer)
            ckpt_file = ckpt_path / f"mlp_epoch{ep}.pt"
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'val_auc': val_auc,
                'seed': seed,
                'mlp_config': mlp_config
            }, str(ckpt_file))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = ep
    return model, best_val_auc, best_epoch


def make_cv(y, n_folds, seed):
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    n_splits = min(n_folds, min_class_count)
    print(f"n_splits: {n_splits}, min_class_count={min_class_count}")
    if n_splits < 2:  # fallback
        n_splits = 2
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_val_auc(model_name, X, y, seed, mlp_arch=None, mlp_epochs=30):
    """Run 4-fold CV for given model, return mean AUC."""
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_name == "logistic":
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, probs))

        elif model_name == "lightgbm" and lgb is not None:
            lgb_train = lgb.Dataset(X_train, label=y_train)
            params = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": seed}
            gbm = lgb.train(params, lgb_train, num_boost_round=100)
            probs = gbm.predict(X_val)
            aucs.append(roc_auc_score(y_val, probs))

        elif model_name == "xgboost" and xgb is not None:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val)
            param = {"objective": "binary:logistic", "eval_metric": "auc", "seed": seed}
            bst = xgb.train(param, dtrain, num_boost_round=100)
            probs = bst.predict(dval)
            aucs.append(roc_auc_score(y_val, probs))

        elif model_name == "tabpfn" and TabPFNClassifier is not None:
            tabpfn = TabPFNClassifier(device="cpu")
            tabpfn.fit(X_train, y_train)
            probs = tabpfn.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, probs))

        elif model_name == "mlp" and torch is not None:
            ckpt_subdir = Path("tmp") / f"mlp_cv_seed{seed}_fold{fold}"
            ckpt_subdir.mkdir(parents=True, exist_ok=True)
            _, val_auc, _ = train_torch_mlp(
                X_train, y_train, X_val, y_val,
                mlp_arch, device="cpu", epochs=mlp_epochs,
                batch_size=32, lr=1e-3, seed=seed,
                ckpt_path=ckpt_subdir
            )
            aucs.append(val_auc)

    return np.mean(aucs)


def train_and_save_all_models_with_cv(
    ds_name: str,
    n_shots: int,
    seeds: List[int],
    mlp_arch: Tuple[int, int],
    out_dir: str = "classical_checkpoints",
    n_folds: int = 4
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        set_global_seed(seed)

        data = get_training_data(ds_name=ds_name,
                                     n_shots=n_shots,
                                     random_seed=seed)

        X_train = data["X_train"]
        y_train = data["y_train"]

        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Stratified 4-fold CV
        cv = make_cv(y_train, n_folds, seed)

        # -------------------------
        # 1) Logistic Regression
        # -------------------------
        print("Tuning Logistic Regression...")
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_log = grid.best_estimator_
        print(f"Best Logistic params: {grid.best_params_}")
        joblib.dump(best_log, seed_dir / "logistic.pkl")

        # # -------------------------
        # # 2) LightGBM
        # # -------------------------
        # if lgb is not None:
        #     print("Tuning LightGBM...")
        #     param_grid = {
        #         'num_leaves': [16, 31, 64],
        #         'learning_rate': [0.01, 0.05, 0.1],
        #         'n_estimators': [50, 100, 200]
        #     }
        #     lgbm = lgb.LGBMClassifier(objective='binary', random_state=seed)
        #     grid = GridSearchCV(lgbm, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        #     grid.fit(X_train, y_train)
        #     best_lgb = grid.best_estimator_
        #     print(f"Best LightGBM params: {grid.best_params_}")
        #     best_lgb.booster_.save_model(str(seed_dir / "lightgbm.txt"))

        # -------------------------
        # 3) XGBoost
        # -------------------------
        if xgb is not None:
            print("Tuning XGBoost...")
            xgbc = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=seed
            )
            # param_grid = {
            #     'max_depth': [3, 5, 7],
            #     'learning_rate': [0.01, 0.05, 0.1],
            #     'n_estimators': [50, 100, 200]
            # }
            # grid = GridSearchCV(xgbc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
            param_distrib = {
                'max_depth': randint(low=1, high=11),
                'n_estimators': [1000],
                'min_child_weight': [1, 100],
                'subsample': uniform(loc=0.5, scale=0.5),
                'learning_rate': loguniform(1e-5, 0.7),
                'colsample_bylevel': uniform(loc=0.5, scale=0.5),
                'colsample_bytree': uniform(loc=0.5, scale=0.5),
                'gamma': loguniform(1e-8, 7),
                'lambda': loguniform(1, 4),
                'alpha': loguniform(1e-8, 1e2)
            }
            grid = RandomizedSearchCV(estimator=xgbc,
                                      param_distributions=param_distrib,
                                      n_iter=20,
                                      cv=cv,
                                      scoring='roc_auc',
                                      n_jobs=-1)
            grid.fit(X_train, y_train)
            best_xgb = grid.best_estimator_
            print(f"Best XGBoost params: {grid.best_params_}")
            best_xgb.save_model(str(seed_dir / "xgboost.json"))

        # -------------------------
        # 4) TabPFN
        # -------------------------
        if TabPFNClassifier is not None:
            print("Training TabPFN (no CV tuning, pretrained)...")
            tabpfn = TabPFNClassifier(device='cpu')
            tabpfn.fit(X_train, y_train)
            joblib.dump(tabpfn, seed_dir / "tabpfn.pkl")

        # -------------------------
        # 5) MLP (manual CV)
        # -------------------------
        if torch is not None:
            print("Tuning MLP with manual CV...")
            # Define search grid
            mlp_epoch_grid = [30, 50, 100, 300]
            mlp_lr_grid = [1e-5, 1e-4, 1e-3, 1e-2]

            best_auc = -1
            best_params = {}
            for epochs in mlp_epoch_grid:
                for lr in mlp_lr_grid:
                    fold_aucs = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        _, val_auc, _ = train_torch_mlp(
                            X_tr, y_tr, X_val, y_val, mlp_arch,
                            device='cpu', epochs=epochs, lr=lr, seed=seed
                        )
                        fold_aucs.append(val_auc)
                    mean_auc = np.mean(fold_aucs)
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = {"epochs": epochs, "lr": lr}

            print(f"Best MLP params: {best_params}, CV AUC={best_auc:.4f}")

            # Train final MLP on full training set with best hyperparameters
            model, _, _ = train_torch_mlp(
                X_train, y_train, X_train, y_train, mlp_arch,
                device='cpu', epochs=best_params["epochs"],
                lr=best_params["lr"], seed=seed
            )
            final_path = seed_dir / "mlp_final.pt"
            torch.save({'model_state_dict': model.state_dict(), 'best_params': best_params}, str(final_path))

    print("\n=== Done training all seeds with CV hyperopt ===")



def evaluate_checkpoints(checkpoint_root: str, X_test: np.ndarray, y_test: np.ndarray, arch: tuple) -> pd.DataFrame:
    root = Path(checkpoint_root)
    rows = []
    for seed_dir in sorted(root.glob('seed_*')):
        seed = int(seed_dir.name.split('_')[-1])
        # Logistic
        log_path = seed_dir / 'logistic.pkl'
        if log_path.exists():
            clf = joblib.load(log_path)
            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            rows.append({'model': 'LogisticRegression', 'seed': seed, 'roc_auc': auc})

        # # LightGBM
        # lgb_path = seed_dir / 'lightgbm.txt'
        # if lgb_path.exists() and lgb is not None:
        #     gbm = lgb.Booster(model_file=str(lgb_path))
        #     probs = gbm.predict(X_test)
        #     auc = roc_auc_score(y_test, probs)
        #     rows.append({'model': 'LightGBM', 'seed': seed, 'roc_auc': auc})

        # XGBoost
        xgb_path = seed_dir / 'xgboost.json'
        if xgb_path.exists() and xgb is not None:
            bst = xgb.Booster()
            bst.load_model(str(xgb_path))
            dtest = xgb.DMatrix(X_test)
            probs = bst.predict(dtest)
            auc = roc_auc_score(y_test, probs)
            rows.append({'model': 'XGBoost', 'seed': seed, 'roc_auc': auc})

        # TabPFN
        tabpfn_path = seed_dir / 'tabpfn.pkl'
        if tabpfn_path.exists() and TabPFNClassifier is not None:
            tabpfn = joblib.load(tabpfn_path)
            probs = tabpfn.predict_proba(X_test)[:, 1]
            rows.append({'model': 'TabPFN', 'seed': seed, 'roc_auc': roc_auc_score(y_test, probs)})

        # MLP - attempt to load final or best checkpoint
        mlp_final = seed_dir / 'mlp_final.pt'
        ckpt_subdir = seed_dir / 'mlp_checkpoints'
        if (mlp_final.exists() or (ckpt_subdir.exists() and any(ckpt_subdir.glob('mlp_epoch*.pt')))) and torch is not None:
            # Prefer final
            if mlp_final.exists():
                data = torch.load(str(mlp_final), map_location='cpu')
                params = data.get('best_params', None)
                model = TorchMLP(input_dim=X_test.shape[1], n_layers=arch[0], width=arch[1])
                model.load_state_dict(data['model_state_dict'])
            else:
                # pick last epoch file
                epochs = sorted(ckpt_subdir.glob('mlp_epoch*.pt'))
                last = epochs[-1]
                data = torch.load(str(last), map_location='cpu')
                arch = data.get('mlp_config', None)
                model = TorchMLP(input_dim=X_test.shape[1], n_layers=arch[0], width=arch[1])
                model.load_state_dict(data['model_state_dict'])
            model.eval()
            with torch.no_grad():
                inp = torch.from_numpy(X_test).float()
                logits = model(inp)
                probs = torch.sigmoid(logits).numpy()
            auc = roc_auc_score(y_test, probs)
            rows.append({'model': 'MLP', 'seed': seed, 'roc_auc': auc})

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-name', type=str)
    parser.add_argument('--n-shots', type=int)
    args = parser.parse_args()

    # User-configurable parameters
    ds_name = args.ds_name
    n_shots = args.n_shots
    seeds = [14, 26, 42, 58, 97]
    mlp_arch = (4, 10)  # (n_hidden_layers, width_of_hidden_layers)
    checkpoint_dir = f"checkpoints/{ds_name}/n{n_shots:0>2}"

    n_queries_dict = {"bank": 43211, "blood": 374, "calhousing": 19640, "heart": 459, "income": 44222}
    test_ds = FewShotDataset(
        dataset_names=[ds_name],
        data_root="./data",
        split="test",
        split_size=1,
        n_shots=n_queries_dict[ds_name],
        n_queries=n_queries_dict[ds_name],
        queries_same_as_shots=True,
        max_n_features=None,
        balance_labels=False,
        col_permutation=False,
        shuffle=True,
        debug=False,
        random_seed=True,
        shots_with_labels=False
    )

    X_test = test_ds[0]["queries_x"].numpy()
    y_test = test_ds[0]["queries_y"].to(torch.long).numpy()

    print(f"Test set size: {len(X_test)}")

    # Train & save
    train_and_save_all_models_with_cv(ds_name=ds_name,
                              n_shots=n_shots,
                              seeds=seeds, 
                              mlp_arch=mlp_arch, 
                              out_dir=checkpoint_dir)

    # Evaluate
    df_results = evaluate_checkpoints(checkpoint_dir, X_test, y_test, mlp_arch)

    # Save CSV and print nicely
    out_csv = Path(checkpoint_dir) / 'results.csv'
    df_results.to_csv(out_csv, index=False)

    # Aggregate and print table
    print(f'\n=== Results per model/seed - {ds_name} dataset, {n_shots} examples ===')
    print(tabulate(df_results.sort_values(['model', 'seed']), headers='keys', tablefmt='github', showindex=False, floatfmt='.4f'))

    # Also print mean per model
    print('\n=== Mean ROC-AUC per model ===')
    print(tabulate(df_results.groupby('model')['roc_auc'].agg(['mean', 'std']).reset_index(), headers='keys', tablefmt='github', showindex=False, floatfmt='.4f'))

    print(f"\nSaved results to: {out_csv}")

    from IPython import get_ipython

    if get_ipython():
        get_ipython().kernel.do_shutdown(restart=True)



