import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from models import make
from models.t0 import T0RegressionModel  # noqa: F401
from datahandles import FewShotDataset
from utils import load_cfg


def load_dataset(cfg, test_list, n_shots, n_queries, max_n_features, test_size=1, test_permutation=False):
    """Create train and test FewShotDataset objects."""
    train_ds = FewShotDataset(
        dataset_names=test_list,
        data_root=cfg.datasets.data_root,
        split='train',
        split_size=cfg.datasets.train_size,
        n_shots=cfg.datasets.n_shots,
        n_queries=cfg.datasets.n_queries,
        queries_same_as_shots=cfg.datasets.queries_same_as_shots,
        max_n_features=max_n_features,
        balance_labels=True,
        col_permutation=False,
        shuffle=False,
        debug=False,
        random_seed=cfg.random_state,
        shots_with_labels=True
    )

    test_ds = FewShotDataset(
        dataset_names=test_list,
        data_root=cfg.datasets.data_root,
        split='test',
        split_size=test_size,
        n_shots=n_shots,
        n_queries=n_queries,
        queries_same_as_shots=True,
        max_n_features=max_n_features,
        balance_labels=False,
        col_permutation=test_permutation,
        shuffle=True,
        debug=False,
        random_seed=True,
        shots_with_labels=False
    )
    return test_ds, train_ds


def compute_metrics(model, queries, X, y):
    """Run hyponet forward and compute metrics."""
    hyponet = model(queries)
    hyponet.eval()
    preds = hyponet.forward(X.unsqueeze(dim=0).cuda())
    preds = preds.detach().cpu().numpy()
    preds = np.squeeze(preds)
    y_pred = np.argmax(preds, axis=1)

    val_counts = pd.Series(y).value_counts()
    const_pred_acc = max(val_counts) / sum(val_counts)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    unbalanced_acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, preds[:, 1])

    return {
        "const_predictor_acc": const_pred_acc,
        "balanced_acc": balanced_acc,
        "unbalanced_acc": unbalanced_acc,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def compute_avg_metrics(model, cfg, ds_name, n_shots, n_queries, n_samples, max_n_features):
    """Average metrics across test samples."""
    if n_shots < 1:
        test_ds, train_ds = load_dataset(cfg, [ds_name], n_shots=n_queries, n_queries=n_queries,
                                         test_size=n_samples, max_n_features=max_n_features)
    else:
        test_ds, train_ds = load_dataset(cfg, [ds_name], n_shots=n_shots, n_queries=n_queries,
                                         test_size=n_samples, max_n_features=max_n_features)

    avg_metrics = {}
    for i in range(n_samples):
        if n_queries < 1:
            raise ValueError("Number of queries must be >= 1")
        queries = train_ds[0]
        X = test_ds[i]['queries_x']
        y = test_ds[i]['queries_y']
        metrics = compute_metrics(model=model, queries=queries, X=X, y=y)
        for key, val in metrics.items():
            avg_metrics[key] = avg_metrics.get(key, 0.0) + val

    avg_metrics = {key: val / n_samples for key, val in avg_metrics.items()}
    return avg_metrics


def evaluate_checkpoint(checkpoint_path, device="cuda"):
    """Load checkpoint, model, dataset, evaluate and return results row."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_cfg(cfg_dict=checkpoint["cfg"])
    model = make(model_name=cfg.hypernet.name, cfg=cfg, sd=checkpoint["model"]).to(device)

    if "tokenizer" in cfg.keys():
        AutoTokenizer.from_pretrained(cfg.tokenizer.model)

    total_training_set_size = cfg.datasets.n_shots * cfg.datasets.train_size \
        if cfg.datasets.queries_same_as_shots \
        else (cfg.datasets.n_queries + cfg.datasets.n_shots) * cfg.datasets.train_size

    ds_name = cfg.datasets.list_combine_train[0]
    max_n_features = cfg.hyponet.in_dim
    n_samples = 1
    n_queries_dict = {"bank": 43211, "calhousing": 19640, "income": 44222}
    n_queries = n_queries_dict[ds_name]
    n_shots = total_training_set_size

    metrics = compute_avg_metrics(model, cfg, ds_name, n_shots, n_queries, n_samples, max_n_features)

    result = {
        "dataset": ds_name,
        "seed": cfg.random_state,
        "train_examples": total_training_set_size,
        "roc_auc": metrics["roc_auc"]
    }

    del model, checkpoint
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", help="List of checkpoint folders")
    parser.add_argument("--outfile", type=str, default="results.csv", help="CSV file to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--epoch", type=str, default="best", help="Select the best or last epoch")
    args = parser.parse_args()

    results = []
    for folder in tqdm(args.folders, desc="Evaluating checkpoints"):
        if args.epoch == "best":
            checkpoint_path = os.path.join(folder, "epoch-best-balacc.pth")
        elif args.epoch == "last":
            checkpoint_path = os.path.join(folder, "epoch-last.pth")
        else:
            print("Epoch must be best or last. Got {args.epoch}")
            return

        print(f"Using the {args.epoch} epoch")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: {checkpoint_path} not found, skipping")
            continue
        result = evaluate_checkpoint(checkpoint_path, device=args.device)
        results.append(result)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(args.outfile, index=False)
    print(f"\nSaved results to {args.outfile}")


if __name__ == "__main__":
    main()