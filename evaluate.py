import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from models import make
from models.t0 import T0RegressionModel  # noqa: F401
from datahandles import FewShotDataset
from utils import load_cfg, dict_to_mlp


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


def compute_metrics(model, queries, X, y, num_classes: int,
                    post_training: bool, post_training_epochs: int):
    """Run hyponet forward and compute multiclass metrics."""
    
    hyponet = dict_to_mlp(
        weight_dict=model(queries[0]).params,
        in_dim=X.shape[1]
    ).cuda()

    if post_training:
        num_trainable_params = sum(
            p.numel() for p in hyponet.parameters() if p.requires_grad
        )
        print(f'Number of trainable parameters in MLP: {num_trainable_params:,}')

        epochs = post_training_epochs
        loss_fn = nn.CrossEntropyLoss()   # ✅ multiclass loss
        opt = optim.Adam(hyponet.parameters(), lr=1e-3)

        best_acc = -1.0
        best_epoch = -1

        for ep in range(1, epochs + 1):
            hyponet.train()
            for query in queries:
                xb = query["queries_x"].to("cuda")
                yb = query["queries_y"].long().to("cuda")  # class indices

                opt.zero_grad()
                logits = hyponet(xb)                      # [B, C]
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            # ---------- Validation ----------
            hyponet.eval()
            ys, preds = [], []

            with torch.no_grad():
                for query in queries:
                    xb = query["queries_x"].to("cuda")
                    yb = query["queries_y"]

                    logits = hyponet(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                    preds.append(probs)
                    ys.append(yb.numpy())

            ys = np.concatenate(ys)
            preds = np.concatenate(preds)

            y_pred = np.argmax(preds, axis=1)
            acc = roc_auc_score(ys, preds, multi_class="ovr")

            if acc > best_acc:
                best_acc = acc
                best_epoch = ep

    # ---------- Final Evaluation ----------
    hyponet.eval()
    logits = hyponet(X.cuda())                 # [N, C]
    preds = torch.softmax(logits, dim=1)
    preds = preds.detach().cpu().numpy()

    y_pred = np.argmax(preds, axis=1)

    if np.isnan(preds).any() or np.isinf(preds).any():
        print("NaNs in preds!", np.isnan(preds).sum())

    val_counts = pd.Series(y).value_counts()
    const_pred_acc = max(val_counts) / sum(val_counts)

    balanced_acc = balanced_accuracy_score(y, y_pred)
    unbalanced_acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")   # macro = class-balanced

    roc_auc = roc_auc_score(
        y_true=y,
        y_score=preds,
        labels=np.arange(num_classes),
        multi_class="ovr",
        average="macro"
    )

    return {
        "const_predictor_acc": const_pred_acc,
        "balanced_acc": balanced_acc,
        "unbalanced_acc": unbalanced_acc,
        "f1_score_macro": f1,
        "roc_auc": roc_auc
    }


# def compute_metrics(model, queries, X, y, post_training: bool, post_training_epochs: int):
#     """Run hyponet forward and compute metrics."""
#     hyponet = dict_to_mlp(weight_dict=model(queries[0]).params, in_dim=X.shape[1]).cuda()

#     if post_training:
#         num_trainable_params = sum(p.numel() for p in hyponet.parameters() if p.requires_grad)
#         print(f'Number of trainable parameters in MLP: {num_trainable_params:,}')

#         epochs = post_training_epochs    
#         loss_fn = nn.BCEWithLogitsLoss()
#         opt = optim.Adam(hyponet.parameters(), lr=1e-3)

#         best_val_auc = -1.0
#         best_epoch = -1
#         for ep in range(1, epochs + 1):
#             hyponet.train()
#             for query in queries:
#                 xb = query["queries_x"]
#                 yb = query["queries_y"].to(torch.float)

#                 xb = xb.to("cuda")
#                 yb = yb.to("cuda")
#                 opt.zero_grad()
#                 logits = hyponet(xb)
#                 loss = loss_fn(logits[:,1], yb)
#                 loss.backward()
#                 opt.step()
#             # Validate
#             hyponet.eval()
#             ys, preds = [], []
#             with torch.no_grad():
#                 for query in queries:
#                     xb = query["queries_x"]
#                     yb = query["queries_y"].to(torch.float)
#                     xb = xb.to("cuda")
#                     logits = hyponet(xb)
#                     probs = torch.sigmoid(logits).cpu().numpy()
#                     preds.append(probs)
#                     ys.append(yb.numpy())
#             ys = np.concatenate(ys)
#             preds = np.concatenate(preds)
#             val_auc = roc_auc_score(y_true=ys, y_score=preds[:,1])
#             if val_auc > best_val_auc:
#                 best_val_auc = val_auc
#                 best_epoch = ep
#             # print(f"epoch: {ep:>2}, roc_auc: {val_auc:.2}, best epoch: {best_epoch:>2}")


#     hyponet.eval()
#     preds = hyponet.forward(X.unsqueeze(dim=0).cuda())
#     preds = preds.detach().cpu().numpy()
#     preds = np.squeeze(preds)
#     y_pred = np.argmax(preds, axis=1)

#     if np.isnan(preds).any() or np.isinf(preds).any():
#         print("NaNs in preds!", np.isnan(preds).sum())

#     val_counts = pd.Series(y).value_counts()
#     const_pred_acc = max(val_counts) / sum(val_counts)
#     balanced_acc = balanced_accuracy_score(y, y_pred)
#     unbalanced_acc = accuracy_score(y, y_pred)
#     f1 = f1_score(y, y_pred)
#     roc_auc = roc_auc_score(y, preds[:, 1])

#     return {
#         "const_predictor_acc": const_pred_acc,
#         "balanced_acc": balanced_acc,
#         "unbalanced_acc": unbalanced_acc,
#         "f1_score": f1,
#         "roc_auc": roc_auc
#     }


def compute_avg_metrics(model, cfg, ds_name, n_shots, n_queries, n_samples, max_n_features, post_training, post_training_epochs):
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
        
        X = test_ds[i]['queries_x']
        y = test_ds[i]['queries_y']
        metrics = compute_metrics(model=model, 
                                  queries=train_ds, 
                                  X=X, 
                                  y=y,
                                  num_classes=cfg.hyponet.out_dim, 
                                  post_training=post_training, 
                                  post_training_epochs=post_training_epochs)
        for key, val in metrics.items():
            avg_metrics[key] = avg_metrics.get(key, 0.0) + val

    avg_metrics = {key: val / n_samples for key, val in avg_metrics.items()}
    return avg_metrics


def evaluate_checkpoint(checkpoint_path, post_training, post_training_epochs, device="cuda"):
    """Load checkpoint, model, dataset, evaluate and return results row."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_cfg(cfg_dict=checkpoint["cfg"])
    model = make(model_name=cfg.hypernet.name, cfg=cfg, sd=checkpoint["model"]).to(device)

    if "tokenizer" in cfg.keys():
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.model)
        def model_fn(query):
                 
            tokens = tokenizer(query["shots"], 
                            truncation=cfg.tokenizer.truncation, 
                            padding=cfg.tokenizer.padding, 
                            max_length=cfg.tokenizer.max_length,
                            return_tensors='pt')
            return model(tokens.to("cuda"))
    else:
        def model_fn(query):
            return model(query)
    
    total_training_set_size = cfg.datasets.n_shots * cfg.datasets.train_size \
        if cfg.datasets.queries_same_as_shots \
        else (cfg.datasets.n_queries + cfg.datasets.n_shots) * cfg.datasets.train_size

    ds_name = cfg.datasets.list_combine_train[0]
    max_n_features = cfg.hyponet.in_dim
    n_samples = 1
    n_queries_dict = {"bank": 43211, "blood": 374, "calhousing": 19640, "car": 864, "creditg": 500, "diabetes": 384, "heart": 459, "higgs": 96049, "income": 44222, "incomemix": 44222, "jungle": 42819, "mfeatfourier": 1000, "vehicle": 423}
    n_queries = n_queries_dict[ds_name]
    n_shots = total_training_set_size

    metrics = compute_avg_metrics(model_fn, 
                                  cfg, 
                                  ds_name, 
                                  n_shots, 
                                  n_queries, 
                                  n_samples, 
                                  max_n_features, 
                                  post_training,
                                  post_training_epochs)

    result = {
        "dataset": ds_name,
        # "seed": cfg.random_state,
        "train_examples": total_training_set_size,
        "roc_auc": metrics["roc_auc"]
    }

    del model, checkpoint
    torch.cuda.empty_cache()
    return result


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("folders", nargs="+", help="List of checkpoint folders")
#     parser.add_argument("--outfile", type=str, default="results.csv", help="CSV file to save results")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
#     parser.add_argument("--epoch", type=str, default="best", help="Select the best or last epoch")
#     parser.add_argument("--post-train", action=argparse.BooleanOptionalAction, help="Whether to train the MLP after inference")
#     parser.add_argument("--pt-epochs", type=int, help="Number of epochs to post-train")
#     args = parser.parse_args()
    
#     print(f"Using the {args.epoch} epoch and post_train={args.post_train}")

#     results = []
#     for folder in tqdm(args.folders, desc="Evaluating checkpoints"):
#         if args.epoch == "best":
#             checkpoint_path = os.path.join(folder, "epoch-best-balacc.pth")
#         elif args.epoch == "last":
#             checkpoint_path = os.path.join(folder, "epoch-last.pth")
#         else:
#             print("Epoch must be best or last. Got {args.epoch}")
#             return

        

#         if not os.path.exists(checkpoint_path):
#             print(f"Warning: {checkpoint_path} not found, skipping")
#             continue
#         result = evaluate_checkpoint(checkpoint_path, args.post_train, args.pt_epochs, device=args.device)
#         results.append(result)

#     df = pd.DataFrame(results)
#     print(df.round(2).to_string(index=False))
#     df.to_csv(args.outfile, index=False)
#     print(f"\nSaved results to {args.outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", help="List of checkpoint folders")
    parser.add_argument("--outfile", type=str, default="results.csv", help="CSV file to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--epoch", type=str, default="best", help="Select the best or last epoch")
    parser.add_argument("--post-train", action=argparse.BooleanOptionalAction, help="Whether to train the MLP after inference")
    parser.add_argument("--pt-epochs", type=int, help="Number of epochs to post-train")
    args = parser.parse_args()
    
    print(f"Using the {args.epoch} epoch and post_train={args.post_train}")

    # Group folders by base checkpoint name (without the seed)
    folder_dict = {}
    for folder in args.folders:
        # Assume seed is in the format "seedXX" at the end
        base_name = folder.rsplit("-seed", 1)[0]
        folder_dict.setdefault(base_name, []).append(folder)

    results_agg = []
    for base_name, seed_folders in tqdm(folder_dict.items(), desc="Evaluating checkpoint groups"):
        all_results = []
        for folder in seed_folders:
            if args.epoch == "best":
                checkpoint_path = os.path.join(folder, "epoch-best-balacc.pth")
            elif args.epoch == "last":
                checkpoint_path = os.path.join(folder, "epoch-last.pth")
            else:
                print(f"Epoch must be 'best' or 'last'. Got {args.epoch}")
                return

            if not os.path.exists(checkpoint_path):
                print(f"Warning: {checkpoint_path} not found, skipping")
                continue

            result = evaluate_checkpoint(checkpoint_path, args.post_train, args.pt_epochs, device=args.device)
            all_results.append(result)

        if all_results:
            # Convert list of dicts to DataFrame
            df_seed = pd.DataFrame(all_results)
            
            # Select only numeric columns
            numeric_cols = df_seed.select_dtypes(include=np.number).columns
            df_mean = df_seed[numeric_cols].mean().add_suffix("_mean")
            df_std = df_seed[numeric_cols].std().add_suffix("_std")
            
            # Keep non-numeric info (like checkpoint name)
            df_combined = pd.concat([df_mean, df_std])
            df_combined["checkpoint"] = base_name
            results_agg.append(df_combined)

    if results_agg:
        final_df = pd.DataFrame(results_agg).drop(columns=["train_examples_std"])
        # Reorder columns: checkpoint first
        cols = ["checkpoint"] + [c for c in final_df.columns if c != "checkpoint"]
        final_df = final_df[cols]
        print(final_df.round(2).to_string(index=False))
        final_df.to_csv(args.outfile, index=False)
        print(f"\nSaved aggregated results to {args.outfile}")
    else:
        print("No valid results found.")


if __name__ == "__main__":
    main()