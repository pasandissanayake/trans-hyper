import torch
import torch.nn as nn

import models
from .base_trainer import BaseTrainer
from .trainers import register
import einops
from torcheval.metrics import MulticlassAccuracy, BinaryF1Score, BinaryAUROC, BinaryRecall
import numpy as np

TRAINER_NAME = "tabpfn_trainer"

@register(TRAINER_NAME)
class TabPFNTrainer(BaseTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)
        self.name = TRAINER_NAME
        
        self.log(f"Shots not used in TabPFN")
        self.log(f"Number of queries: {cfg.datasets.n_queries}")

        self.current_best_eval_acc = 0
        self.current_best_eval_balacc = 0

    def get_hyponet(self, data):
        queries_x = data['queries_x'].cuda()
        queries_y = data['queries_y'].cuda()
        data = {"queries_x": queries_x, "queries_y": queries_y}
        hyponet = self.model_ddp(data)
        return hyponet, queries_x, queries_y

    def compute_loss(self, data):
        hyponet, queries_x, queries_y = self.get_hyponet(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class"),
                         einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)"))
        return loss
    
    def compute_metrics(self, data, acc_only=False):
        """
        Works for both binary and multiclass classification.
        Uses torcheval metrics only.
        """

        from torcheval.metrics import (
            MulticlassAccuracy,
            MulticlassF1Score,
            MulticlassAUROC,
            BinaryAccuracy,
            BinaryF1Score,
            BinaryAUROC,
        )

        hyponet, queries_x, queries_y = self.get_hyponet(data)

        # logits: (B, Q, C)
        logits = hyponet(queries_x)

        # flatten episodic dims
        logits = einops.rearrange(logits, "b q c -> (b q) c")
        targets = einops.rearrange(queries_y, "b q -> (b q)").long()

        num_classes = logits.size(-1)

        # =======================
        # BINARY CLASSIFICATION
        # =======================
        if num_classes == 1 or num_classes == 2:
            if num_classes == 2:
                # use positive class logit
                logits_pos = logits[:, 1]
            else:
                logits_pos = logits.squeeze(1)

            probs = torch.sigmoid(logits_pos)
            preds = (probs >= 0.5).long()

            acc_metric = BinaryAccuracy()
            acc_metric.update(preds, targets)
            accuracy = acc_metric.compute().item()

            if acc_only:
                return {"acc": accuracy}

            f1_metric = BinaryF1Score()
            f1_metric.update(preds, targets)
            f1 = f1_metric.compute().item()

            auc_metric = BinaryAUROC()
            auc_metric.update(probs, targets)
            auc = auc_metric.compute().item()

            return {
                "acc": accuracy,
                "bal_acc": accuracy, # patch: simply return accuracy itself
                "f1": f1,
                "auc": auc,
            }

        # =======================
        # MULTICLASS CLASSIFICATION
        # =======================
        else:
            preds = torch.argmax(logits, dim=1)

            acc_metric = MulticlassAccuracy(num_classes=num_classes, average="micro")
            acc_metric.update(preds, targets)
            accuracy = acc_metric.compute().item()

            if acc_only:
                return {"acc": accuracy}

            f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")
            f1_metric.update(preds, targets)
            f1 = f1_metric.compute().item()

            auc_metric = MulticlassAUROC(num_classes=num_classes, average="macro")
            auc_metric.update(logits, targets)  # expects logits
            auc = auc_metric.compute().item()

            return {
                "acc": accuracy,
                "bal_acc": accuracy, # patch: simply return accuracy itself
                "f1": f1,
                "auc": auc,
            }
            
    # def compute_metrics(self, data, acc_only=False):
    #     """
    #     Compute accuracy, balanced accuracy, F1 score, and ROC-AUC 
    #     for binary classification using torcheval only.

    #     Parameters
    #     ----------
    #     predictions : torch.Tensor
    #         Raw model outputs (logits) of shape (N,) or (N, 1).
    #     targets : torch.Tensor
    #         Ground truth binary labels of shape (N,).

    #     Returns
    #     -------
    #     dict
    #         Dictionary containing accuracy, balanced accuracy, F1 score, and ROC-AUC.
    #     """
        
    #     hyponet, queries_x, queries_y = self.get_hyponet(data)
    #     predictions = einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class")
        
    #     targets = einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)")

    #     targets = targets.long().squeeze()

    #     # Handle [N, 2] case: take positive class logits
    #     if predictions.dim() == 2 and predictions.size(1) == 2:
    #         predictions = predictions[:, 1]

    #     predictions = predictions.squeeze()
    #     probs = torch.sigmoid(predictions)  # probs for positive class
    #     preds_binary = (probs >= 0.5).int()

    #     # Accuracy
    #     acc_metric = MulticlassAccuracy(num_classes=2, average="micro")
    #     acc_metric.update(preds_binary, targets)
    #     accuracy = acc_metric.compute().item()

    #     # F1 Score
    #     f1_metric = BinaryF1Score()
    #     f1_metric.update(preds_binary, targets)
    #     f1_score = f1_metric.compute().item()

    #     # ROC AUC
    #     rocauc_metric = BinaryAUROC()
    #     rocauc_metric.update(probs, targets)
    #     roc_auc = rocauc_metric.compute().item()

    #     # Recall for positive class (TPR / sensitivity)
    #     recall_pos = BinaryRecall()
    #     recall_pos.update(preds_binary, targets)
    #     tpr = recall_pos.compute().item()

    #     # Recall for negative class (TNR / specificity) by label swap
    #     recall_neg = BinaryRecall()
    #     recall_neg.update(1 - preds_binary, 1 - targets)
    #     tnr = recall_neg.compute().item()

    #     balanced_accuracy = (tpr + tnr) / 2.0

    #     if acc_only:
    #         return {"acc": accuracy}

    #     return {
    #         # "acc": accuracy,
    #         "bal_acc": balanced_accuracy,
    #         # "f1_score": f1_score,
    #         # "roc_auc": roc_auc,
    #     }

    def train_step(self, data):
        loss = self.compute_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics = self.compute_metrics(data)
        metrics["loss"] = loss.item()
        return metrics

    def evaluate_step(self, data):
        with torch.no_grad():
            loss = self.compute_loss(data)
            metrics = self.compute_metrics(data)
            metrics["loss"] = loss.item()
        # save the current best checkpoint (w.r.t. roc_auc)
        # if self.current_best_eval_acc < metrics["roc_auc"]:
        #     self.current_best_eval_acc = metrics["roc_auc"]
        #     self.save_checkpoint('epoch-best-rocauc.pth')
        if self.current_best_eval_balacc < metrics["bal_acc"]:
            self.current_best_eval_balacc = metrics["bal_acc"]
            self.save_checkpoint('epoch-best-balacc.pth')
        return metrics