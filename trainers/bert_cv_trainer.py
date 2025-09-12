import torch
import torch.nn as nn

import models
from .base_cv_trainer import BaseCVTrainer
from .trainers import register
import einops
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from torcheval.metrics import MulticlassAccuracy, BinaryF1Score, BinaryAUROC, BinaryRecall
import numpy as np

TRAINER_NAME = "bert_cv_trainer"

@register(TRAINER_NAME)
class BertCVTrainer(BaseCVTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)
        self.name = TRAINER_NAME
        self.tokenizer = models.make(model_name=self.cfg.tokenizer.name, cfg=self.cfg, sd=None)
        self.current_best_eval_acc = 0

    def compute_loss(self, data):
        shots = data['shots']
        shots = self.tokenizer(shots)
        input_ids = shots['input_ids'].cuda()
        attention_mask = shots['attention_mask'].cuda()
        queries_x = data['queries_x'].cuda()
        queries_y = data['queries_y'].cuda()

        hyponet = self.model_ddp({'input_ids': input_ids, 'attention_mask': attention_mask})
        criterion = nn.CrossEntropyLoss()
        loss = criterion(einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class"),
                         einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)"))
        return loss
    
    def accuracy(self, outputs, labels):
        """
        Calculates accuracy for a multiclass classification problem.

        Args:
            outputs (torch.Tensor): Model outputs (logits or probabilities).
                                    Shape: (batch_size, num_classes)
            labels (torch.Tensor): True labels. Shape: (batch_size,)

        Returns:
            float: Accuracy as a percentage.
        """
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy
        
    def compute_metrics(self, data, acc_only=False):
        """
        Compute accuracy, balanced accuracy, F1 score, and ROC-AUC 
        for binary classification using torcheval only.

        Parameters
        ----------
        predictions : torch.Tensor
            Raw model outputs (logits) of shape (N,) or (N, 1).
        targets : torch.Tensor
            Ground truth binary labels of shape (N,).

        Returns
        -------
        dict
            Dictionary containing accuracy, balanced accuracy, F1 score, and ROC-AUC.
        """
        shots = data['shots']
        shots = self.tokenizer(shots)
        input_ids = shots['input_ids'].cuda()
        attention_mask = shots['attention_mask'].cuda()
        queries_x = data['queries_x'].cuda()
        queries_y = data['queries_y'].cuda()

        hyponet = self.model_ddp({'input_ids': input_ids, 'attention_mask': attention_mask})
        predictions = einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class")
        
        targets = einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)")

        targets = targets.long().squeeze()

        # Handle [N, 2] case: take positive class logits
        if predictions.dim() == 2 and predictions.size(1) == 2:
            predictions = predictions[:, 1]

        predictions = predictions.squeeze()
        probs = torch.sigmoid(predictions)  # probs for positive class
        preds_binary = (probs >= 0.5).int()

        # Accuracy
        acc_metric = MulticlassAccuracy(num_classes=2, average="micro")
        acc_metric.update(preds_binary, targets)
        accuracy = acc_metric.compute().item()
        return {"acc": accuracy}

    def train_step(self, data):
        loss = self.compute_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics = self.compute_metrics(data, acc_only=True)
        metrics["loss"] = loss.item()
        return metrics

    def evaluate_step(self, data):
        with torch.no_grad():
            loss = self.compute_loss(data)
            metrics = self.compute_metrics(data)
            metrics["loss"] = loss.item()
        # save the current best checkpoint (w.r.t. accuracy)
        if self.current_best_eval_acc < metrics["acc"]:
            self.current_best_eval_acc = metrics["acc"]
            self.save_checkpoint(f'epoch-best-acc-fold{self.current_fold}.pth')
        return metrics