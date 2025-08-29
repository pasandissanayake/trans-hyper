import torch
import torch.nn as nn

import models
from trainers import BaseTrainer
from trainers import register
import einops
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
import numpy as np

TRAINER_NAME = "bert_trainer"

@register(TRAINER_NAME)
class BertTrainer(BaseTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)
        self.name = TRAINER_NAME
        self.tokenizer = models.make(model_name=self.cfg.tokenizer.name(), cfg=self.cfg, sd=None)
        self.log(f"Number of shots: {cfg.datasets.n_shots()}")
        self.log(f"Number of queries: {cfg.datasets.n_queries()}")

        self.current_best_eval_acc = 0
        self.current_best_eval_balacc = 0

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
        accuracy = 100 * correct / total
        return accuracy
    
    def compute_metrics(self, data, acc_only=False):
        shots = data['shots']
        shots = self.tokenizer(shots)
        input_ids = shots['input_ids'].cuda()
        attention_mask = shots['attention_mask'].cuda()
        queries_x = data['queries_x'].cuda()
        queries_y = data['queries_y'].cuda()

        hyponet = self.model_ddp({'input_ids': input_ids, 'attention_mask': attention_mask})
        predictions = einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class")
        predictions = predictions.cpu().detach().numpy()

        y_pred = np.argmax(predictions, axis=1)
        y_true = einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)").cpu().detach().numpy()
        
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        roc_auc = roc_auc_score(y_true=y_true, y_score=predictions[:, 1])

        if acc_only:
            return {'acc': acc}
        else:
            return {
                'acc': acc,
                'bal_acc': bal_acc,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
    
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
        if self.current_best_eval_acc <= metrics["acc"]:
            self.current_best_eval_acc = metrics["acc"]
            self.save_checkpoint('epoch-best-acc.pth')
        if self.current_best_eval_balacc <= metrics["bal_acc"]:
            self.current_best_eval_balacc = metrics["bal_acc"]
            self.save_checkpoint('epoch-best-balacc.pth')
        return metrics