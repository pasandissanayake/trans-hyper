import torch
import torch.nn as nn

import models
from trainers import BaseTrainer
from trainers import register
import einops

TRAINER_NAME = "bert_trainer"

@register(TRAINER_NAME)
class BertTrainer(BaseTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)
        self.name = TRAINER_NAME

    def compute_loss(self, data):
        tokenizer = models.make(model_name=self.cfg.tokenizer.model(), cfg=self.cfg, sd=None)
        shots = data['shots']
        shots = tokenizer(shots)
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
    
    def compute_accuracy(self, data):
        tokenizer = models.make(model_name=self.cfg.tokenizer.model(), cfg=self.cfg, sd=None)
        shots = data['shots']
        shots = tokenizer(shots)
        input_ids = shots['input_ids'].cuda()
        attention_mask = shots['attention_mask'].cuda()
        queries_x = data['queries_x'].cuda()
        queries_y = data['queries_y'].cuda()

        hyponet = self.model_ddp({'input_ids': input_ids, 'attention_mask': attention_mask})

        return self.accuracy(einops.rearrange(hyponet(queries_x), "batch n_queries n_class -> (batch n_queries) n_class"),
                         einops.rearrange(queries_y, "batch n_queries -> (batch n_queries)"))

    def evaluate_step(self, data):
        with torch.no_grad():
            loss = self.compute_loss(data)
            acc = self.compute_accuracy(data)
        return {'loss': loss.item(), 'acc': acc}