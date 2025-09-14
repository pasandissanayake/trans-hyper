import torch
import torch.nn as nn

import models
from .bert_cv_trainer import BertCVTrainer
from .trainers import register
import einops

TRAINER_NAME = "t0_cv_trainer"

@register(TRAINER_NAME)
class T0Trainer(BertCVTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)