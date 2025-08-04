import torch
import torch.nn as nn

import models
from trainers import BertTrainer
from trainers import register
import einops

TRAINER_NAME = "t0_trainer"

@register(TRAINER_NAME)
class T0Trainer(BertTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)