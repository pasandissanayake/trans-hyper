import torch
import torch.nn as nn

import models
from trainers import BertTrainer
from trainers import register
import einops
import os.path as osp

TRAINER_NAME = "t0_trainer"

@register(TRAINER_NAME)
class T0Trainer(BertTrainer):

    def __init__(self, rank, cfg, train_ds=None, test_ds=None):
        super().__init__(rank=rank, cfg=cfg, train_ds=train_ds, test_ds=test_ds)

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_state = self.model.regressor.state_dict()
        optimizer_state = self.optimizer.state_dict()
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg.env.save_dir, filename))