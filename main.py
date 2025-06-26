import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from torch.optim import AdamW
import torch.nn.functional as F

from datahandles import *
from utils import *
from models import *
import trainers

import argparse
import os
import einops

from utils import ConfigObject

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name. If not provided, will use the cfg filename.')
    args = parser.parse_args()

    return args


def make_cfg(args):
    cfg = Config(args.cfg)

    if args.name is None:
            exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name

    if args.tag is not None:
        exp_name += '_' + args.tag

    setattr(cfg, "env", ConfigObject())
    setattr(cfg.env, "exp_name", ConfigObject(exp_name))
    setattr(cfg.env, "total_gpus", ConfigObject(torch.cuda.device_count()))
    setattr(cfg.env, "save_dir", ConfigObject(os.path.join(args.save_root, exp_name)))
    setattr(cfg.env, "wandb_upload", ConfigObject(args.wandb_upload))
    setattr(cfg.env, "port", ConfigObject(str(29600 + args.port_offset)))
    setattr(cfg.env, "cudnn", ConfigObject(args.cudnn))
   
    return cfg


def main():
    args = parse_args()
    cfg = make_cfg(args)

    if cfg.debug(): print('UNIVERSAL DEBUG MODE ENABLED')

    train_ds = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    test_ds = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    trainer = trainers.BaseTrainer(0, cfg, train_ds, test_ds)
    
    trainer.run()


# ⚙️ Usage
if __name__ == "__main__":
    main()
    