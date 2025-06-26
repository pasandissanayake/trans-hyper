import torch
import argparse
import os

from datahandles import FewshotDataset
from utils import *
from trainers import trainers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name. If not provided, will use the cfg filename.')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--port-offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
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
    setattr(cfg.env, "exp_name", ConfigObject(exp_name)) # type: ignore
    setattr(cfg.env, "total_gpus", ConfigObject(torch.cuda.device_count())) # type: ignore
    setattr(cfg.env, "save_dir", ConfigObject(os.path.join(args.save_root, exp_name))) # type: ignore
    setattr(cfg.env, "wandb_upload", ConfigObject(args.wandb_upload)) # type: ignore
    setattr(cfg.env, "port", ConfigObject(str(29600 + args.port_offset))) # type: ignore
    setattr(cfg.env, "cudnn", ConfigObject(args.cudnn)) # type: ignore
   
    return cfg


def main():
    args = parse_args()
    cfg = make_cfg(args)

    if cfg.debug(): print('UNIVERSAL DEBUG MODE ENABLED') # type: ignore

    train_ds = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    test_ds = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    trainer = trainers[cfg.trainer.name()](0, cfg, train_ds, test_ds) # type: ignore
    
    trainer.run()


# ⚙️ Usage
if __name__ == "__main__":
    main()
    