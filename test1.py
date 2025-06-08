import argparse
import os

import torch
from torch.utils.data import DataLoader

from utils import *
from datasets import *

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

    setattr(cfg, 'env', ConfigObject())
    setattr(cfg.env, 'exp_name', ConfigObject(exp_name))
    setattr(cfg.env, 'total_gpus', ConfigObject(torch.cuda.device_count()))

    return cfg


def main():
    args = parse_args()
    cfg = make_cfg(args)

    if cfg.debug(): print('UNIVERSAL DEBUG MODE ENABLED')
    # combined_dataset = CombinedDataset(cfg, 'train')
    # print(combined_dataset[5009])

    fewshot_dataset = FewshotDataset(cfg, 'train', n_shots=3, n_queries=4)
    print(f"Fewshot dataset length: {len(fewshot_dataset)}")
    
    dataloader = DataLoader(fewshot_dataset, batch_size=2, shuffle=False)
    dl_iter = iter(dataloader)
    for i, batch in enumerate(dl_iter):
        # print(f"Batch {i}:\n{batch}")
        shots = batch['shots']
        queries = batch['queries']
        print(f"Batch {i}:\nShots:\n{shots}\n\nQueries:\n{queries}")
        if i >= 2:
            break

if __name__ == '__main__':
    main()