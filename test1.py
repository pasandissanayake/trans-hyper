import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import *
from datahandles import *

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

    fewshot_dataset = FewshotDataset(cfg, 'train', n_shots=3, n_queries=5)
    print(f"Fewshot dataset length: {len(fewshot_dataset)}")
    
    dataloader = DataLoader(fewshot_dataset, batch_size=2, shuffle=True)
    dl_iter = iter(dataloader)
    for i, batch in enumerate(dl_iter):
        shots = batch['shots']
        queries_x = np.array(batch['queries_x'])
        queries_y = np.array(batch['queries_y'])

        print(f"Batch {i}:\nShots:\n{len(shots)}\n\nQueries_x:\n{queries_x.shape}\n\nQueries_y:\n{queries_y.shape}\n")
        if i >= 0:
            break

if __name__ == '__main__':
    main()