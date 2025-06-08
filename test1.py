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
    combined_dataset = CombinedDataset(cfg, 'test')
    print(combined_dataset[5009])
    
    dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=False)

    # Iterate through the DataLoader
    for data in dataloader:
        print(f"Batch data shape: {data['input'].shape}, {data['label'].shape}")
        # Here you can add your processing logic for each batch
        # For example, you could pass the batch to a model for inference or training
        # For demonstration, we will just print the first few elements


if __name__ == '__main__':
    main()