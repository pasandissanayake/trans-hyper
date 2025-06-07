import argparse
import os

import torch

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
    data = AdultDataset(cfg)

    print(data.test.columns)


if __name__ == '__main__':
    main()