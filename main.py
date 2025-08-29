import torch
import argparse
import os
import yaml
import wandb
from datetime import datetime

from datahandles import MetaDatasetBuilder
from utils import Config, ConfigObject
from trainers import trainers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the config file.')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name. If not provided, will use the cfg filename.')
    parser.add_argument('--group', type=str, default=None,
                        help='Experiment group name for WandB.')
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
    setattr(cfg.env, "exp_group", ConfigObject(args.group)) # type: ignore
    setattr(cfg.env, "total_gpus", ConfigObject(torch.cuda.device_count())) # type: ignore
    setattr(cfg.env, "save_dir", ConfigObject(os.path.join(args.save_root, exp_name))) # type: ignore
    setattr(cfg.env, "wandb_upload", ConfigObject(args.wandb_upload)) # type: ignore
    setattr(cfg.env, "port", ConfigObject(str(29600 + args.port_offset))) # type: ignore
    setattr(cfg.env, "cudnn", ConfigObject(args.cudnn)) # type: ignore
   
    return cfg

def adopt_wandb_cfg(cfg, wandb_cfg):
    # cfg.trainer.optimizer.args.lr(wandb_cfg.learning_rate)
    # cfg.trainer.batch_size(wandb_cfg.batch_size)
    # cfg.datasets.n_shots(wandb_cfg.n_shots)
    return cfg

def train(cfg:Config, sweep:bool):
    if cfg.env.wandb_upload():
        wandb_name = os.environ["WANDB_NAME"]
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        wandb.init(name=f"{wandb_name}-{timestamp}", group=cfg.env.exp_group())
    if sweep:
        cfg = adopt_wandb_cfg(cfg, wandb.config)

    meta_datasets = MetaDatasetBuilder(
        data_root=cfg.datasets.data_root(),
        train_datasets=cfg.datasets.list_combine_train(),
        val_datasets=cfg.datasets.list_combine_val(),
        test_datasets=cfg.datasets.list_combine_test(),
        train_size=cfg.datasets.train_size(),
        val_size=cfg.datasets.val_size(),
        test_size=cfg.datasets.test_size(),
        train_permutation=cfg.datasets.train_permutation(),
        val_permutation=cfg.datasets.val_permutation(),
        test_permutation=cfg.datasets.test_permutation(),
        train_balance=cfg.datasets.balanced.train(),
        val_balance=cfg.datasets.balanced.val(),
        test_balance=cfg.datasets.balanced.test(),
        n_shots=cfg.datasets.n_shots(),
        n_queries=cfg.datasets.n_queries(),
        shuffle=True,
        max_n_features=103,
        queries_same_as_shots=cfg.datasets.queries_same_as_shots(),
        debug=cfg.debug() or cfg.debug_datasets()
    ).get_datasets()
    
    train_ds = meta_datasets['train']
    test_ds = meta_datasets['val']

    trainer = trainers[cfg.trainer.name()](0, cfg, train_ds, test_ds) # type: ignore
    
    trainer.run()

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    cfg = make_cfg(args)

    if cfg.debug(): print('UNIVERSAL DEBUG MODE ENABLED') # type: ignore

    if cfg.env.wandb_upload():
        with open(cfg.wandb_auth(), 'r') as f:
            wandb_auth = yaml.load(f, Loader=yaml.FullLoader)
        os.environ['WANDB_DIR'] = cfg.env.save_dir()
        os.environ['WANDB_NAME'] = cfg.env.exp_name()
        os.environ['WANDB_API_KEY'] = wandb_auth['api_key']

        if cfg.wandb_sweep_cfg():
            with open(cfg.wandb_sweep_cfg(), 'r') as f:
                sweep_cfg = yaml.load(f, Loader=yaml.FullLoader)
            sweep_id = wandb.sweep(sweep_cfg, project=wandb_auth['project'])
            def train_wrapper():
                train(cfg, sweep=True)
            wandb.agent(sweep_id, train_wrapper, count=5)
        else:
            train(cfg=cfg, sweep=False)
    else:
        train(cfg=cfg, sweep=False)
    

if __name__ == "__main__":
    main()
    