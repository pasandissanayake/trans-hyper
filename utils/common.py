import os
import shutil
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter


def check_any_substring(target_string, list_of_substrings):
    for substring in list_of_substrings:
        if substring in target_string:
            return True
    return False


def ensure_path(path, replace=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if replace and (basename.startswith('_') or input('{} exists, replace? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def dict_to_mlp(weight_dict: dict[str, torch.Tensor], in_dim:int) -> nn.Sequential:
    """
    Convert a dictionary of 'wbX' -> tensor(out_features, in_features+1)
    into a PyTorch MLP with the given weights and biases.
    
    Args:
        weight_dict: dict with keys like 'wb0', 'wb1', ... and tensors 
                     where the last column is the bias.
    
    Returns:
        model: nn.Sequential containing the layers with weights loaded.
    """
    layers = []
    
    # Sort layers by number (wb0, wb1, ...)
    sorted_keys = sorted(weight_dict.keys(), key=lambda k: int(k[2:]))
    
    for i, key in enumerate(sorted_keys):
        in_dim = in_dim + 1 # inputs and bias
        wb = weight_dict[key][0]
        out_dim = len(wb) // in_dim
        wb = torch.reshape(wb, (in_dim, out_dim))

        in_features = in_dim - 1  # last col = bias
        out_features = out_dim
        bias = wb[-1, :]
        weight = wb[:-1, :]
        
        # Create linear layer
        layer = nn.Linear(in_features, out_features)
        
        # Assign weights and bias (ensure no grad issues)
        with torch.no_grad():
            layer.weight.copy_(torch.transpose(weight, 0 ,1))
            layer.bias.copy_(bias)
        
        layers.append(layer)
        
        # Optionally add non-linearity (ReLU here, skip after last)
        if i < len(sorted_keys) - 1:
            layers.append(nn.ReLU())

        in_dim = out_dim
    
    return nn.Sequential(*layers)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'w')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    logger = set_logger(os.path.join(save_dir, 'log.txt'))
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    return logger, writer


def compute_num_params(model, text=True, trainable_only=False):
    if trainable_only:
        tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        elif tot >= 1e3:
            return '{:.1f}K'.format(tot / 1e3)
        else:
            return str(tot)
    else:
        return tot


def make_optimizer(params, cfg, sd=None):
    optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[cfg.trainer.optimizer.name()](params, **cfg.trainer.optimizer.args.to_dict())
    if sd is not None:
        optimizer.load_state_dict(sd)
    return optimizer


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(secs):
    if secs >= 3600:
        return f'{secs / 3600:.1f}h'
    elif secs >= 60:
        return f'{secs / 60:.1f}m'
    else:
        return f'{secs:.1f}s'