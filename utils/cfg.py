import yaml
import os
from munch import Munch, munchify


def load_cfg(cfg_file=None, cfg_dict=None) -> Munch:
    cfg = Munch()
    if cfg_file is not None:
        with open(cfg_file, 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)        
            cfg = munchify(cfg_dict)
    elif cfg_dict is not None:
        cfg = munchify(cfg_dict)
    else:
        raise ValueError("Config init error. Both cfg_file and cfg_dict are None")
    return cfg
