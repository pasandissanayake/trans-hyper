import yaml
import os

class ConfigObject:
    def __init__(self, val=None) -> None:
        if val is None:
            self.val = None
        else:
            self.val = val

    def __call__(self):
        return self.val
    
    def __str__(self):  
        return str(self.val)
    
    def __repr__(self):
        return f"{self.val}"


class Config:
    def __init__(self, cfg_file, debug=False) -> None:
        with open(cfg_file, 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        def translate_cfg(obj, d):
            for k, v in d.items():
                setattr(obj, k, ConfigObject())
                if isinstance(v, dict):
                    if debug: print(f"Translating config key: {k}")
                    translate_cfg(getattr(obj, k), v)
                else:
                    if debug: print(f"Setting config key: {k} with value: {v}")
                    setattr(obj, k, ConfigObject(v))
        translate_cfg(self, cfg_dict)

    