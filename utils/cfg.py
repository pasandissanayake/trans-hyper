import yaml
import os

class ConfigObject:
    def __init__(self, val=None) -> None:
        if val is None:
            self.val = None
        else:
            self.val = val

    def __call__(self, val=None):
        if val is not None:
            self.val = val
        else:
            return self.val
    
    def __str__(self):  
        return str(self.val)
    
    def __repr__(self):
        return f"{self.val}"
    
    def to_dict(self):
        val_dict = {}
        def unwrap(obj, obj_name, parent_dict):
            if isinstance(obj, ConfigObject):
                if obj.val is None:
                    attr_list = list(obj.__dict__.keys())
                    attr_list.remove("val")
                    parent_dict[obj_name] = {}
                    for attr in attr_list:
                        unwrap(getattr(obj, attr), attr, parent_dict[obj_name])
                else:
                    parent_dict[obj_name] = obj.val
            else:
                raise ValueError("Config object parse error")
                
        unwrap(self, None, val_dict)
        return val_dict[None]


class Config:
    def __init__(self, cfg_file=None, cfg_dict=None, debug=False) -> None:
        def translate_cfg(obj, d):
            for k, v in d.items():
                setattr(obj, k, ConfigObject())
                if isinstance(v, dict):
                    if debug: print(f"Translating config key: {k}")
                    translate_cfg(getattr(obj, k), v)
                else:
                    if debug: print(f"Setting config key: {k} with value: {v}")
                    setattr(obj, k, ConfigObject(v))

        if cfg_file is not None:
            with open(cfg_file, 'r') as f:
                cfg_dict = yaml.load(f, Loader=yaml.FullLoader)        
            translate_cfg(self, cfg_dict)
        elif cfg_dict is not None:
            translate_cfg(self, cfg_dict)
        else:
            raise ValueError("Config init error. Both cfg_file and cfg_dict are None")
        
    def to_dict(self):
        val_dict = {}
        def unwrap(obj, obj_name, parent_dict):
            if isinstance(obj, ConfigObject):
                if obj.val is None:
                    attr_list = list(obj.__dict__.keys())
                    attr_list.remove("val")
                    parent_dict[obj_name] = {}
                    for attr in attr_list:
                        unwrap(getattr(obj, attr), attr, parent_dict[obj_name])
                else:
                    parent_dict[obj_name] = obj.val
            elif isinstance(obj, Config):
                attr_list = list(obj.__dict__.keys())
                for attr in attr_list:
                    unwrap(getattr(obj, attr), attr, parent_dict)
            else:
                raise ValueError("Config object parse error")
                
        unwrap(self, None, val_dict)
        return val_dict

