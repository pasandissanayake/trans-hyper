import torch.nn as nn

from models import register
from .layers import batched_linear_mm


HYPONET_NAME = "hypo_mlp"

@register(HYPONET_NAME)
class HypoMlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = HYPONET_NAME
        self.cfg = cfg
        self.hyponet_cfg = getattr(self.cfg.hyponet, self.name)
        self.debug = self.cfg.debug() or self.cfg.debug_hyponet()

        self.depth = self.hyponet_cfg.depth()
        self.in_dim = self.hyponet_cfg.in_dim()
        self.out_dim = self.hyponet_cfg.out_dim()
        self.hidden_dim = self.hyponet_cfg.hidden_dim()
        
        # create parameter shapes dict()
        self.param_shapes = dict()
        for i in range(self.depth):
            d1 = self.hidden_dim + 1 if i > 0 else self.in_dim + 1
            d2 = self.hidden_dim if i < self.depth - 1 else self.out_dim
            self.param_shapes[f'wb{i}'] = (d1, d2)

        self.relu = nn.ReLU()
        self.params = {}

        if self.debug:
            print(f"Hyponet {self.name} initialized with parameter shapes {self.param_shapes}")

    def set_params(self, params):
        self.params = params

    def forward(self, x):
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
        return x