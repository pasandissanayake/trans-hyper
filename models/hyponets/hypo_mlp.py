import torch
import torch.nn as nn
import einops

from models import register
from .layers import batched_linear_mm


@register('hypo_mlp')
class HypoMlp(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.depth = depth
        self.param_shapes = dict()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # create parameter shapes dict()
        for i in range(depth):
            d1 = hidden_dim + 1 if i > 0 else self.in_dim + 1
            d2 = hidden_dim if i < depth - 1 else self.out_dim
            self.param_shapes[f'wb{i}'] = (d1, d2)

        self.relu = nn.ReLU()
        self.params = {}

    def set_params(self, params):
        self.params = params

    def forward(self, x):
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
        return x