import torch
import einops
import numpy as np

device = "cpu"

a = torch.ones((3,2,3), device=device)
b = torch.zeros((3,2,1), device=device)

c, _ = einops.pack([a, b], "batch n_queries *")

print(c.shape)

