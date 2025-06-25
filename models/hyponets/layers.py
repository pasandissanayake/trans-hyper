import torch
import einops


def batched_linear_mm(x, wb):
    # args shapes --> x: (batch, n_queries, D1); wb: (batch, (D1 + 1) x D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    x, _ = einops.pack([x, one], "batch n_queries *")
    wb = einops.rearrange(wb, "batch (in_dim out_dim) -> batch in_dim out_dim", in_dim=x.shape[2])
    wb = einops.repeat(wb, "batch in_dim out_dim -> batch n_queries in_dim out_dim", n_queries=x.shape[1])
    return einops.einsum(x, wb, "batch n_queries in_dim, batch n_queries in_dim out_dim -> batch n_queries out_dim")