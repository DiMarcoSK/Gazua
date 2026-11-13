from __future__ import annotations

import torch
from tqdm import tqdm

from .model import batches, tokenize


@torch.no_grad()
def mean_resid_last_token(model, tok, prompts: list[str], batch_size: int = 16) -> torch.Tensor:
    """Mean residual stream entering each layer at the last prompt token: [n_layers, d_model]."""
    total, n = None, 0
    for b in tqdm(list(batches(prompts, batch_size)), desc="activations", leave=False):
        enc = tokenize(tok, b, model.device)
        out = model(**enc, output_hidden_states=True)
        # the final hidden state is post-norm, so it is dropped
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states[:-1]]).float()
        s = hs.sum(dim=1)
        total = s if total is None else total + s
        n += hs.shape[1]
    return total / n


def refusal_directions(mean_harmful: torch.Tensor, mean_harmless: torch.Tensor) -> torch.Tensor:
    d = mean_harmful - mean_harmless
    return d / d.norm(dim=-1, keepdim=True)
