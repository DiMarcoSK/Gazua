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


@torch.no_grad()
def massive_dims(model, tok, prompts: list[str], ratio: float = 50.0, batch_size: int = 8) -> torch.Tensor:
    """Hidden dims carrying massive activations (attention-sink features).

    Ablating any component of r along these dims shifts the sink token by orders of
    magnitude more than any other position, which wrecks attention everywhere.
    """
    peak = None
    for b in batches(prompts, batch_size):
        enc = tokenize(tok, b, model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[1:-1]
        mask = enc["attention_mask"].bool()
        p = torch.stack([h.abs()[mask].amax(0) for h in hs]).amax(0).float()
        peak = p if peak is None else torch.maximum(peak, p)
    return torch.nonzero(peak > ratio * peak.median()).flatten().cpu()


def refusal_directions(
    mean_harmful: torch.Tensor, mean_harmless: torch.Tensor, exclude_dims: torch.Tensor | None = None
) -> torch.Tensor:
    d = mean_harmful - mean_harmless
    if exclude_dims is not None and len(exclude_dims):
        d = d.clone()
        d[:, exclude_dims] = 0
    return d / d.norm(dim=-1, keepdim=True)
