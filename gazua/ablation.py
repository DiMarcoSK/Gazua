"""Directional ablation, either via forward hooks or baked into the weights.

Exact for Llama/Qwen-style blocks only: Gemma-2/3 apply norms after o_proj/down_proj.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


def _project_out(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r = r.to(device=x.device, dtype=x.dtype)
    return x - (x @ r).unsqueeze(-1) * r


def writer_modules(model) -> list[nn.Module]:
    mods: list[nn.Module] = [model.get_input_embeddings()]
    for layer in model.model.layers:
        mods += [layer.self_attn.o_proj, layer.mlp.down_proj]
    return mods


@contextmanager
def ablate_direction(model, r: torch.Tensor):
    r = r / r.norm()
    hook = lambda _mod, _inp, out: _project_out(out, r)  # noqa: E731
    handles = [m.register_forward_hook(hook) for m in writer_modules(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def orthogonalize_(model, r: torch.Tensor) -> None:
    emb = model.get_input_embeddings()

    # untie so lm_head is not orthogonalized along with the embedding
    if getattr(model.config, "tie_word_embeddings", False):
        model.lm_head.weight = nn.Parameter(emb.weight.detach().clone())
        model.config.tie_word_embeddings = False

    r = (r / r.norm()).to(device=emb.weight.device, dtype=emb.weight.dtype)

    W = emb.weight
    W -= (W @ r).unsqueeze(-1) * r

    for m in writer_modules(model)[1:]:
        W = m.weight
        W -= r.unsqueeze(-1) * (r @ W).unsqueeze(0)
        if m.bias is not None:
            m.bias -= (m.bias @ r) * r
