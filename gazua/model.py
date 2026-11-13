from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_id: str = DEFAULT_MODEL, device: str | None = None, dtype=torch.float32):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"  # keeps the last prompt token at position -1
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, tok


def format_prompts(tok, prompts: list[str]) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def tokenize(tok, prompts: list[str], device):
    texts = format_prompts(tok, prompts)
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def batches(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
