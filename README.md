# Gazua

A study of *abliteration*: finding and removing the refusal direction of a small
open-weight LLM, following Arditi et al. (2024),
[*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717).

Default model: `Qwen/Qwen2.5-0.5B-Instruct` (runs on CPU).

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Pipeline

```powershell
python scripts/01_directions.py    # activations + per-layer refusal direction
python scripts/02_select_layer.py  # layer sweep: refusal rate vs. KL
python scripts/03_export.py        # orthogonalize weights, save to models/
```

| File | Purpose |
|---|---|
| `gazua/data.py` | AdvBench (refused prompts) and Alpaca (harmless prompts) |
| `gazua/directions.py` | mean residual stream at the last token; `r = μ_harmful − μ_harmless` |
| `gazua/ablation.py` | removes `r` via forward hooks or directly from the weights |
| `gazua/evaluate.py` | refusal rate (substring match) and KL on harmless prompts |

## Method

1. Run harmful and harmless prompts through the model and record the residual
   stream entering each layer at the last prompt token.
2. The difference of means between the two sets, normalized, is the candidate
   refusal direction `r` for that layer.
3. For each candidate layer, ablate `r` everywhere and measure the refusal rate
   on held-out harmful prompts and the KL divergence on harmless ones.
4. Bake the best direction into the weights.

Only three places write into the residual stream: the embedding, attention
(`o_proj`) and the MLP (`down_proj`). If none of them can write along `r`, the
model never represents `r`. Hook ablation (`x ← x − (x·r) r`) and weight
orthogonalization (`W ← (I − r rᵀ) W`) are therefore equivalent, and step 3
checks this numerically.

## Notes

- The resulting model has its safeguards removed. Keep it local and use it for study only.
- Substring-based refusal detection is crude; inspect the completions stored in
  `artifacts/layer_sweep.json`.
- Gemma-2/3 apply norms after `o_proj`/`down_proj`, so `ablation.py` would need
  to hook the norm outputs instead.
