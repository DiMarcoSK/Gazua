import argparse
import json
from pathlib import Path

import torch

from gazua.ablation import ablate_direction, orthogonalize_
from gazua.data import load_splits
from gazua.evaluate import next_token_logprobs
from gazua.model import load

p = argparse.ArgumentParser(description="Orthogonalize weights against the chosen direction and save the model.")
p.add_argument("--directions", default="artifacts/directions.pt")
p.add_argument("--sweep", default="artifacts/layer_sweep.json")
p.add_argument("--layer", type=int, help="defaults to best_layer from the sweep")
p.add_argument("--out-dir", default=None)
args = p.parse_args()

saved = torch.load(args.directions)
layer = args.layer
if layer is None:
    layer = json.loads(Path(args.sweep).read_text(encoding="utf-8"))["best_layer"]
r = saved["directions"][layer]

model, tok = load(saved["model"])
check = load_splits(n_test=8)["harmless_test"]

with ablate_direction(model, r):
    logp_hooks = next_token_logprobs(model, tok, check)
orthogonalize_(model, r)
logp_weights = next_token_logprobs(model, tok, check)

diff = (logp_hooks - logp_weights).abs().max().item()
print(f"max |hooks - weights| (log-probs): {diff:.2e}")
if diff > 1e-2:
    raise SystemExit("hooks and weights disagree: orthogonalization is wrong")

name = saved["model"].split("/")[-1]
out_dir = Path(args.out_dir or f"models/{name}-abliterated-L{layer}")
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print(f"model saved to {out_dir}")
