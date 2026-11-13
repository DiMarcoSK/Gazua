import argparse
import json
from pathlib import Path

import torch

from gazua.ablation import ablate_direction
from gazua.data import load_splits
from gazua.evaluate import generate, mean_kl, next_token_logprobs, refusal_rate
from gazua.model import load

p = argparse.ArgumentParser(description="Sweep candidate layers: refusal rate vs. KL on harmless prompts.")
p.add_argument("--directions", default="artifacts/directions.pt")
p.add_argument("--n-test", type=int, default=16)
p.add_argument("--start", type=float, default=0.2, help="sweep start, as a fraction of depth")
p.add_argument("--end", type=float, default=0.8, help="sweep end, as a fraction of depth")
p.add_argument("--max-kl", type=float, default=0.1)
p.add_argument("--refusal-tol", type=float, default=0.05, help="slack above the lowest refusal rate")
p.add_argument("--out", default="artifacts/layer_sweep.json")
args = p.parse_args()

saved = torch.load(args.directions)
dirs = saved["directions"]
n_layers = dirs.shape[0]
model, tok = load(saved["model"])

splits = load_splits(n_test=args.n_test)
harmful, harmless = splits["harmful_test"], splits["harmless_test"]

base_refusal = refusal_rate(generate(model, tok, harmful))
base_logp = next_token_logprobs(model, tok, harmless)
print(f"baseline: refusal={base_refusal:.0%}")

results = []
for layer in range(int(args.start * n_layers), int(args.end * n_layers) + 1):
    with ablate_direction(model, dirs[layer]):
        completions = generate(model, tok, harmful)
        kl = mean_kl(base_logp, next_token_logprobs(model, tok, harmless))
    rr = refusal_rate(completions)
    results.append({"layer": layer, "refusal_rate": rr, "kl": kl, "completions": completions})
    print(f"layer {layer:3d}: refusal={rr:5.0%}  KL={kl:.4f}")

floor = min(r["refusal_rate"] for r in results)
best = min((r for r in results if r["refusal_rate"] <= floor + args.refusal_tol), key=lambda r: r["kl"])
print(f"\nbest layer: {best['layer']} (refusal={best['refusal_rate']:.0%}, KL={best['kl']:.4f})")
if best["kl"] > args.max_kl:
    print(f"warning: KL above {args.max_kl}; the model may be noticeably degraded")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(
    json.dumps(
        {"model": saved["model"], "baseline_refusal": base_refusal, "best_layer": best["layer"], "sweep": results},
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"saved to {args.out}")
