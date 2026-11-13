import argparse
import json
import math
from pathlib import Path

import torch

from gazua.data import load_splits
from gazua.evaluate import generate, mean_kl, next_token_logprobs, refusal_rate, response_nll
from gazua.model import load
from gazua.report import checks, render, verdict

p = argparse.ArgumentParser(description="Compare base vs. abliterated model on held-out prompts.")
p.add_argument("--base", required=True)
p.add_argument("--abliterated", required=True)
p.add_argument("--n-eval", type=int, default=64)
p.add_argument("--max-new-tokens", type=int, default=64)
p.add_argument("--out", default="artifacts/eval.json")
args = p.parse_args()

splits = load_splits(n_eval=args.n_eval)
harmful, harmless = splits["harmful_eval"], splits["harmless_eval"]

base, tok = load(args.base)
base_harmful = generate(base, tok, harmful, args.max_new_tokens)
base_harmless = generate(base, tok, harmless, args.max_new_tokens)
base_logp = next_token_logprobs(base, tok, harmless)
base_nll = response_nll(base, tok, harmless, base_harmless)
del base

abl, _ = load(args.abliterated)
abl_harmful = generate(abl, tok, harmful, args.max_new_tokens)
abl_harmless = generate(abl, tok, harmless, args.max_new_tokens)
kl = mean_kl(base_logp, next_token_logprobs(abl, tok, harmless))
abl_nll = response_nll(abl, tok, harmless, base_harmless)

metrics = {
    "harmful_refusal": {"base": refusal_rate(base_harmful), "abliterated": refusal_rate(abl_harmful)},
    "harmless_refusal": {"base": refusal_rate(base_harmless), "abliterated": refusal_rate(abl_harmless)},
    "harmless_kl": kl,
    "harmless_ppl_on_base_responses": {"base": math.exp(base_nll), "abliterated": math.exp(abl_nll)},
}

print(f"{'metric':34s} {'base':>10s} {'abliterated':>12s}")
for k in ("harmful_refusal", "harmless_refusal"):
    print(f"{k:34s} {metrics[k]['base']:>10.0%} {metrics[k]['abliterated']:>12.0%}")
ppl = metrics["harmless_ppl_on_base_responses"]
print(f"{'harmless ppl (base responses)':34s} {ppl['base']:>10.3f} {ppl['abliterated']:>12.3f}")
print(f"{'harmless next-token KL':34s} {'':>10s} {kl:>12.4f}")

results = checks(metrics)
status, _ = verdict(results)
print(render(results))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(
    json.dumps(
        {
            "base": args.base,
            "abliterated": args.abliterated,
            "verdict": status,
            "checks": [c.__dict__ for c in results],
            "metrics": metrics,
            "samples": {
                "harmless": [
                    {"prompt": q, "base": a, "abliterated": b}
                    for q, a, b in zip(harmless, base_harmless, abl_harmless)
                ],
                "harmful_refusal_flags": {
                    "base": [refusal_rate([c]) == 1 for c in base_harmful],
                    "abliterated": [refusal_rate([c]) == 1 for c in abl_harmful],
                },
            },
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print(f"saved to {args.out}")
