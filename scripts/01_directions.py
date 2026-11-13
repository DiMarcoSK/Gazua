import argparse
from pathlib import Path

import torch

from gazua.data import load_splits
from gazua.directions import massive_dims, mean_resid_last_token, refusal_directions
from gazua.model import DEFAULT_MODEL, load

p = argparse.ArgumentParser(description="Collect activations and compute per-layer refusal directions.")
p.add_argument("--model", default=DEFAULT_MODEL)
p.add_argument("--n-train", type=int, default=128)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--keep-massive", action="store_true", help="do not zero massive-activation dims in r")
p.add_argument("--out", default="artifacts/directions.pt")
args = p.parse_args()

splits = load_splits(n_train=args.n_train)
model, tok = load(args.model)

mu_harmful = mean_resid_last_token(model, tok, splits["harmful_train"], args.batch_size)
mu_harmless = mean_resid_last_token(model, tok, splits["harmless_train"], args.batch_size)

massive = torch.tensor([], dtype=torch.long)
if not args.keep_massive:
    massive = massive_dims(model, tok, splits["harmless_train"][:32])
    print(f"massive-activation dims excluded from r: {massive.tolist()}")
dirs = refusal_directions(mu_harmful, mu_harmless, exclude_dims=massive)

gap = (mu_harmful - mu_harmless).norm(dim=-1)
print("layer | ||mu_harmful - mu_harmless||")
for layer, g in enumerate(gap.tolist()):
    print(f"{layer:5d} | {g:8.3f}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": args.model,
        "directions": dirs,
        "massive_dims": massive,
        "mu_harmful": mu_harmful,
        "mu_harmless": mu_harmless,
    },
    args.out,
)
print(f"saved to {args.out}")
