# Abliteration: state of the art (Nov 2026)

Research notes behind [BACKLOG.md](BACKLOG.md). Claims marked *(self-reported)* come from
model cards, blogs or single unreviewed preprints and still need reproduction.

## 1. Core method

**Arditi et al. 2024** ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) showed that refusal
is mediated by one direction in the residual stream:

- `r = μ_harmful − μ_harmless`, computed per layer **and per post-instruction token position**
  (the chat-template tokens after the user message), not only at the last token.
- Candidates are filtered and ranked: ablating `r` must lower the refusal score, *adding* `r` on
  harmless prompts must induce refusal, KL on harmless prompts must stay under 0.1, and layers past
  about 80% of depth are excluded.
- Ablation `x ← x − (x·r̂)r̂` is applied to every residual writer. Baking it into the weights
  (`W ← (I − r̂r̂ᵀ)W`) makes it permanent.

Gazua already implements this, with two gaps: only the last token position is used, and the
"induce refusal" check is missing.

## 2. Refinements to the direction and the weight edit

| Technique | Idea | Source |
|---|---|---|
| Projected abliteration | Remove from `r` its component along the harmless mean, so only the part orthogonal to "normal helpfulness" is ablated | [grimjim, Oct 2025](https://huggingface.co/blog/grimjim/projected-abliteration) |
| Norm-preserving biprojected / MPOA | After the edit, rescale each weight row back to its original norm. Biprojection also measures `r` at one layer and removes it at others | [grimjim, Nov 2025](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration), [llm-abliteration](https://github.com/jim-plus/llm-abliteration/) |
| Winsorization | Clip extreme activations before taking means. Required for Gemma 3 (GeGLU outliers); without it the result was incoherent | grimjim (above) |
| Massive-activation dims | Zero `r` on attention-sink dims | already in `gazua/directions.py` |
| Interpolated direction | A float `direction_index` that lerps between adjacent layer directions | [Heretic](https://github.com/p-e-w/heretic) |
| Per-layer weight kernel | Ablation strength varies by layer (peak, edge weight, width) with separate params for attention and MLP; MLP edits hurt capability more | Heretic |
| COSMIC | Pick direction and layer by cosine similarity of activations, without reading the output text or relying on refusal strings | [arXiv:2506.00085](https://arxiv.org/abs/2506.00085) (ACL Findings 2025) |
| Affine concept editing (ACE) | Ablate relative to a reference mean (affine, not linear) and steer with a strength α; also works on RWKV | [arXiv:2411.09003](https://arxiv.org/abs/2411.09003) |
| ORBA | Reflection-based bounded ablation | [grimjim blog](https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation) |
| ARA (Arbitrary-Rank Ablation) | Treat the edit as a matrix optimization (L-BFGS, LoRA rank ~50, row magnitudes preserved). Default in Heretic 2.0 (1 Nov 2026); preprint not yet out | [Heretic PR #211](https://github.com/p-e-w/heretic/pull/211) |

## 3. Refusal is not exactly one direction

- **Concept cones** ([Wollschläger et al., arXiv:2502.17420](https://arxiv.org/abs/2502.17420)): there are
  independent refusal directions and cones up to 5-D, found by gradient-based search. Attack success
  improves with cone dimension and plateaus at about 4. Inducing refusal is much easier than removing it.
- **More than one direction** ([arXiv:2602.02132](https://arxiv.org/html/2602.02132v1)): a small shared
  core of SAE refusal latents plus a long tail of style- and domain-specific ones.
- **Contrast-set pitfall** ([arXiv:2603.22061](https://arxiv.org/abs/2603.22061)): topic-matched harmless
  prompts cancel the signal and give no usable direction (Qwen 3.5 2B). Unmatched contrast works.
- **Concept-specific refusal** (RepIt, ICLR 2026): with a dozen samples, refusal can be removed for one
  concept while kept elsewhere.
- **Cross-lingual** ([arXiv:2505.17306](https://arxiv.org/abs/2505.17306), NeurIPS 2025): an English
  direction removes refusal in 14 safety-aligned languages.

## 4. Architectures

- **Post-norm blocks (Gemma 2/3/4)**: `o_proj` and `down_proj` outputs pass through an RMSNorm before
  being added to the residual, so orthogonalizing `W` against `r` is not exact.
  Hypothesis to test: RMSNorm output is `g ⊙ y / rms(y)`, so `out·r = y·(g ⊙ r) / rms(y)`.
  Orthogonalizing `W` against `û = normalize(g_eff ⊙ r)` (with `g_eff = 1 + g` in Gemma's
  parametrization) would make the edit exact again.
- **MoE**: refusal is spread across experts. Ablating only the experts that fire leaves the others
  able to restore refusal when routing shifts, so every expert must be edited. Fused expert tensors
  (`gate_up_proj`/`down_proj` stored as 3-D parameters, not `nn.Linear`) are silently skipped by tools
  that only walk `nn.Linear`. Whether to edit the router is disputed. On Kimi-K2.5 (384 experts, top-8)
  standard abliteration failed; part of refusal may live in *which* experts fire
  ([Heretic #221](https://github.com/p-e-w/heretic/issues/221), [arXiv:2606.04160](https://arxiv.org/pdf/2606.04160), [arXiv:2609.09793](https://arxiv.org/pdf/2609.09793)).
- **Hybrid (linear attention / SSM / conv)**: writers include `linear_attn.out_proj` and SSM `out_proj`.
  Reports conflict: on Qwen3.6-27B only `ffn_down` worked and the direction spans all 64 layers; on
  LFM2.5 only the 6 full-attention layers mattered *(self-reported)*.
- **VLMs**: refusal lives in the text decoder; editing only the decoder keeps vision intact
  *(self-reported)*.
- **Reasoning models**: refusal is decided early in the chain of thought, and the CoT reinforces it
  ([arXiv:2507.03167](https://www.alphaxiv.org/abs/2507.03167), [arXiv:2605.26772](https://arxiv.org/abs/2605.26772)).
  Ablation degrades these models more. Tooling must skip `<think>` blocks when scoring, and response-prefix
  detection that leaves the model inside its CoT produced a degenerate direction on Qwen3
  ([Heretic #487](https://github.com/p-e-w/heretic/issues/487)).

## 5. Evaluation

- **Substring matching is weak**: in one validation, 93% precision but 56% recall on refusals, κ = 0.54
  vs 0.88 for an LLM judge. It also counts rambling, non-answers as "compliance"
  ([StrongREJECT](https://bair.berkeley.edu/blog/2024/08/28/strong-reject/), [HarmMetric Eval](https://arxiv.org/pdf/2509.24384)).
  Heretic still defaults to keywords, with an optional ModernBERT refusal classifier.
- **KL**: Heretic uses first-token `KL(base ‖ edited)` on 100 harmless prompts. The Pareto front
  (refusals vs KL) is the standard way to report a result.
- **Capability**: math is the most sensitive skill. GSM8K changed from +1.5 to −18.8 pp across 4 tools and
  16 models; single-pass tools (DECCP, ErisForge) preserved more than optimization-based Heretic 1.x
  ([arXiv:2512.13655](https://arxiv.org/abs/2512.13655)).
- **Off-target effects**: abliterated MoE models became more optimistic, wrote longer, and hedged less,
  with the direction of some effects differing by family ([arXiv:2607.17427](https://arxiv.org/abs/2607.17427)).
  The same study found a stale chat template silently changing prompts.
- **Over-refusal** should be measured too (XSTest, OR-Bench): a good edit lowers both harmful and
  spurious refusals.

## 6. Defenses (for robustness testing)

- **Extended refusal** fine-tuning ([arXiv:2505.19056](https://arxiv.org/abs/2505.19056)): responses that
  give an overview, then refuse, then justify spread the signal across tokens. Refusal drops by at most
  10% after abliteration (vs 70-80% for baselines), at the cost of coherence.
- Concept-cone results suggest multi-direction ablation is the natural counter.

## 7. Existing tools

| Tool | Notes |
|---|---|
| [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0) | Reference implementation. Optuna TPE over kernel params, LoRA-based edits, MPOA, ARA, plugin scorers, bnb 4-bit, reproducibility manifests, MoE/hybrid/VLM support |
| [llm-abliteration](https://github.com/jim-plus/llm-abliteration/) (NousResearch fork) | Projected and norm-preserving biprojected abliteration |
| FailSpy abliterator | TransformerLens-based; ran on only 5/16 models in the comparison |
| DECCP, ErisForge | Single-pass; best capability preservation in arXiv:2512.13655 |
| remove-refusals-with-transformers | Minimal didactic implementation |
| [Vauban](https://vauban.readthedocs.io/) | Docs with a clear concept explanation |

Heretic is AGPL-3.0: read it for ideas, but do not copy code into Gazua unless Gazua adopts a compatible license.
