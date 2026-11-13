# Gazua backlog

Goal: turn the current study pipeline into a tool that abliterates any open-weight
Hugging Face causal LM (dense, post-norm, MoE, hybrid, VLM, reasoning) with measured,
reproducible quality. Evidence for each choice is in [RESEARCH.md](RESEARCH.md).

**Priority**: P0 blocks the next milestone, P1 is needed for the milestone, P2 is valuable but deferrable.
**Size**: S ≤ 1 day, M 2-4 days, L ≥ 1 week.
**Constraint**: the dev machine is CPU-only (AMD RX 7600 on Windows). Every P0/P1 item must run on
CPU with a ≤ 1.5B model; GPU paths are optional accelerators.

## Definition of success

A run on a supported model must report, against the base model:

| Metric | Target |
|---|---|
| Harmful refusal rate (judge, not substrings) | ≤ 5% |
| Harmless first-token KL | ≤ 0.1 (Pareto front reported) |
| Over-refusal on XSTest-safe | ≤ base |
| GSM8K / MMLU / IFEval delta | ≥ −2 pp |
| Hook vs baked weights | max logit diff ≤ 1e-3 (fp32) |

## Milestones

| Milestone | Outcome | Epics |
|---|---|---|
| **M0 Foundation** | Current pipeline under tests, config-driven, one CLI | E0 |
| **M1 Trustworthy eval** | Results you can believe: judge, KL, capability, over-refusal | E2, E5 |
| **M2 Faithful Arditi + refinements** | Position sweep, induce check, projected, MPOA, kernel | E3, E4 |
| **M3 Any dense model** | Architecture adapters, auto writer discovery, Gemma post-norm | E1, E8 (dense) |
| **M4 Auto-tuning** | Optuna multi-objective search with Pareto front | E6 |
| **M5 Scale** | Models larger than RAM, bf16, streaming edit, export | E7, E9 |
| **M6 Hard architectures** | MoE, hybrid, VLM, reasoning | E8 |
| **M7 Research track** | Multi-direction, ACE, ARA-like optimization, robustness to defenses | E10 |

---

## E0 Foundation

**GZ-001 Package layout and single CLI** · P0 · M
Replace `scripts/0x_*.py` with `gazua` subcommands: `extract`, `select`, `apply`, `eval`, `run` (all steps).
- AC: `gazua run --model Qwen/Qwen2.5-0.5B-Instruct` reproduces today's result end to end.
- AC: scripts become thin wrappers or are removed.

**GZ-002 Typed run config** · P0 · S
One TOML file (dataclass/pydantic) for model, dtype, datasets, splits, method, eval, seed. CLI flags override it.
- AC: the full config is saved next to every artifact.

**GZ-003 Run manifest and reproducibility** · P1 · S
Write `manifest.json` with model revision hash, dataset hashes, library versions, seed, config, chosen direction, metrics.
- AC: `gazua run --reproduce manifest.json` gives the same direction (cosine ≥ 0.9999) on the same machine.

**GZ-004 Test suite on tiny models** · P0 · M
pytest with `hf-internal-testing/tiny-random-*` models (Llama, Qwen2, Gemma2, Mixtral, Qwen3-MoE).
- AC: test that hook ablation and baked weights agree (≤ 1e-4 in fp32).
- AC: test that, after baking, every writer output has `|out·r̂| ≤ 1e-5`.
- AC: suite runs on CPU in < 2 min.

**GZ-005 CI** · P1 · S
GitHub Actions on CPU: lint (ruff), tests.

**GZ-006 Artifact cache** · P1 · S
Cache activations, baseline logits and baseline generations by (model revision, prompts hash, config hash).
- AC: re-running `select` after `extract` skips every forward pass on the base model.

## E1 Architecture abstraction

**GZ-101 `ModelAdapter` interface** · P0 · M
Exposes `layers()`, `residual_writers(layer)` (with kind: attn, mlp, expert, shared_expert, ssm, conv), `embedding()`, `lm_head()`, `post_writer_norm(writer)`, `is_tied()`.
- AC: `ablation.py` and `directions.py` no longer reference `model.model.layers` or `mlp.down_proj` directly.

**GZ-102 Auto-discovery of residual writers** · P0 · L
Find the modules whose output is added to the residual stream, by structure first (known names) and then by numerical check: zeroing a candidate must change `hidden_states[i+1] − hidden_states[i]` by exactly its output.
- AC: correct writer set on every tiny model in GZ-004 with no per-architecture code.
- AC: fails loudly (lists unclaimed parameters) instead of silently skipping writers. Silent skipping is the documented MoE failure mode.

**GZ-103 Weight edit on non-`nn.Linear` parameters** · P0 · M
Support 3-D fused expert tensors (`[n_experts, d_out, d_in]`), Conv1d-style weights (GPT-2), and transposed layouts.
- AC: unit tests per layout check `r̂ᵀW_e = 0` for every expert `e`.

**GZ-104 Tied embeddings and lm_head** · P1 · S
Keep the current untie logic, but make embedding orthogonalization optional (Heretic skips it; its benefit is unclear) and measure both.

**GZ-105 Hook-mode ablation for any adapter** · P0 · S
Same writer set as the weight edit, used during search (cheap, reversible).

## E2 Data

**GZ-201 Dataset registry** · P0 · M
Loaders with license notes and caching. Harmful: AdvBench, HarmBench, JailbreakBench, StrongREJECT, SORRY-Bench, `mlabonne/harmful_behaviors`. Harmless: Alpaca (no-input), `mlabonne/harmless_alpaca`. Over-refusal: XSTest, OR-Bench-hard.
- AC: any registered set can be used for extraction or evaluation from the config.

**GZ-202 Leak-free splits** · P0 · S
Deduplicate (normalized text and near-duplicates) across sets; keep extraction, selection and final eval disjoint. Final eval comes from a *different source* than extraction (e.g. extract on AdvBench, eval on HarmBench) to test generalization.

**GZ-203 Unmatched contrast by default** · P1 · S
Topic-matched harmless prompts produced no usable direction (arXiv:2603.22061). Default to unmatched; keep matched as an experiment flag.

**GZ-204 Chat template handling** · P0 · M
Use the model's own template, with an optional system prompt; detect generation-prompt tokens; support a "no thinking" switch where the template has one (Qwen3 `enable_thinking`).
- AC: the rendered prompt is logged in the manifest. A stale template silently changed results in arXiv:2607.17427.

**GZ-205 Multilingual prompt sets** · P2 · S
PolyRefuse or translated subsets, to verify cross-lingual transfer (arXiv:2505.17306).

## E3 Direction extraction

**GZ-301 Post-instruction position sweep** · P0 · M
Collect means at every post-instruction token position (template tokens after the user message), not only the last one, as Arditi et al. do. Candidates become (layer, position).

**GZ-302 Arditi selection criteria** · P0 · M
Rank candidates by: refusal score after ablation, refusal induced by *adding* `r` on harmless prompts, KL ≤ threshold, layer ≤ 0.8 depth. Use a logit-based refusal score (probability of refusal-start tokens) so selection needs no generation.
- AC: selection on Qwen2.5-0.5B runs in < 10 min on CPU.

**GZ-303 Winsorization** · P1 · S
Clip per-component activations at a quantile before averaging. Needed for Gemma 3 GeGLU outliers. Keep the existing massive-dims masking as an alternative and compare.

**GZ-304 Projected direction** · P1 · S
Option to remove from `r` its projection on the harmless mean (grimjim). Report cos(r, μ_harmless).

**GZ-305 Interpolated direction index** · P2 · S
Float index lerping between adjacent layer directions (Heretic), so the optimizer can search a continuous space.

**GZ-306 COSMIC selection** · P2 · M
Output-free selection by activation cosine similarity (arXiv:2506.00085). Useful for models whose refusals do not use stock phrases, or non-English.

**GZ-307 Numerics** · P0 · S
Accumulate means in fp32/fp64 regardless of model dtype; stream means instead of storing all activations.

## E4 Intervention methods

**GZ-401 Method interface** · P0 · S
`Modifier.apply(adapter, directions, params)` for both hook and weight modes, so methods are swappable and comparable.

**GZ-402 Per-layer, per-component weight kernel** · P1 · M
Strength profile over layers (peak position, peak weight, edge weight, width) with separate params for attention, MLP and experts. MLP edits hurt capability more, so MLP weight may reach 0.

**GZ-403 Norm preservation (MPOA)** · P1 · S
After the edit, rescale each row of `W` to its original norm.
- AC: A/B on two models; report KL and GSM8K deltas.

**GZ-404 Biprojection** · P2 · S
Measure `r` at one layer and remove it at a set of other layers.

**GZ-405 Exact post-norm ablation (Gemma)** · P1 · M
Test the hypothesis in RESEARCH.md §4: orthogonalize writers against `normalize(g_eff ⊙ r)`.
- AC: on tiny Gemma2/Gemma3, hook ablation applied *after* the post-norm matches the baked weights to ≤ 1e-4. If it fails, fall back to hook-only for these models and document why.

**GZ-406 LoRA delta export** · P2 · M
Express the edit as a low-rank delta (rank 1 per direction, small rank with MPOA via `svd_lowrank`), saved as a PEFT adapter, so the base model stays untouched and the edit can be toggled.

**GZ-407 Strength scaling** · P1 · S
Partial ablation `W ← W − α r̂r̂ᵀW` with α ∈ [0, 1.5], used by the kernel and by the optimizer.

## E5 Evaluation

**GZ-501 Tiered refusal judge** · P0 · M
1. Keyword matcher (current, extended with Heretic-style normalization; skip `<think>` blocks).
2. Small local classifier (e.g. a ModernBERT refusal classifier), runs on CPU.
3. Optional LLM judge (local model or API) using the StrongREJECT rubric, which scores *usefulness* of a non-refusal, not just absence of refusal.
- AC: on a 200-sample hand-labelled set, report precision, recall and κ for each tier.

**GZ-502 KL suite** · P0 · S
First-token KL (current) plus KL averaged over the first N tokens of base-model responses (teacher-forced), which catches drift that first-token KL misses.

**GZ-503 Capability benchmarks** · P1 · M
`lm-eval` integration with a CPU-sized preset: GSM8K (most sensitive, arXiv:2512.13655), MMLU subset, IFEval, ARC-Easy. Report the delta with confidence intervals.

**GZ-504 Over-refusal** · P1 · S
XSTest safe prompts and OR-Bench-hard; report refusal rate before and after.

**GZ-505 Off-target behaviour probes** · P2 · M
Tone, hedging and verbosity metrics on neutral prompts (response length, uncertainty-word rate), following arXiv:2607.17427.

**GZ-506 Run report** · P1 · M
One HTML/Markdown report per run: metrics table, Pareto plot, layer-sweep plot, side-by-side samples (harmless only by default), manifest.

**GZ-507 Hand-labelling tool** · P2 · S
Small CLI to label completions as refusal / compliance / degenerate, producing the gold set for GZ-501.

## E6 Optimization

**GZ-601 Multi-objective search** · P1 · L
Optuna TPE over (direction index, kernel params per component, α, projected on/off, MPOA on/off). Objectives: refusal (judge tier 1-2) and KL. Startup random trials, then TPE.
- AC: resumable from journal storage; Pareto front saved; user picks a trial or a rule picks it (lowest refusal with KL ≤ τ).

**GZ-602 Cheap inner loop** · P1 · M
Trials use hook mode, cached baselines, a logit-based refusal proxy and ~64 prompts. Only Pareto candidates get full generation and benchmarks.
- AC: a 50-trial search on Qwen2.5-0.5B finishes on CPU in < 2 h.

**GZ-603 Capability-aware objective** · P2 · M
Optional third objective (e.g. GSM8K-mini accuracy) for models where math degrades.

## E7 Scale and hardware

**GZ-701 dtype support** · P0 · S
bf16/fp16 loading with fp32 math for the edit (cast row blocks to fp32, edit, cast back).

**GZ-702 Streaming weight edit** · P1 · L
Apply the edit shard by shard over `safetensors` files, without loading the model: read tensor, edit, write. Peak RAM ≈ the largest tensor.
- AC: edit a 7B bf16 model on a 16 GB RAM machine; output loads and matches the in-memory edit bit-for-bit on a small model.

**GZ-703 Low-memory activation extraction** · P1 · L
Run the forward pass layer by layer (load layer i, push the batch's hidden states through it, unload), and stop at the deepest candidate layer. Activations needed for extraction are small.
- AC: extract directions for a 7B model with < 8 GB RAM; matches the full-model result (cosine ≥ 0.999).

**GZ-704 AMD GPU spike** · P2 · S
Check whether the RX 7600 (gfx1102) works with PyTorch ROCm on Windows or WSL2, or with `torch-directml` for inference only. Record the outcome as a decision.

**GZ-705 Quantized base models** · P2 · M
Load bnb-4bit (CUDA only) or GPTQ/AWQ for extraction and search; apply the final edit to the full-precision weights.

## E8 Architecture coverage

Each item ends with a row in a compatibility matrix (`docs/COMPATIBILITY.md`): model, writer set, method, metrics.

**GZ-801 Dense pre-norm** · P0 · M — Llama 3.x, Qwen2.5/Qwen3 dense, Mistral, Phi-3/4, SmolLM. Mostly covered by E1.

**GZ-802 Post-norm (Gemma 2/3/4)** · P1 · M — Depends on GZ-405 and GZ-303.

**GZ-803 MoE** · P1 · L — Qwen3-MoE, Mixtral, OLMoE, DeepSeek-V2-Lite. Edit every expert and shared experts; router untouched by default, router edit as an experiment.
- AC: test that refusal does not return when routing is forced to unedited-in-the-past experts (regression test for partial edits).
- Research spike: measure refusal carried by routing alone (compare expert-usage histograms on harmful vs harmless prompts), following the Kimi-K2.5 report.

**GZ-804 Hybrid** · P2 · L — Qwen3.5/3.6 (linear attention), LFM2/2.5 (conv), Falcon-H1 / Nemotron-H (Mamba). Add `ssm`/`linear_attn`/`conv` writers; run the per-component sweep, since reports disagree on which writers matter.

**GZ-805 VLMs** · P2 · M — Qwen2.5-VL/Qwen3-VL, Gemma 3: edit only the language decoder; verify vision with a small captioning/VQA check.

**GZ-806 Reasoning models** · P1 · M — Qwen3 thinking, DeepSeek-R1 distills.
- AC: extraction uses the correct generation prefix (the Heretic #487 failure); judges score only the final answer; report CoT-level refusal separately.

## E9 Export

**GZ-901 Safetensors export** · P0 · S — Save weights, config (untied embeddings), tokenizer, generation config; verify the saved model reloads and matches.

**GZ-902 Model card** · P1 · S — Auto-generated card with method, direction choice, all metrics, manifest, and a clear note that safeguards are removed. Local only by default, consistent with the README.

**GZ-903 GGUF** · P2 · S — Wrap llama.cpp's `convert_hf_to_gguf.py` + quantize; smoke-test refusal rate on the GGUF.

**GZ-904 MLX** · P2 · S — `mlx_lm.convert` wrapper.

## E10 Research track

Each item is an experiment with a written result in `docs/experiments/`.

**GZ-1001 Reproduce Arditi** · P0 · M — Qwen2.5-0.5B/1.5B, Llama-3.2-1B: refusal and KL close to the paper's trends. Baseline for everything else.

**GZ-1002 Method shoot-out** · P1 · L — Classic vs projected vs MPOA vs kernel vs Heretic (run as an external tool) on 3-4 small models, same eval. Mirrors arXiv:2512.13655 with a better judge.

**GZ-1003 Multi-direction / cone ablation** · P1 · L — Top-k directions via SVD of per-prompt differences, then gradient-based search for refusal cones (arXiv:2502.17420). Measure gains vs k (expect a plateau near 4).

**GZ-1004 Affine concept editing** · P2 · M — ACE (arXiv:2411.09003) as a modifier; compare with linear ablation on KL.

**GZ-1005 Optimization-based edit (ARA-like)** · P2 · L — Learn a low-rank weight delta minimizing refusal-proxy loss with a KL penalty (L-BFGS or Adam). Re-check once the ARA preprint is published.

**GZ-1006 Robustness to defenses** · P2 · M — Fine-tune a 0.5B model with extended-refusal data (arXiv:2505.19056) and test single vs multi-direction ablation on it.

**GZ-1007 Concept-specific ablation** · P2 · M — Remove refusal for one category only (RepIt), keeping the rest. Useful as a precision benchmark for the tool.

**GZ-1008 Gemma post-norm proof** · P1 · S — Write up the `g ⊙ r` derivation with numerical evidence (pairs with GZ-405).

---

## Order of work

1. M0: GZ-001, 002, 004, 307, 701, 901
2. M1: GZ-201, 202, 204, 501 (tiers 1-2), 502, 504, 1001
3. M2: GZ-301, 302, 303, 304, 401, 403, 407
4. M3: GZ-101, 102, 103, 105, 801, 405, 802
5. M4: GZ-402, 601, 602, 506
6. M5: GZ-702, 703, 003, 006
7. M6: GZ-803, 806, 804, 805
8. M7: GZ-1002, 1003, then the rest of E10

## Known issues in the current code

- `ablation.writer_modules` hard-codes `model.model.layers`, `self_attn.o_proj`, `mlp.down_proj`: breaks on MoE, hybrids, GPT-2, VLM wrappers (GZ-101/102).
- Only the last prompt token is used for directions (GZ-301).
- Layer selection uses substring refusal on 16 generated prompts: noisy (GZ-302, GZ-501).
- KL is first-token only (GZ-502).
- Models always load in fp32 (GZ-701).
- AdvBench is the only harmful source, used for both extraction and evaluation (GZ-202).

## Risks

| Risk | Mitigation |
|---|---|
| CPU-only limits model size and search budget | Logit proxies, caching, streaming (GZ-602, 702, 703); keep P0/P1 testable on ≤ 1.5B |
| New architectures land faster than adapters | Numerical writer discovery (GZ-102) instead of name lists |
| Metrics drift with judge choice | Labelled gold set and judge agreement (GZ-501, 507) |
| License contamination from Heretic (AGPL) | Reimplement from papers and blog posts; do not copy code |
| Misuse of produced models | Local-only default, model card warnings, no Hub upload command |
