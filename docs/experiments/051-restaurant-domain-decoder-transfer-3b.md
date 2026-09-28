# Experiment 051 — Does the 3B decoder interaction transfer to English restaurant reviews?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 050. Laya's probabilities are uncalibrated and are used only for agenda triage. Freeze this protocol and runner before scoring the restaurant split.

## Question and motivation

On 943 English laptop test items, Experiment 050 found a Qwen2.5-3B context gain of +2.299 VA RMSE points with finite-grid decoding, versus −0.367 with free greedy decoding. The interaction was +2.666 (95% source-ID interval [+2.575, +2.757]). A post-run audit found that finite-grid aspect-only outputs frequently collapsed to (1.0, 1.0). The result may depend on the product domain, the benchmark annotation distribution, or the output convention.

Experiment 051 tests the identical Qwen2.5-3B, prompt, and decoder factorial on a disjoint **English restaurant** test split. This keeps language, prompt, model and source release fixed while changing product domain. It is a new set of public source items within DimABSA, not a separate corpus or blind test.

## Data and sample

- Use official `eng_restaurant_test_task2.jsonl` from DimABSA commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256 `e90386169422c84ddd42d51029f23934b54724fafe270b65138836067492857d`. The local cache was byte-verified against that pinned raw GitHub path before this protocol was frozen.
- Evaluate all rows with a non-null target aspect and opinion for which the aspect and opinion each occur exactly once, every distinct non-null opinion span occurs exactly once, and no opinion span overlaps the target aspect. Choose the first eligible target triplet in source order, at most one per unique source ID; require valid two-dimensional `V#A` gold values. The source audit found 963 eligible IDs: 197 with valence below 4.5, 48 from 4.5 through 5.5 inclusive, and 718 above 5.5.
- Reuse the 049/050 one-decimal prompt construction and all-annotated-opinion mask. Keep all text, aspects, IDs, prompts, raw outputs and per-item predictions in ignored `.context/` files. Publish aggregates and source provenance only.

## Conditions and model

For every eligible source ID, ask for `aspect_only` and `opinion_masked` VA outputs under both decoding methods, for 3,852 generations total. Keep prompt bytes identical between decoder arms for every ID and evidence condition.

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS inference.
- `finite_grid`: same finite 6,561-candidate JSON grid of one-decimal valence and arousal scores in [1, 9].
- `free_greedy`: ordinary greedy generation, maximum 40 new tokens, strict JSON parser, no retries or output repair.

## Outcomes and rules

- For each decoder, calculate the opinion-masked context gain as `RMSE(aspect_only) - RMSE(opinion_masked)` over both VA dimensions. Positive values mean the remaining review context reduces error.
- **Primary contrast:** `context_gain(finite_grid) - context_gain(free_greedy)`. Positive values mean the finite output grammar amplifies the measured value of context on English restaurant reviews.
- Bootstrap source review IDs with replacement, retaining both evidence conditions and both VA dimensions; use 10,000 draws, seed `20260951`, and percentile 95% intervals.
- The 3B free-output invalid rate must be ≤2%; if it exceeds 2%, withhold all restaurant score contrasts. Otherwise calculate on complete source IDs and report coverage.
- The registered practical rule is estimate ≥ +0.25 and lower interval bound > 0. Failing it is inconclusive; it does not prove decoder equivalence.
- **Secondary domain contrast:** compare this restaurant interaction with Experiment 050's laptop interaction. The estimand is `interaction(laptop) - interaction(restaurant)`. Bootstrap the two source domains independently, preserving all cells within each source ID, and report the interval descriptively; it is not a second confirmatory claim.
- Report invalid outputs, runtime, data/model/protocol hashes, both decoder-specific context gains, the primary interaction, and the secondary domain difference. Do not interpret one benchmark release as a general law about constrained generation.

## Compute and limits

The 3B checkpoint was run in Experiment 050 on the same 24-GB MacBook Air. This run adds 3,852 generations, with a resumable JSONL checkpoint after each batch. Public test labels may overlap model pretraining. The split is a within-release domain transfer, and the full eligible data retain a naturally positive-heavy VA distribution.

## Provenance

- Dataset: [DimABSA paper](https://arxiv.org/abs/2601.23022) and [official repository](https://github.com/DimABSA/DimABSA2026).
- Parent outcomes: [049 English laptop at 1.5B](../../results/laptop-decoder-transfer-v1/README.md), [050 English laptop at 3B](../../results/laptop-model-size-v1/README.md).
- Private Laya selection: `.context/laya-research-triage-052.json` (not committed).
- Runner/analyzer: `scripts/run_restaurant_domain_decoder_051.py`, `scripts/analyze_restaurant_domain_decoder_051.py`.
