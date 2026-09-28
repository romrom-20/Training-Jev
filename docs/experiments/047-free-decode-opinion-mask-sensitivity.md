# Experiment 047 — Does the opinion-mask gain survive free greedy decoding?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 046. This is a decoder-sensitivity test on the existing held-out sample, not a new-sample replication. Commit this protocol, runner, analyzer, and tests before generating new Qwen outputs. Laya's choice is agenda triage only; its probabilities are uncalibrated.

## Question and literature boundary

Experiment 043 found that Qwen2.5-3B's finite one-decimal VA grammar improved RMSE by 1.845 points when opinion-masked review context was added to the target aspect. Experiments 044 and 045 found that shuffling the masked words did not reliably hurt performance, and Experiment 046's non-pretrained sparse model recovered only a small part of the masked-context gain. Does the large Qwen context gain survive ordinary unconstrained greedy generation?

The DimABSA benchmark includes prompted and fine-tuned continuous VA systems; structured constrained decoding is one valid output strategy, not evidence by itself that the model's unconstrained responses have the same behavior. This test makes no novelty claim about constrained generation or dimensional ABSA. It isolates the main decoder limitation left by this project using a paired prompt contrast. See [the DimABSA dataset paper](https://aclanthology.org/2026.acl-long.1881/) and [the SemEval-2026 task paper](https://aclanthology.org/2026.semeval-1.452/).

## Sample and prompts

- Reconstruct exactly Experiment 043's 217 held-out IDs and target indices from the pinned Russian, Ukrainian, and Tatar test sources. Verify the source hashes, 043 protocol hash, and 043 private-output hash `1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8`; verify all gold VA pairs against 043.
- Generate two prompts for each ID/language: `aspect_only` and `opinion_masked`, for 1,302 total judgments. Keep the 043 sample fixed; do not select by prior errors or predictions.
- Use the same base instruction as Experiment 041, which asks for exactly one JSON object with numeric valence and arousal from 1 to 9 and gives no decimal precision or finite candidate list. Aside from the output contract, the aspect and masked text construction matches the 043 conditions.
- Keep prompts, review text, IDs, raw answers, and item predictions in ignored `.context/`.

## Model, decoder, and validity gate

- Run `Qwen/Qwen2.5-3B-Instruct` at the task-ladder pinned revision on local MPS, greedy (`do_sample=False`), using the same Transformers generation defaults and `max_new_tokens=40` as the prior free-generation runner. There is no candidate trie, grammar constraint, score rounding, or retry.
- Parse only one JSON object with exactly numeric `valence` and `arousal` values, both finite and within [1, 9]. A single fenced JSON object is accepted; prose, missing/extra keys, non-numbers, and out-of-range values are invalid.
- If more than 2% of all 1,302 outputs are invalid, classify the run as a protocol-execution failure and withhold every score comparison. Otherwise, analyze only complete paired source-ID clusters; no item is selectively retried.

## Outcome

- **Primary decoder-sensitivity contrast:** `RMSE(aspect_only) - RMSE(opinion_masked)` for Qwen2.5-3B. Positive means context improves VA accuracy.
- Bootstrap the 217 source IDs with replacement, preserving all three language rows and both VA dimensions; use 10,000 draws and seed `20260947`.
- Report the estimate and percentile 95% source-ID cluster interval. The large context gain is considered practically reproduced in free decoding if the estimate is at least 0.25 and the interval's lower bound is above zero. Otherwise, the finite-grid result has not been reproduced under this free-decoding setup; a failed rule is not equivalence evidence.
- Report invalid coverage by condition, language-specific RMSE, runtime, and hashes descriptively. Do not publish raw text or item outputs.

## Limits

This reuses the same public DimABSA release, selected IDs, model family and prompt template. The key change removes the finite output grid and decimal-precision instruction together; it therefore tests the free-output contract as a whole, not the trie in isolation. Model self-reports and parsed VA scores are compared with dataset annotations, not fresh human ratings. A positive result would reduce one decoder concern but would not independently replicate the benchmark effect or establish that masked context is sufficient in ordinary language use.

## Provenance

- Parent outcomes: `results/expanded-opinion-mask-repair-v1/README.md`, `results/opinion-mask-word-order-v1/README.md`, `results/opinion-mask-order-model-size-v1/README.md`, and `results/masked-context-lexical-baseline-v1/README.md`.
- Private Laya agenda trace: `.context/laya-research-triage-047.json` (not committed).
- Runner/analyzer: `scripts/run_free_decode_opinion_mask_sensitivity_047.py`, `scripts/analyze_free_decode_opinion_mask_sensitivity_047.py`.
