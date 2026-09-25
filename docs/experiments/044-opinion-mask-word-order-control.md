# Experiment 044 — Does word order contribute to the residual-context effect?

**Status:** adaptive post-result follow-up; protocol, runner, analyzer, and tests must be committed before the 044 shuffled-context judgments. Laya selected this next test. The result is exploratory because the direction was chosen after the Experiment 043 outcome.

## Question and literature boundary

Experiment 043 found that opinion-masked context substantially reduced continuous valence–arousal (VA) error relative to the aspect-only prior. This follow-up asks whether the advantage depends on the natural order of the remaining whitespace tokens or survives when those same tokens are deterministically shuffled.

Aspect sentiment literature already studies aspect-to-context syntax and target-opinion alignment; this is not a novelty claim about word order or ABSA. The scoped literature does not answer whether the particular post-opinion-mask VA gain in Experiment 043 survives a same-token order shuffle under this local instruction-tuned model. Because the follow-up is selected after seeing 043, all intervals and gates below are exploratory diagnostics, not independent confirmation.

- [DimABSA dataset paper](https://aclanthology.org/2026.acl-long.1881/)
- [Hu et al., target-specific opinion phrase selection](https://aclanthology.org/K19-1091/)
- [Phan and Ogunbona, context and syntactical features in ABSA](https://aclanthology.org/2020.acl-main.293/)
- [Kumaran et al., confidence-driven abstention](https://www.nature.com/articles/s42256-026-01293-x) (adjacent metacognition literature; not direct evidence for this order test)

## Sample and fixed comparators

- Reuse exactly the 217 disjoint aligned IDs and three language rows from Experiment 043. Do not resample or use item predictions to select cases.
- Verify Experiment 043's private generation file against SHA-256 `1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8`, its protocol hash, 1,302 complete rows, and zero invalid outputs.
- The natural `opinion_masked` and `aspect_only` predictions from 043 are frozen paired comparators. Generate only the new `opinion_shuffled` condition, one output per ID and language (651 new outputs).
- Shuffle the whitespace-separated tokens of each already-masked source sentence by ascending SHA-256 of `exp044|20260944|<ID>|<language>|<original-token-index>`. If this ordering accidentally equals the original order and the sentence has multiple tokens, rotate left by one token. The named aspect and all other prompt fields stay unchanged.
- Assert exact token multiset preservation and that all annotated opinion spans remain represented by the same `[MASKED]` placeholders. The shuffle changes order and natural syntax; it does not create a human-validated paraphrase.
- All text, IDs and per-item outputs remain in ignored `.context/`. The public bundle contains only aggregates, hashes and provenance.

## Model and decoding

Run the same frozen Qwen2.5-3B-Instruct revision, locally on MPS, with greedy decoding and the exact 6,561-output finite grammar of one-decimal VA pairs from Experiment 043. Reuse the same output instruction. The grammar has no abstention option and may alter answer content; every contrast is conditional on this constrained decoder.

## Outcomes

- **Primary exploratory contrast:** `RMSE(opinion_shuffled) - RMSE(opinion_masked)`. Positive values mean natural word order improves on the same bag of words.
- Bootstrap by the 217 source IDs with replacement, keeping all three languages and both VA dimensions together; 10,000 draws, seed `20260944`.
- Report the primary estimate and percentile 95% interval. A natural-order contribution is practically supported in this sample if the estimate is at least 0.25 VA points and the interval's lower bound is above zero. Failure to meet this rule is inconclusive; it does not prove the order is irrelevant.
- **Descriptive anchor:** `RMSE(aspect_only) - RMSE(opinion_shuffled)`, using the frozen 043 aspect-only baseline, to show whether the shuffled token inventory still beats an aspect-only prior. Report its paired interval without a second decision gate.
- Report language-specific RMSEs, output coverage, hashes and runtime descriptively. If invalid outputs exceed 2% of the 651 new generations, stop score analysis.

## Limits

This is an adaptive within-sample mechanism check, not an independent replication. Token shuffling simultaneously disrupts syntax, scope, negation and discourse links; an effect cannot be attributed to syntax alone. The baseline predictions are reused from a prior result, while only the shuffled condition is newly generated. The task remains one public restaurant-review benchmark, one local model and constrained VA decoding. A follow-up in a new corpus or model would still be needed before a general claim or LessWrong post.

## Provenance

- Parent protocol and outcomes: `docs/experiments/043-expanded-opinion-mask-repair.md`, `results/expanded-opinion-mask-repair-v1/README.md`.
- Runner: `scripts/run_opinion_mask_word_order_control_044.py`.
- Analyzer: `scripts/analyze_opinion_mask_word_order_control_044.py`.
