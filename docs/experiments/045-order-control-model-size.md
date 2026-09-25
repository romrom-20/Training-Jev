# Experiment 045 — Does the order effect depend on model size?

**Status:** adaptive follow-up chosen by Laya after Experiments 043 and 044. Commit this protocol, runner, analyzer and tests before collecting any Qwen2.5-1.5B judgments. Because the follow-up is selected after observing 043/044, this is an exploratory model-size comparison, not independent confirmation.

## Question and scope

Experiment 043 found a large aspect-only versus opinion-masked VA gain under Qwen2.5-3B. Experiment 044 found that, on those same items, a deterministic shuffle of the masked sentence's whitespace tokens slightly outperformed its natural order for Qwen2.5-3B. Does that natural-versus-shuffled difference persist in the smaller Qwen2.5-1.5B checkpoint?

Word-order robustness and syntax-sensitive aspect sentiment have both been studied in other settings. The 2021 BERT study reports that shuffling token order can minimally affect several GLUE tasks; ABSA work has modeled syntactic context and target-specific word relations. This experiment makes no novelty claim about those topics. It tests a narrow model-size boundary for the observed multilingual continuous-VA order effect.

- [Sinha et al. (2021), BERT without word ordering](https://aclanthology.org/2021.acl-short.27/)
- [Phan and Ogunbona (2020), context and syntactical features in ABSA](https://aclanthology.org/2020.acl-main.293/)
- [DimABSA dataset paper](https://aclanthology.org/2026.acl-long.1881/)

## Sample and conditions

- Reuse exactly the 217 sentence IDs and Russian, Ukrainian, and Tatar rows from Experiment 043. No resampling or item-level outcome selection.
- Verify the pinned source hashes and Experiment 043/044 private-output hashes before generation. Use Experiment 043's deterministic 217-ID selection and Experiment 044's deterministic token shuffler.
- For each ID and language, collect two new Qwen2.5-1.5B judgments: `opinion_masked` in natural order and `opinion_shuffled` with the exact same token multiset and shuffle seed as Experiment 044. Collect both conditions under the same model size; do not compare 1.5B against 3B outputs for the primary estimate.
- There are 217 × 3 × 2 = 1,302 new judgments. Keep source text, aspects, IDs and outputs in ignored `.context/`.

## Model and output

Run `Qwen/Qwen2.5-1.5B-Instruct` at the task-ladder pinned revision, locally on MPS, greedy. Use the exact 6,561-candidate JSON grammar from Experiment 043: valence and arousal from 1.0 to 9.0 in 0.1 increments. Both prompts use the same output instruction and grammar; there is no abstention candidate. Verify all candidate token boundaries before inference. Do not retry invalid outputs. More than 2% invalid generations stops score interpretation.

## Outcomes and decision rule

- **Primary exploratory contrast:** `RMSE(opinion_shuffled) - RMSE(opinion_masked)` for Qwen2.5-1.5B. Positive favors natural word order.
- Bootstrap 217 source IDs with replacement, preserving the three language rows and both VA dimensions; use 10,000 draws and seed `20260945`.
- Report the estimate and percentile 95% interval. A natural-order contribution is supported for this sample if the estimate is at least 0.25 VA points and the interval's lower bound is above zero. Otherwise the model-size result is inconclusive or favors shuffled input; it does not establish equivalence.
- Report language-specific contrasts descriptively, output coverage, runtime and provenance. No item-level labels or predictions are published.

## Limits

This is a post-result model-size follow-up using the same 217 public benchmark items and the same Qwen family. The finite output grid may alter answer content. Whitespace token shuffling disrupts syntax, negation scope and discourse at once. Any result remains conditional on this dataset, model family, and decoder; it does not establish broader language understanding or an independent replication.

## Provenance

- Parent protocols and aggregate results: `docs/experiments/043-expanded-opinion-mask-repair.md`, `docs/experiments/044-opinion-mask-word-order-control.md`.
- Runner: `scripts/run_opinion_mask_order_model_size_045.py`.
- Analyzer: `scripts/analyze_opinion_mask_order_model_size_045.py`.
