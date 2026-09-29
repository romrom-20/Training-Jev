# Experiment 065 — SIGHAN output-order model-size replication

**Status:** Laya-selected after Experiment 064; preregistered before any Qwen2.5-1.5B inference. Laya's uncalibrated probabilities are agenda triage only.

## Question and motivation

Experiment 064 found no clear aggregate output-order moderation on Chinese SIGHAN
with Qwen2.5-3B (-0.077; 95% recipient-bootstrap interval [-0.331, +0.170]),
after two English DimABSA splits showed negative estimates (-0.258 and -0.528).
This does not prove a release or language difference. It does leave model-size
dependence open. Experiment 065 holds the SIGHAN texts, recipients and donor
assignments fixed while repeating the scoring design with Qwen2.5-1.5B.

Nearby literature already reports order-sensitive LLM scoring and effects of
structured-output field/schema design; this is not a test of generic order effects
or a novelty claim. See the [literature-boundary audit](064-literature-boundary-audit.md).

## Data, cases and contexts

- Use the exact 180 source IDs and 3 same-category plus 3 cross-category donor maps
  from Experiment 064. Their private prediction artifact must match SHA-256
  `48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7`; extract
  donor assignments once and require identical recipient-to-donor pairs across
  order and decoder rows.
- Use the same SIGHAN 2024 Simplified test input and gold files pinned in 064,
  and confirm that the deterministic 064 sample IDs, categories, polarities and
  gold labels match the source again.
- Preserve polarity-matched same-category and cross-category donor conditions,
  output serialization order (valence-first / arousal-first), English instruction,
  candidate grids, and free-greedy settings from 064.
- Since no 1.5B own-review baseline exists for these exact IDs, generate its own
  masked-review baseline under both field orders for both decoders. Do not compare
  a 1.5B donor output against another model's baseline.

## Generation and compute

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Run greedy MPS inference on the
  24-GB MacBook Air; no cloud inference.
- Reuse 064's recipient-to-donor maps exactly, avoiding a tokenizer-dependent
  remapping between model sizes. Use the 1.5B tokenizer only for its own prompt
  length and candidate-token preflight.
- Generate 4,320 donor-context outputs (180 IDs × 3 maps × 2 category conditions
  × 2 JSON orders × 2 decoders) and 720 own-review baselines (180 × 2 orders × 2
  decoders): **5,040 new outputs**. Free-greedy cap: 40 tokens; strict parser;
  no retry or repair. Persist resumable private JSONL and manifest under ignored
  `.context/`.
- Verify protocol, source, parent-artifact and map hashes; exact case and job
  counts; polarity/category constraints; unique candidate-token sequences; and
  prompt/token boundaries before loading the target model.

## Outcomes and analysis

For each model, order, donor condition and decoder, calculate context gain as
`RMSE(own masked review) - RMSE(donor masked review)`. The category-match effect
is the same-minus-cross difference in the finite-grid-minus-free-greedy context
gain. The model-specific aggregate order moderation is this effect under
arousal-first minus its value under valence-first.

The **primary endpoint** is the difference between model-specific aggregate
order moderations, `1.5B minus 3B`. Use 10,000 paired recipient-ID bootstrap draws
(seed `20260965`) on the intersection of IDs complete in every 3B and 1.5B cell;
retain all conditions, orders, decoders and fixed donor maps together. Report the
two-sided percentile 95% interval. A difference interval excluding zero indicates
model-size dependence in this setup. An interval including zero is only a failure
to detect a size difference; it does not prove the interaction transfers, since
both model estimates may be near zero. Report both models' order moderations and
intervals as secondary outcomes.

If either model's donor-context free-greedy invalid-output rate exceeds 2%,
withhold the primary score comparison. Otherwise report invalid counts and
complete-case coverage. No item-level content or individual prediction is
released.

## Limits

The 1.5B and 3B comparison uses one dataset, one selected recipient cohort, one
prompt, one model family, and one laptop. The test set is public and pretraining
exposure cannot be excluded. Cross-category donors alter lexical and semantic
content as well as category. The comparison can assess whether the measured
interaction differs by model size in this setup; it cannot explain the earlier
English/Chinese difference or establish model-general order effects.

## Provenance

- Prior replication and exact maps: [Experiment 064](../../results/sighan-output-key-order-replication-v1/README.md).
- Prior English split estimates: [Experiment 062](../../results/output-key-order-topic-match-v1/README.md) and [Experiment 063](../../results/output-key-order-restaurant-replication-v1/README.md).
- Local Laya choice trace: `.context/laya-research-triage-065.json` (ignored).
