# Experiment 076 — does forced-score coupling survive role and corpus controls?

**Status:** registered before model inference after local Laya triage. Laya chose
the combined small-sample design (C, uncalibrated agenda probabilities 0.372):
resolve the schema confound on the reused corpus, and check transfer on fresh
recipient texts from an independent laptop-review dataset. This is a local,
low-compute test, not a novelty or publication decision.

## Motivation and literature boundary

Experiment 075 found that moving a forced valence/arousal score from an open
assistant prefix into a user instruction changed the expected score shift, but
also changed the output from a two-key object to a one-key object. That result
cannot identify which prompt change caused the difference.

Related effects are established. Chen et al. (2024) study output order in LLM
scoring; Huang et al. (2025) study anchoring in LLMs; Kapetanovic et al. (2026)
test prior-score metadata in rubric judges; and 2026 work finds that requested
structured-output formats alter answer distributions. This experiment tests a
bounded interaction among forced affect coordinates, output order, small-model
size, role location, and recipient corpus. It does not claim to introduce
anchoring, output-order bias, or structured-output sensitivity.

## Part A — matched final request and output schema

### Question

On a fixed set of laptop reviews, does the high-minus-low anchor effect on the
other coordinate differ when the same anchor appears in a prior assistant turn
versus a prior user turn, with the same final request and one-key JSON output?

### Design

- Use 24 of Experiment 071's 64 English DimABSA laptop cases, selected before
  inference by SHA-256 ranking within the 32 negative-valence and 32
  positive-valence strata. Select 12 per stratum using the key
  `exp076-role-control|<case_id>`. Sorted selected IDs joined by newline have
  SHA-256 `92f9debeb17ba951aea6840b34d8f24992f3558358f98fb0083bc91ea487e983`.
- Models: pinned Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct revisions
  from Experiments 073–075, local MPS.
- Cross two target dimensions, forced values 2.0/8.0, and two locations. In the
  assistant-location condition, a preceding assistant turn contains a JSON
  object with the fixed coordinate. In the user-location condition, the first
  user message supplies the same coordinate and the prior assistant turn gives
  a neutral acknowledgement. Both end with the exact same final user request
  to estimate the other coordinate and return a one-key JSON object. Thus the
  target output schema and final request are held fixed; role location is still
  bundled with the minimal acknowledgement/turn-context difference required
  by chat-message alternation.
- Score all 81 canonical one-decimal numeric values plus closing brace at the
  target field's first-token position; record one greedy continuation. No
  retries, repair, or constrained decoding.
- 24 recipients × 2 models × 2 target axes × 2 anchor locations × 2 values =
  384 model-context cells.

### Primary outcomes

For each model, axis, and role-location condition, estimate the paired 8−2
change in expected score over the normalized 81-value support. The primary
contrast is assistant-location minus user-location in that change. Report
greedy shifts, canonical probability mass, and parse failures as diagnostics.
Use a paired recipient bootstrap (10,000 draws, seed `20260976`). Withhold a
cell contrast if any relevant arm has more than 2% invalid greedy outputs.

## Part B — recipient/corpus transfer

### Question

Does the original partial-assistant-prefix effect and its field-order
asymmetry appear on new laptop review texts from a different benchmark source?

### Design

- Use 24 unique English laptop sentences from the SemEval-2014 laptop test
  triplet file `14lap_test_triplets.txt`, pinned by SHA-256
  `413a3f655409af25bcb03a9499709925349fff3edbc3a7ca95fc6cdf788eeb92` from
  repository commit `d0df6600b259b6114de23cc5047c7e776cd89750`.
- Select 12 positive and 12 negative aspect-level triplets, one target per
  sentence, ordered within polarity by SHA-256 of
  `exp076-aste14-fresh|<line_index>`. The selected `(line, triplet)` IDs joined
  in line order have SHA-256
  `3960ab23eeb21c749a205f459971dfc6e185e39851ba973ea8538262daa56020`.
- Hide every annotated opinion span in each sentence by replacing it with
  `[MASKED]`; retain the target aspect and other sentence context. Polarity is
  used only for balanced sampling, not as an outcome or correctness label.
- Apply the same score definitions and 0.1 numeric grid as Part A. Cross
  assistant-prefix field order (valence-first/arousal-first), forced values
  2.0/8.0, and the same two Qwen sizes.
- 24 recipients × 2 models × 2 orders × 2 values = 192 model-context cells.
- The source texts are unseen by this project's recorded scoring runs, but
  possible inclusion in model pretraining is unknown. They have no valence or
  arousal gold labels in this design; inference is only about paired response
  shifts under the registered forced-prefix manipulation.

### Outcomes

For each model/order, estimate the paired 8−2 expected-score shift, the
valence-first minus arousal-first difference in that shift, greedy shifts,
canonical mass, and parse failures. Use paired sentence bootstrap intervals
(10,000 draws, seed `20260976`). Treat Part B as a small transfer probe, not
an independent model-training or pretraining-contamination guarantee.

## Combined limits and stop rule

This experiment tests artificial forced coordinates rather than ordinary
ratings. Part A reuses public texts, and Part B is small and unlabelled for VA.
The questions, score descriptions, and project sample may appear in public
pretraining data. Both parts are finite-set estimates. A clean result would
justify a larger, preregistered replication with more independent texts; it
would not establish a cognitive mechanism or a family-wide scaling law. A null
or unstable result will be recorded as such. No LessWrong publication decision
is part of this protocol.

## Reproduction

Runner, analyzer, and tests are added after this protocol is frozen. Private
raw prompts and per-context model outputs stay under ignored `.context/`.
