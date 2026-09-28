# Experiment 062 — output key order and donor-topic match

**Status:** preregistered before new target-model inference. Laya's pinned local
triage selected the reversed-key-order robustness test (choice A; uncalibrated
probability 0.297).

## Question

Does the same-category versus cross-category decoder interaction from Experiment
060 persist when the numeric JSON output order is reversed from valence-first to
arousal-first?

## Design

- Use the same 184 recipient cases (89 negative, 95 positive), same-category
  donor maps from 057, cross-category donor maps from 060, and exact own-review
  baseline cases. Verify the frozen prediction-file hashes before constructing
  jobs. Keep aspect, gold, donor ID, and assignment fixed across output-order arms.
- Change only the requested/finite output serialization from
  `{"valence":v,"arousal":a}` to `{"arousal":a,"valence":v}`. Keep the same
  one-decimal 1–9 values, 6,561 grid points, continuous target meaning, prompt
  semantics for the review condition, free-greedy 40-token cap, and no retries.
  Parse finite outputs back to canonical `[valence, arousal]`; free JSON is parsed
  by key name and also returned canonically.
- Generate arousal-first responses for own `opinion_masked` reviews (368 outputs),
  same-category donor reviews (1,104), and cross-category donor reviews (1,104):
  2,576 new Qwen2.5-3B judgments total. Use the same pinned model and local MPS
  inference. The valence-first outputs are already observed in 050, 057, and 060.
- Primary endpoint: on the 184 paired recipients, calculate the average over
  three maps of `[finite-minus-free matched-review interaction(cross-category)]`
  minus the same interaction for same-category donors, using the new arousal-first
  runs. Positive means crossing the category increases the finite-grid advantage
  penalty relative to a same-category donor. Use 10,000 recipient-bootstrap draws,
  seed `20260962`, retaining both donor conditions, assignments, decoders, own
  baselines, and VA dimensions.
- Secondary endpoint: compare the arousal-first cross-minus-same estimate with the
  already observed valence-first cross-minus-same estimate from 057/060 using a
  paired recipient bootstrap, seed `20260963`. This is a key-order moderation
  diagnostic on the same sample and maps, not an independent replication.
- If more than 2% of the 1,104 new donor-context free-greedy outputs are invalid,
  withhold score contrasts. Report assignment-specific estimates and all invalid
  counts. Use recipient ID as the bootstrap unit; intervals condition on the fixed
  donor maps.

## Limits

This tests one simple key-order change after observing the original-order results.
It does not establish general serialization robustness, isolate a mechanism, or
generalize beyond this public laptop split and Qwen2.5 family. Existing work already
studies numeric-token, whitespace, structured-output, and positional format effects;
the narrow question here is whether one such surface change alters the observed
continuous-VA topic-match interaction. Donor topic differences still change lexical
and semantic content as well as official category. No LessWrong post will be written
or published until this experiment is complete.

- Same-category comparator: [Experiment 057](../../results/category-matched-context-swap-v1/README.md)
- Cross-category comparator: [Experiment 060](../../results/cross-category-donor-control-v1/README.md)
- Runner/analyzer: `scripts/run_output_key_order_topic_match_062.py`,
  `scripts/analyze_output_key_order_topic_match_062.py`
