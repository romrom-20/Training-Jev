# Experiment 068 — Fresh-sample replication of forced-prefix coupling

**Status:** Laya-selected after Experiment 067; preregister before inference.
Laya's choice scores are uncalibrated agenda triage only.

## Question

Does the opposing order-specific continuation pattern from Experiment 067 recur
on fresh SIGHAN recipients when the model, Chinese scoring prompt and forced
assistant-prefix intervention are held fixed?

In Exp067, setting the first output coordinate to 8.0 rather than 2.0 changed
the second coordinate by +0.156 points when valence came first (95% interval
[-0.047, +0.359]) and -0.688 points when arousal came first ([-0.984, -0.391]).
The equal-weighted primary across orders was -0.266 ([-0.469, -0.070]). This
fresh-recipient replication tests whether the order asymmetry survives new
reviews; it does not test a new effect or claim a general LLM property.

## Cohort and prompts

- Start from the 1,916 eligible unique recipient IDs in the pinned SIGHAN source
  used by Experiment 052. Exclude all 180 IDs selected in Experiment 064.
- In the remaining IDs, take 32 negative and 32 positive food-quality reviews,
  each selected in SHA-256 order using salt
  `exp068-fresh-prefix-replication-v1|case_id`:
  - Negative `食物#品质`: 32.
  - Positive `食物#品质`: 32.
  This yields a polarity-balanced fresh sample in one shared target category,
  disjoint from all preceding SIGHAN cohorts. The 064 sample exhausted the
  eligible negative reviews in several other categories, so the proposed
  six-cell replication could not be formed; this focused replication does not
  test category generality. Sorted selected IDs joined by newline have
  SHA-256 `1e738712d6ec47bb90a5bdd47d676f3b79fcb958b9c0d945d3081428924132bd`.
- Reuse Experiment 067's exact Chinese scoring template and partial-assistant
  prefix construction. Mask opinions as in Experiments 066–067. No donor context
  is used.
- Model: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy MPS inference on the 24-GB
  MacBook Air; no cloud inference.

## Intervention and generation

For every recipient, generate the same four cells as Exp067:

| Output order | Forced first coordinate | Assistant output prefix |
|---|---:|---|
| valence then arousal | 2.0 | `{"valence": 2.0, "arousal": ` |
| valence then arousal | 8.0 | `{"valence": 8.0, "arousal": ` |
| arousal then valence | 2.0 | `{"arousal": 2.0, "valence": ` |
| arousal then valence | 8.0 | `{"arousal": 8.0, "valence": ` |

The prefix is appended after Qwen's normal assistant-generation marker, not
added to the user instruction. Use deterministic free-greedy continuation, a
40-token cap, strict parsing, no retries and no repairs. Save raw outputs only in
ignored `.context/`. Total: 64 × 2 output orders × 2 forced values = **256**.

Before loading the target model, verify protocol, source and Exp064 cohort hashes;
verify the exact fresh-ID sample hash and disjointness; and check all four cells
per recipient plus the assistant-prefix/chat-template boundary. Freeze protocol,
runner, analyzer and tests in git before inference.

## Registered analysis

For each output order, calculate the recipient-paired second-coordinate shift
`prediction(second | first=8.0) - prediction(second | first=2.0)`. The primary
endpoint is the equal-weighted mean of the two order-specific shifts. Bootstrap
recipient IDs with replacement, retaining all four cells together; use 10,000
replicates and seed `20260968`. Report each order-specific effect and percentile
95% interval alongside the primary.

If any forced-value × output-order arm has an invalid-output rate above 2%,
withhold the score contrasts. Otherwise, analyze IDs complete in all four cells
and report invalid counts. The key replication question is whether the order-
specific estimates preserve the signs observed in Exp067: positive valence-first
and negative arousal-first. An interval including zero means the fresh sample did
not resolve that order-specific effect; it does not show equivalence. A replicate
would remain a finding about forced continuation on this model and prompt, not
ordinary affect ratings or the earlier natural output-order moderation.

## Compute and provenance

The source input and gold hashes are the pinned SIGHAN hashes in Experiment 052.
The excluded Experiment 064 output artifact SHA-256 is
`48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7`. The prefix
construction and prompt are bound to Experiment 067's protocol SHA-256
`5b9fa3cf3649b1c935dd4f7ed8b20f1c122bef2a7ca2060903a7928b7645f932`. Publish
aggregate results and hashes only. Public-test pretraining exposure cannot be
ruled out.
