# 033 — At what answer length do judge readings stabilize?

## Status

Post-result dose-response follow-up to experiments 031–032. Freeze this protocol
before collecting any new prefix judgments. It reuses the same items and complete
32-token continuations; it is not an independent sample replication.

## Question

Experiment 031 showed that Laya's mixed/unclear rate falls between 8 and 32 tokens.
Experiment 032 showed that two binary judges' agreement and review-label-proxy
accuracy rise on the same continuation pairs. Where along the generated text do
those changes happen? Use nested prefixes from the exact same greedy 32-token
continuations, so tokens added at each step are the only input change.

## Frozen design

- Reuse the 699 target/item 32-token generations from experiment 031. Verify the
  private-generation file hash. For each output, decode prefixes at token budgets
  4, 8, 12, 16, 24, and 32. If generation reached EOS before a prefix boundary,
  reuse its completed text at later boundaries. Do not regenerate target answers.
- For intermediate lengths 4, 12, 16, and 24, classify each prefix with the exact
  Laya four-way typed choice from experiment 030 and exact Qwen2.5-3B/Phi-3 Mini
  binary judge prompts from 027/028/032. Same neutral Laya option mapping per item.
  Reuse already collected 8/32 labels from 031/032; check them byte-for-byte at
  the label level when assembling the analysis.
- Report by budget and target: Laya mixed/unclear rate, Qwen/Phi polarity-label
  agreement, parseability, and each judge's strict accuracy against the review-label
  proxy. The primary curve is Qwen/Phi agreement across the six nested prefixes.
  Secondary: the earliest tested prefix from which each binary judge's label stays
  equal to its own 32-token label at all later checkpoints. Treat this as within-run
  label stability, not correctness.
- Bootstrap paired curves by source sentence (10,000 resamples; seed `20260933`),
  keeping all target-model rows for a sentence together. Show point estimates and
  descriptive 95% intervals. Do not treat repeated prefixes as independent samples.
- No threshold or p-value claim. Report actual prefix token counts, especially for
  EOS-terminated generations. The source review label is only a proxy for answer
  meaning, and cross-engine agreement is not human validation.

## Privacy and interpretation

Answer/review text and token IDs remain in ignored `.context/`. Tracked files may
contain only IDs, prefix lengths, labels, probabilities, and summaries. A rapid rise
in agreement would locate when these evaluators become consistent under this prompt;
it would not show that the resulting consensus is correct. A flat or nonmonotone
curve would be equally informative about the limits of using short-answer judges.
No LessWrong decision follows from this experiment alone.
