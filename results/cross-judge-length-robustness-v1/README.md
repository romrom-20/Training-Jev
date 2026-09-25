# Experiment 032 — cross-judge length robustness

## Result

Two independent local binary judges changed their labels on the exact same
8-token-prefix and 32-token-continuation pairs used in experiment 031. Their 8-token
labels reproduced experiments 027/028 exactly on all 699 rows each. In the paired
analysis, Qwen2.5-3B's strict accuracy against the review-label proxy increased from
74.0% to 96.7% (32-minus-8 = +22.7 percentage points; source-sentence-cluster
bootstrap 95% interval +19.3 to +26.3). Phi-3 Mini increased from 69.8% to 95.9%
(+26.0 pp; +22.0 to +30.0). Every target family improved under both judges.

Qwen/Phi agreement rose from 75.8% of jointly parseable 8-token answers (653 rows)
to 99.1% at 32 tokens (693 rows). Parseability also increased. These outputs are
judgments of answer text; the sentiment in the original review is only a proxy for
what the generated text communicates.

| Judge | Accuracy at 8 | Accuracy at 32 | Paired change (95% cluster interval) |
|---|---:|---:|---:|
| Qwen2.5-3B | 74.0% | 96.7% | +22.7 pp (+19.3, +26.3) |
| Phi-3 Mini | 69.8% | 95.9% | +26.0 pp (+22.0, +30.0) |

## Interpretation

The independent judges support the conclusion that the eight-token cap materially
changed how these small evaluators read the model answers. The cross-judge
convergence is striking, but it is not human validation: both judges see the same
additional text and may share language-model biases. This follow-up reuses the same
items and generated texts as 031, and 560/699 32-token answers still hit their cap.
The review-level label does not tell us whether a given answer faithfully expresses
the review sentiment. We therefore treat this as a robust evaluator-sensitivity
result and a promising research lead, not a result about answer truth or a publication
claim.

## Artifacts

- [`analysis.json`](analysis.json): paired metrics and sentence-cluster intervals.
- [`predictions.json`](predictions.json): label-only judgments and item identifiers.
- [`manifest.json`](manifest.json), [`audit.json`](audit.json): provenance, hashes,
  and privacy checks.
- Raw review/answer text remains ignored under `.context/`.
- Frozen follow-up protocol: [`032-cross-judge-length-robustness.md`](../../docs/experiments/032-cross-judge-length-robustness.md).
