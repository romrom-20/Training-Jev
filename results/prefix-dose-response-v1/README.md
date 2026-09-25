# Experiment 033 — where do short-answer judges stabilize?

## Result

The same greedy 32-token completions were evaluated at six nested prefix lengths.
The two binary judges' agreement rose sharply from 8 to 12 tokens, then approached
a plateau. Laya's mixed/unclear rate declined at every checkpoint:

| Prefix tokens | Qwen–Phi agreement | Laya mixed/unclear | Qwen accuracy vs review proxy | Phi accuracy vs review proxy |
|---:|---:|---:|---:|---:|
| 4 | 100.0% (492 jointly parseable) | 99.6% | 61.5% | 44.2% |
| 8 | 75.8% (653) | 45.5% | 74.0% | 69.8% |
| 12 | 92.6% (678) | 33.8% | 89.0% | 89.1% |
| 16 | 97.5% (685) | 26.9% | 94.6% | 92.6% |
| 24 | 99.0% (687) | 17.3% | 95.9% | 95.1% |
| 32 | 99.1% (693) | 12.7% | 96.7% | 95.9% |

The apparent 4-token agreement is deceptive: Phi parses only 70.4% there, Laya
marks 99.6% of prefixes mixed/unclear, and the two judges' accuracy against the
review-level proxy is modest. Agreement is not correctness. By 12 tokens, 92.6% of
jointly parseable Qwen/Phi judgments agree; around 90% of each judge's parseable
labels are already stable through the 32-token endpoint by that checkpoint. By 16,
agreement is 97.5%. These stability figures describe this run's own final label,
not the true meaning of the answer.

## Interpretation and limits

This dose response strengthens a narrow observation: very short prefixes of these
open aspect-sentiment answers produce evaluator disagreement and abstention, while
slightly longer prefixes rapidly improve cross-judge consistency on this dataset.
It does not prove that the convergent judgment is right. The gold is the source
review sentiment, not a human label for each generated answer; all three judges are
automated; and the sweep reuses the same 233-item sample and answer continuations
from 031/032. The 4-token point is a direct example of high agreement with poor
validity. No LessWrong post is warranted yet: an independent domain or
human-coded answer set is needed before making claims about semantic accuracy.

## Artifacts

- [`analysis.json`](analysis.json): all six prefix curves, cluster-bootstrap
  intervals, per-target breakdowns, and label stability.
- [`predictions.json`](predictions.json): IDs, prefix lengths, and label-only outputs.
- [`manifest.json`](manifest.json) and [`audit.json`](audit.json): provenance,
  hashes, source-label reproduction checks, and privacy audit.
- Raw answer text and token IDs remain in ignored `.context/`.
- Frozen protocol: [`033-prefix-dose-response.md`](../../docs/experiments/033-prefix-dose-response.md).
