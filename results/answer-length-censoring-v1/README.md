# Experiment 031 — answer-length censoring

## Result

All 699 8-token generations (233 items × 3 target models) hit the generation cap;
none emitted EOS before the cap. The cap-hit answers were genuine prefixes of the
32-token continuations. Laya marked 45.5% of the short answers mixed/unclear and
12.7% of their 32-token continuations mixed/unclear: a paired difference of
−32.8 percentage points (source-sentence-cluster bootstrap 95% interval −36.9 to
−28.6 pp; 112 sentence clusters). The direction held for every target:

| Target | Mixed/unclear, 8 tokens | Mixed/unclear, 32 tokens | Paired difference (95% cluster interval) | Still capped at 32 |
|---|---:|---:|---:|---:|
| Granite 3.1 2B | 45.1% | 8.6% | −36.5 pp (−43.3, −29.7) | 214/233 |
| Qwen2.5 1.5B | 61.4% | 16.3% | −45.1 pp (−52.4, −37.9) | 229/233 |
| SmolLM2 1.7B | 30.0% | 13.3% | −16.7 pp (−22.6, −11.2) | 117/233 |

The change is not just from reaching a natural end at 32 tokens: 560/699 longer
answers still hit the 32-token cap. Yet their Laya label distribution had already
shifted. The 8-token Laya labels exactly reproduced experiment 030 on all 60
overlapping generated answers.

## What this does and does not show

This is evidence that an 8-token generation cap materially changes **Laya's
classification** of these model answers. It is not evidence that the 32-token label
is the correct reading, and the 32-token responses are usually still truncated.
The experiment manipulates cap and added content together, so it does not isolate a
generic preference for verbosity from the arrival of sentiment-bearing words. The
review label is only a proxy for the generated answer's meaning. Laya's source-screen
performance varies by aspect, especially price. Cross-engine replication is the
next test; no publication decision follows yet.

## Artifacts and reproduction

- [`analysis.json`](analysis.json): preregistered paired outcome, transitions,
  target-specific cluster intervals, and limitations.
- [`predictions.json`](predictions.json): item-level label-only outputs; no reviews,
  answers, or token IDs.
- [`manifest.json`](manifest.json) and [`audit.json`](audit.json): model revisions,
  hashes, runtime notes, and privacy/protocol checks.
- Raw text and token IDs remain only in ignored `.context/length-censoring-031/`.
- Frozen protocol: [`031-answer-length-censoring.md`](../../docs/experiments/031-answer-length-censoring.md).

The local run used an MPS MacBook Air, one target checkpoint loaded at a time,
`laya==0.3.20`, and the frozen Laya weights revision listed in `manifest.json`.
