# Exploratory pilot: can a small local classifier evaluate free-form answers?

## Question

Experiment 025 could not score most open answers because they did not contain an
explicit `positive` or `negative` token. Before adding a semantic evaluator, test
whether a cheap aspect-aware classifier transfers well enough to be trusted. This
is an evaluator feasibility check, not a model-behavior experiment.

## Procedure

- Fit a word 1–2 gram TF-IDF plus balanced logistic-regression classifier on the
  positive and negative aspect-term examples in MAMS-ATSA train (6,144 instances).
- Represent each training instance as its aspect term followed by its full review.
- Evaluate without adaptation on MAMS-ATSA validation (728), MAMS-ATSA test (729),
  and the 233 multi-aspect SemEval 2014 Restaurant examples used in experiments
  023–025. For SemEval, the category (`food`, `service`, or `price`) is the aspect.
- Keep only row identifiers and predictions in the result bundle; do not persist
  source text or train-set model weights.

This pilot was run before a written protocol and is therefore exploratory. The
source-review transfer is only proxy evidence for classifying a model's generated
answer, which may use a different register and may omit sentiment cues.

## Outcome

The exact, reproducible metrics and data hashes are in
[`analysis.json`](../../results/freeform-evaluator-feasibility-v1/analysis.json).
MAMS test macro-F1 was about 0.793. Transfer to SemEval reached about 0.786
macro-F1, with negative recall around 0.709. The classifier misses too many
negative aspects to serve as a reliable semantic judge for the open answers.
No generated answers were classified, and experiment 025's interpretation is
unchanged.

Because this pilot was not preregistered, the thresholds are decision guidance for
future work rather than a frozen test: a candidate evaluator should reach at least
0.85 macro-F1 and 0.80 negative recall on an untouched aspect-level validation set,
and should also be validated on human-labeled short answer summaries before use on
model generations.

## Decision

Do not use this classifier to score open generations. A follow-up needs a judge
validated specifically on short, natural answer text, ideally with blinded human
labels or a benchmark of human-written aspect summaries. Agreement between local
LLM judges alone would not establish correctness. Any future generated-answer
study should freeze the judge and its acceptance criteria before inspecting
target outputs.

Reproduce with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/freeform_evaluator_feasibility.py
```
