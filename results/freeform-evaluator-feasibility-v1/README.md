# Free-form evaluator feasibility pilot

A light aspect-aware TF-IDF classifier was fit on 6,144 positive/negative
MAMS-ATSA training examples and evaluated without fitting on MAMS validation/test
and the 233-item SemEval restaurant set. It achieved 79.3% macro-F1 on MAMS test
and 78.6% on SemEval transfer; SemEval negative recall was 70.9%.

This is a failed feasibility check for using this classifier as a semantic judge.
It evaluated original review text, not generated answers, so it does not resolve
experiment 025's open-output measurement problem. The pilot was exploratory and
not preregistered. It stores item IDs and labels/predictions only, with no review
text or fitted model.

Reproduce from the repository root:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/freeform_evaluator_feasibility.py \
  --output /tmp/freeform-evaluator-feasibility
```

The checked-in `analysis.json` was produced with that runner. Dataset hashes,
per-item predictions, and full metrics are included there. The method and its
limitations are described in the
[experiment note](../../docs/experiments/026-freeform-evaluator-feasibility.md).
