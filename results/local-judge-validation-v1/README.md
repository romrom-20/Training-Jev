# Local semantic judge: source-label screen (experiment 027)

The cached Qwen2.5-3B-Instruct judge passed the preregistered source-label gate on
233 SemEval restaurant aspect items: 96.1% accuracy, 98.2% negative recall, and
99.6% exact-answer parseability. The gate required at least 90% accuracy and 80%
negative recall. The capture used the local 24 GB MacBook Air at float32; no API or
cloud inference was used.

This result permits the conditional natural-completion evaluation in the frozen
[protocol](../../docs/experiments/027-local-judge-validation.md). It does not by
itself establish that the judge is valid on generated text. `analysis.json` stores
only stimulus IDs, aspect categories, gold labels, and judge labels; it contains
no review text or completions.

Reproduce the source-label screen from the repository root with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/local_judge_validation.py
```
