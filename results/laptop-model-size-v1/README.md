# Experiment 050: Qwen2.5 model-size check on laptop reviews

## Result

With zero invalids and all 943 IDs complete, Qwen2.5-3B's finite-grid context gain was +2.299 VA RMSE points and its free-greedy gain was −0.367. Their registered interaction was +2.666 (95% source-ID interval [+2.575, +2.757]), passing the +0.25 practical rule. The descriptive paired difference in interaction from 1.5B was +2.412 (95% interval [+2.309, +2.515]).

This repeats Experiment 049's exact English laptop IDs, target aspects, gold labels, and prompt bytes at Qwen2.5-3B. It is a paired model-size diagnostic within one public benchmark split, not an independent sample or corpus replication.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/050-laptop-model-size-factorial.md`
- Runner/analyzer: `scripts/run_laptop_model_size_050.py`, `scripts/analyze_laptop_model_size_050.py`
- Aggregate result and provenance: `summary.json`
