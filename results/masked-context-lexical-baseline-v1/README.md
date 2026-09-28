# Experiment 046: sparse lexical baseline for masked VA context

## Result

The sparse model's aspect-only minus opinion-masked RMSE gain was 0.154 (95% source-ID cluster interval [0.101, 0.209]); the preregistered lexical-gain rule did not pass.
The CPU run took 0.9 minutes.

This adaptive diagnostic uses per-language word/character TF-IDF and ridge models trained on the official DimABSA training split. Ridge regularization was selected by grouped cross-validation over training review IDs only. Evaluation reuses the 217 held-out source IDs from Experiments 043–045 and is not an independent corpus replication. The sparse model is not a pretrained LLM; a positive gain would show predictive residual lexical information, not causal sufficiency.

The comparison with Qwen2.5-3B's Experiment 043 gain is descriptive. No review text, aspects, IDs, or per-item predictions are published; private predictions remain in ignored `.context/`.

- Protocol: `docs/experiments/046-masked-context-lexical-baseline.md`
- Runner/analyzer: `scripts/run_masked_context_lexical_baseline_046.py`, `scripts/analyze_masked_context_lexical_baseline_046.py`
- Aggregate result and training/provenance record: `summary.json`
