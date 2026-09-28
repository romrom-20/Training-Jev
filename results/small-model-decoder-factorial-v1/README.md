# Experiment 048: matched-prompt decoder comparison at 1.5B

## Result

With identical prompt wording, the finite-grid context gain was 0.288 (95% source-ID interval [0.180, 0.398]) and the free-greedy gain was 0.130 (95% interval [-0.002, 0.264]). Their preregistered interaction was 0.159 (95% interval [0.064, 0.253]); the interaction rule did not pass.
Generation took 28.6 minutes after model load.

Both decoder arms used the same Qwen2.5-1.5B checkpoint, sample, aspect-only/opinion-masked inputs, and one-decimal prompt wording. Only the finite 6,561-value grammar differs from free greedy generation. This adaptive test reuses the same DimABSA release and held-out IDs; it is a within-sample decoder diagnostic, not an independent replication.

No review text, aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/048-small-model-decoder-factorial.md`
- Runner/analyzer: `scripts/run_small_model_decoder_factorial_048.py`, `scripts/analyze_small_model_decoder_factorial_048.py`
- Aggregate result and provenance: `summary.json`
