# Experiment 064: output order × category match on SIGHAN 2024

## Result

The registered aggregate output-order moderation was -0.077 VA-RMSE points (95% recipient-bootstrap interval [-0.331, 0.170]) on 173 complete recipients.

This was selected as an independent release and language check after two English DimABSA tests. The interval includes zero, so the registered test does not show that the negative aggregate moderation transfers to SIGHAN. It remains compatible with modest effects in either direction; this alone does not establish a true release or language difference.

Coordinate-specific order moderations are descriptive secondary outcomes: valence -0.006, arousal -0.135. Raw texts, IDs, donor maps and item predictions remain private in the ignored `.context/` directory. Only aggregates and provenance hashes are released.

The test crosses same- versus cross-category polarity-matched donor context, valence-first versus arousal-first JSON key order, and finite-grid versus free-greedy decoding. It uses a fixed 180-review balanced sample, 4,680 new generations, and the frozen Experiment 052 valence-first own-review baseline. Cross-category donors alter review wording and meaning as well as category. Results apply to this prompt and Qwen2.5-3B setup.

- Frozen protocol: `docs/experiments/064-sighan-output-key-order-replication.md`
- Public aggregate and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_output_key_order_064.py`, `scripts/analyze_sighan_output_key_order_064.py`
- Prior English-domain tests: [Experiment 062](../output-key-order-topic-match-v1/README.md), [Experiment 063](../output-key-order-restaurant-replication-v1/README.md)
- Source release: [Lee et al. (2024)](https://aclanthology.org/2024.sighan-1.19/)
