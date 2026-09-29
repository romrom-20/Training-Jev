# Experiment 063: output key order on restaurant reviews

## Result

The registered arousal-specific order moderation was -0.204 VA-RMSE points (95% recipient-bootstrap interval [-0.487, 0.067]) on 172 complete restaurant recipients.

The preregistered primary contrast subtracts valence's JSON-order moderation from arousal's, where order moderation is the change in the same-category-minus-cross-category context-gain decoder interaction when moving from valence-first to arousal-first output. The negative direction follows the algebraically matched coordinate contrast from Experiment 062; the frozen protocol's prose direction was sign-reversed. This is a fresh recipient sample within DimABSA's English restaurant split; it is not an independent corpus. Valence-first own-review baselines are reused from Experiment 051. Category crossings also change review content and meaning.

Coordinate-wise order moderations were -0.419 for valence (95% interval [-0.764, -0.076]) and -0.623 for arousal ([-0.902, -0.345]). Both moved in the same direction as Experiment 062's coordinate audit. The preregistered protocol text said a positive primary estimate would follow 062; that prose had the sign reversed relative to its stated formula. See `docs/experiments/063-analysis-sign-audit.md`. These coordinate outcomes are secondary; only the primary coordinate-difference contrast was confirmatory.

The protocol was frozen before inference. No item text, case IDs, donor maps or individual predictions are released.

- Protocol: `docs/experiments/063-output-key-order-restaurant-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_output_key_order_restaurant_063.py`, `scripts/analyze_output_key_order_restaurant_063.py`
- Related: [Experiment 062](../output-key-order-topic-match-v1/README.md), [Experiment 051](../restaurant-domain-decoder-transfer-v1/README.md)
