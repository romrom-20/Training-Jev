# Experiment 051: Qwen2.5-3B restaurant-domain transfer

## Result

On English restaurant reviews, the finite-grid context gain was 2.428 (95% source-ID interval [2.317, 2.541]) and the free-greedy gain was -0.348 (95% interval [-0.415, -0.281]). The registered interaction was 2.776 (95% interval [2.679, 2.872]); its practical gate passed. The descriptive laptop-minus-restaurant interaction difference was -0.111 (95% independent-domain bootstrap interval [-0.241, 0.023]).

This uses the full eligible English restaurant test split with the same 3B model, prompts and decoders as the English laptop run. It is a same-language product-domain transfer within one public DimABSA release, not an independent corpus replication.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/051-restaurant-domain-decoder-transfer-3b.md`
- Runner/analyzer: `scripts/run_restaurant_domain_decoder_051.py`, `scripts/analyze_restaurant_domain_decoder_051.py`
- Aggregate result and provenance: `summary.json`
