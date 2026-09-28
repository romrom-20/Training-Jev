# Experiment 049: English laptop-domain decoder transfer

## Result

On the English laptop domain, the finite-grid context gain was 0.740 (95% source-ID interval [0.684, 0.796]) and the free-greedy gain was 0.486 (95% interval [0.425, 0.548]). The primary decoder interaction was 0.254 (95% interval [0.196, 0.313]); its practical gate passed.

The design is an adaptive transfer test on the full eligible English laptop test split in the pinned DimABSA release. It uses the same Qwen2.5-1.5B model, one-decimal prompt, 6,561-value finite grid, and free-greedy parser as Experiment 048. This is a separate source-domain sample, but it is not an independent corpus or blind evaluation.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/049-laptop-domain-decoder-transfer.md`
- Runner/analyzer: `scripts/run_laptop_decoder_transfer_049.py`, `scripts/analyze_laptop_decoder_transfer_049.py`
- Aggregate result and provenance: `summary.json`
