# Experiment 066: instruction-language control on SIGHAN

## Result

The Chinese-minus-English instruction-language difference in order moderation was -0.118 VA-RMSE points (95% paired recipient-bootstrap interval [-0.529, 0.296]) on 81 complete IDs.

The model-specific aggregate order moderations were 0.096 under the English prompt and -0.023 under the translated Chinese prompt. Their intervals and the registered difference are in `summary.json`.

This paired test holds SIGHAN reviews, recipients, donor maps, output orders and decoders fixed while changing prompt language. It uses a deterministic 88-review subset and 2,464 new Qwen2.5-3B generations. Raw text, IDs, maps and predictions remain private in `.context/`.

The translation was not independently rated by a professional translator. A prompt-language effect here would not establish that language caused the earlier English/Chinese benchmark difference, because the releases and domains also differ. Results apply to this selected cohort and model setup.

- Frozen protocol: `docs/experiments/066-sighan-instruction-language-control.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_instruction_language_066.py`, `scripts/analyze_sighan_instruction_language_066.py`
- Prior SIGHAN tests: [Experiment 064](../sighan-output-key-order-replication-v1/README.md), [Experiment 065](../sighan-order-model-size-replication-v1/README.md)
