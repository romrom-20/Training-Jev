# Local Laya decision-engine audit (experiment 030)

Laya's pinned English base checkpoint was run locally on MPS for the 233-item source screen and 60 paired generated answers. It returns four choices rather than forcing binary polarity. On source reviews it produced clear positive/negative labels for 88.0%; conditional accuracy was 94.6%, while strict all-item accuracy was 83.3%.

- granite-3.1-2b: four-way counts {'positive': 6, 'negative': 9, 'mixed': 4, 'unclear': 1}; clear-polarity coverage 75.0%; clear-only accuracy against review gold 53.3% (8/15 clear answers).
- qwen-1.5b: four-way counts {'positive': 3, 'negative': 5, 'mixed': 10, 'unclear': 2}; clear-polarity coverage 40.0%; clear-only accuracy against review gold 75.0% (6/8 clear answers).
- smollm2-1.7b: four-way counts {'positive': 11, 'negative': 6, 'mixed': 3, 'unclear': 0}; clear-polarity coverage 85.0%; clear-only accuracy against review gold 88.2% (15/17 clear answers).

The report contains IDs, labels, and probabilities, not review or answer text. Review gold is only a proxy for generated-answer meaning; Laya is an additional automated judge, not human ground truth. The model card reports weak zero-shot results on other typed-decision tasks, and this domain was not calibration-tuned. A runtime warning flagged an invalid high-option temperature entry; only four choices were used and confidence remains descriptive. These results do not justify a novelty claim or LessWrong post. Post-hoc inspection found repeated identical aspect-answer inputs in the sample; the alternate duplicate-input cluster intervals are exploratory sensitivity checks.

See `summary.json`, `analysis.json`, `source-analysis.json`, `audit.json`, `DATA_ATTRIBUTION.md`, and the [preregistered protocol](../../docs/experiments/030-laya-decision-audit.md). To reproduce, run `uv sync --extra dev --extra decision-engine`, regenerate the ignored paired answers with `scripts/prepare_human_audit_029.py`, then run `scripts/run_laya_decision_audit_030.py` and `scripts/analyze_laya_decision_audit_030.py`.
