# Experiment 044: opinion-mask word-order control

## Result

Shuffled-minus-natural masked RMSE was -0.165 (95% cluster interval [-0.301, -0.027]); shuffling slightly improved the score. The shuffled-versus-aspect-only RMSE gain was 2.009 (95% interval [1.854, 2.166]); the exploratory order-contribution rule did not pass.

This is an adaptive within-sample follow-up to the positive Experiment 043 result, not independent confirmation. The word-order shuffle keeps each sentence's whitespace-token multiset, including `[MASKED]` placeholders, but disrupts syntax and discourse. The same Qwen checkpoint and 0.1-grid constrained decoder are used. A null order contrast does not prove bag-of-words sufficiency.

No review text, aspect strings, IDs or item-level outputs are included; private material remains in ignored `.context/`.

- Protocol: `docs/experiments/044-opinion-mask-word-order-control.md`
- Runner/analyzer: `scripts/run_opinion_mask_word_order_control_044.py`, `scripts/analyze_opinion_mask_word_order_control_044.py`
- Aggregate result and provenance: `summary.json`
