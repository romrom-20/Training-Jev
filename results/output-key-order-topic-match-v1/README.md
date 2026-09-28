# Experiment 062: output key order and topic match

## Result

With arousal-first JSON, the cross-minus-same category interaction was 0.037 (95% recipient-bootstrap interval [-0.115, 0.193]). Its change from the valence-first estimate was -0.258 (95% paired interval [-0.449, -0.061]).

The run reversed only the numeric JSON key order. Same-category and cross-category donors use the same recipients, polarities, assignments, target aspects and own-review baselines. The key-order comparison reuses the already observed valence-first outputs, so it is a registered robustness follow-up rather than independent replication.

The coordinate-wise diagnostic was not preregistered. It suggests that the moderation is concentrated in arousal: the arousal-axis moderation was -0.447 (95% paired recipient-bootstrap interval [-0.652, -0.235]), while the valence-axis interval included zero. Treat this as hypothesis-generating; Experiment 063 will test it on a separate English restaurant sample.

General output-format sensitivity is already studied; this experiment tests only whether one key-order change alters the continuous-VA decoder-by-topic-match effect on this public laptop split and Qwen2.5 family. Donor categories differ in lexical and semantic content. No text, item IDs, donor maps, or individual predictions are published.

- Protocol: `docs/experiments/062-output-key-order-topic-match.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_output_key_order_topic_match_062.py`, `scripts/analyze_output_key_order_topic_match_062.py`
