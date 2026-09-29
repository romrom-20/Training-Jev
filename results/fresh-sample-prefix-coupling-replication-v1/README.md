# Experiment 068: fresh-sample forced-prefix replication

## Result

The fresh-sample average second-coordinate shift after forcing the first score from 2.0 to 8.0 was -0.420 points (95% recipient-bootstrap interval [-0.616, -0.226]) on 64 complete recipients.

The registered replication targets the opposite-sign order-specific pattern from Experiment 067. Fresh recipients were selected from the same pinned SIGHAN release but excluded all 180 IDs from Experiment 064. Available negative non-quality categories had been exhausted by that cohort, so all 64 replication recipients are food-quality reviews, balanced 32 negative and 32 positive. This tests sample robustness within one category, not category generality.

The valence-first shift was -0.031 points (95% interval [-0.234, +0.172]); the arousal-first shift was -0.809 ([-1.120, -0.509]). The arousal-first effect matches Exp067's direction and magnitude; the valence-first effect did not replicate.

Order-specific estimates, intervals, invalid rates and replication indicators are in `summary.json`. The forced-prefix intervention measures conditional continuation sensitivity, not natural rating behavior. Raw IDs and outputs remain private in `.context/`.

- Frozen protocol: `docs/experiments/068-fresh-sample-prefix-coupling-replication.md`
- Prior result: [Experiment 067](../forced-coordinate-prefix-coupling-v1/README.md)
- Runner/analyzer: `scripts/run_fresh_sample_prefix_replication_068.py`, `scripts/analyze_fresh_sample_prefix_replication_068.py`
