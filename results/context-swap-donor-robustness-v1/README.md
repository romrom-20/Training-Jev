# Experiment 054: review-donor assignment robustness

## Result

Across three fixed donor permutations and 217 complete IDs, the mean finite-minus-free matched-review interaction was 0.443 VA RMSE points (95% recipient-bootstrap interval [0.277, 0.611]). The three permutation estimates ranged from 0.381 to 0.477; this range is descriptive.

Each target aspect and gold stayed fixed while its opinion-masked review was replaced by a review from another case. Three independent deterministic donor derangements test whether Experiment 053's finite-decoder sensitivity depends on one arbitrary assignment. Positive per-decoder matched-review advantage means the recipient's own review predicts better than donor reviews.

This follow-up does not control coarse gold-valence polarity, and its intervals condition on the three mappings. The 217-case subset is polarity-balanced rather than prevalence-representative. Public test examples may have appeared in model training data. No review text, IDs, donor mappings, or item-level predictions are published.

- Protocol: `docs/experiments/054-context-swap-donor-robustness.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_context_swap_donor_robustness_054.py`, `scripts/analyze_context_swap_donor_robustness_054.py`
