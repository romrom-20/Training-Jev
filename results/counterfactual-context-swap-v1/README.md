# Experiment 053: counterfactual review-context swap

## Result

On 217 complete recipient IDs, the finite-minus-free matched-review interaction was 0.563 VA RMSE points (95% recipient-bootstrap interval [0.342, 0.778]). This exploratory interval conditions on one frozen donor assignment.

The test keeps each target aspect and gold fixed, replacing its opinion-masked review with a length-matched masked review from another selected laptop case. A positive decoder-specific matched-review advantage means the model did better with the recipient's own review. The experiment asks whether that advantage differs between finite-grid and free-greedy decoding; it does not establish general context understanding.

The 217-case sample was balanced across negative, neutral, and positive valence buckets and is not prevalence-representative. Intervals condition on a single deterministic donor permutation. The public test may have been seen during pretraining. No raw review text, IDs, donor assignments, or item-level outputs are published.

- Protocol: `docs/experiments/053-counterfactual-review-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_counterfactual_context_swap_053.py`, `scripts/analyze_counterfactual_context_swap_053.py`
