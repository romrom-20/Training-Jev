# Experiment 057: category-matched context swap

## Result

With both donor polarity and official aspect category matched, the mean finite-minus-free matched-review interaction was 0.107 VA RMSE points (95% recipient-bootstrap interval [-0.086, 0.296]). The fixed assignment estimates ranged from 0.055 to 0.186; 3 unique mappings were represented.


A **post-hoc, same-recipient comparison** against Experiment 056's polarity-only donors estimated a category-control minus polarity-only interaction of -0.304 (95% recipient-bootstrap interval [-0.463, -0.143]). This contrast was not preregistered and is exploratory.

Donor reviews share the recipient's gold-valence polarity and official DimABSA aspect category, while the recipient's own aspect and gold remain fixed. The control tests whether the decoder-specific own-review advantage persists when donor and recipient are about the same broad topic and sentiment.

Only 184 of the 217 fresh cases could be swapped without self-donation inside their category/polarity cell; 33 singleton cases were excluded by rule. The test contains negative and positive cases only, uses one public split and one model family, and may overlap model pretraining. Category matching does not identify which exact contextual words matter. No review text, IDs, categories by case, donor mappings, or item-level predictions are published.

- Protocol: `docs/experiments/057-category-matched-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_category_matched_context_swap_057.py`, `scripts/analyze_category_matched_context_swap_057.py`
