# Experiment 060: cross-category donor control

## Result

The polarity-matched cross-category interaction was 0.401; the same-category interaction was 0.107. Their paired cross-minus-same difference was 0.294 (95% recipient-bootstrap interval [0.141, 0.448]).

The two conditions use the same 184 recipients, recipient gold values, own-review baselines, model and polarity constraints. Same-category results are from Experiment 057. Three cross-category one-to-one donor matchings keep polarity fixed while requiring a different official category. The bootstrap resamples recipient IDs and retains all mappings; it does not generalize over possible donor assignments.

The contrast tests broad official-category match on this sample. Category mismatches also change lexical and semantic content; this design cannot identify a specific word-level mechanism. There is one public benchmark split and one model family, with possible pretraining exposure. No text, IDs, donor mappings, or item-level outputs are published.

- Protocol: `docs/experiments/060-cross-category-donor-control.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_cross_category_donor_control_060.py`, `scripts/analyze_cross_category_donor_control_060.py`
