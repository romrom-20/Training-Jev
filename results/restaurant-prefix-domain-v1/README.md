# Experiment 072: restaurant-domain prefix coupling

## Result

The registered 64-recipient experiment completed locally on MPS: 768 model-context
cells (three pinned models × 64 IDs × four forced prefixes), zero invalid greedy
outputs, and no resumed cells. The raw prompt/output rows remain private under
`.context/`; the public aggregate is in `summary.json`.

For valence-first prefixes, the 8−2 change in expected second score was +1.934
(95% recipient-bootstrap interval [+1.852, +2.016]) for Qwen2.5-1.5B,
+0.935 [+0.812, +1.063] for Qwen2.5-3B, and +1.836 [+1.750, +1.922] for
SmolLM2-1.7B. For arousal-first prefixes it was +1.247 [+1.032, +1.460],
+0.045 [−0.245, +0.353], and +1.229 [+1.147, +1.311], respectively. The
within-recipient Qwen 1.5B-minus-3B contrast was +0.998 [+0.841, +1.155]
valence-first and +1.202 [+0.936, +1.458] arousal-first.

The descriptive, independent-bootstrap restaurant-minus-laptop contrasts against
Experiment 071 were small relative to their intervals: Qwen 1.5B −0.062
[−0.189, +0.066] and +0.185 [−0.107, +0.472]; Qwen 3B +0.164
[−0.036, +0.359] and +0.272 [−0.110, +0.662]; SmolLM2 +0.022
[−0.102, +0.146] and −0.109 [−0.219, +0.000]. This looks compatible with
similar expected-shift magnitudes across these two DimABSA domains, but it is not a
formal equivalence result and the restaurant IDs were reused from Experiment 051.

A repeated decoding gap also appears: SmolLM2's greedy score shifts (+3.734 and
+2.531) exceed its restricted-grid expected shifts (+1.836 and +1.229). The same
pattern appeared on laptops in 071 (+4.016 and +2.891 greedy versus +1.814 and
+1.338 expected). Qwen's greedy and expected shifts track more closely. This
post-hoc cross-domain pattern motivates a targeted follow-up; it is not evidence
that either statistic is better calibrated to human ratings.

## Interpretation and limits

The outcome is consistent with a bounded observation: on two English DimABSA
product domains, the conditional numeric score distribution moves with a forced
coordinate in Qwen 1.5B and SmolLM2; Qwen 3B moves less, especially when arousal
comes first. The restaurant sample repeats the treatment on IDs previously scored
in 051, so it adds domain-condition evidence but no fresh-item replication. Prefixes
are artificial; no ordinary user-rating behavior or universal scale law follows.

Prior work already compares expected-value scores with greedy scores for language
model evaluation and discusses positional bias, so that general contrast is not
novel. See [Zawistowski (2024)](https://arxiv.org/abs/2406.10267). Work on modes
and output distributions also cautions against equating a mode with the whole
conditional distribution; see [Yoshida et al. (ACL 2024)](https://aclanthology.org/2024.acl-long.855/).
The next useful question is whether the model-specific gap depends on the answer
being supplied as a partial assistant continuation, rather than as user-provided
semantic information.

## Reproduction

- Frozen protocol: [`072`](../../docs/experiments/072-restaurant-prefix-domain-replication.md)
- Runner: `scripts/run_restaurant_prefix_domain_072.py`
- Analyzer: `scripts/analyze_restaurant_prefix_domain_072.py`
- Public aggregate: [`summary.json`](summary.json)
- Source split and models are pinned in the protocol. MPS; wall time 1,424.7 s.
