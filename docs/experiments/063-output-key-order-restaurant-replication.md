# Experiment 063 — Does output-field order selectively alter arousal context sensitivity on restaurant reviews?

**Status:** Laya-selected, externally transferred replication, preregistered before new target-model inference. Laya's probabilities are uncalibrated agenda triage only. The fresh English laptop split did not contain enough unused negative cases for a balanced test; the source was therefore moved to the separate English restaurant split already used for domain transfer in 051.

## Motivation and literature boundary

In 062, reversing only the output JSON key order changed the same-recipient cross-minus-same-category decoder interaction from +0.294 to +0.037 VA RMSE points (paired recipient-bootstrap 95% interval for the change [-0.449, -0.061]). A post-hoc coordinate audit located the change in arousal rather than valence. These laptop findings share recipients and donor maps and need independent-domain confirmation.

General LLM sensitivity to input ordering has been studied (Guan et al., 2025, [The Order Effect](https://arxiv.org/abs/2502.04134)). Dimensional aspect sentiment systems also produce continuous VA scores in structured formats (Zhou et al., 2026, [TeleAI at SemEval-2026 Task 3](https://aclanthology.org/2026.semeval-1.233/)). Those works do not test whether output-key position moderates the decoder-by-context-match interaction in continuous aspect VA scoring. This experiment estimates that narrow interaction; it does not claim general order effects are novel.

## Sample and donor maps

- Source: the pinned English restaurant Task 2 test split from DimABSA release commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256 `e90386169422c84ddd42d51029f23934b54724fafe270b65138836067492857d`; official Task 3 category annotations for the same IDs, SHA-256 `a24cd3d6b8b2dd3bcebf5016f2e8bb90736fe2e89671aced73229062ca516f79`.
- Start with the 963 eligible IDs selected by Experiment 051. Match Task 3 categories by exact text and `(Aspect, Opinion, VA)` tuple. Exclude neutral valence; within negative (`V < 4.5`) and positive (`V > 5.5`) strata, sort by SHA-256 of `exp063-rest-sample-v1|case_id`, select the first 90 per polarity, then exclude cases whose selected polarity/category cell is a singleton. The frozen selection is expected to contain 173 recipients (86 negative, 87 positive); preflight must verify these exact values before inference.
- Construct three same-category, same-polarity donor derangements within each `(polarity, category)` cell, and three cross-category, same-polarity one-to-one donor assignments within each polarity. Both assignment types minimize absolute tokenizer-token-length differences, using deterministic hash tie-breaking. Verify no self-donors, each mapping is a permutation, all same-category assignments share category, all cross-category assignments differ in category, and all donors preserve recipient valence-polarity.
- The donor mapping is fixed across output order and decoder. Review text is passed through the registered all-opinion mask before use. No individual text, case ID, donor map, prompt, or prediction is published.

## Conditions and generation

- Donor conditions: same official category and cross official category. Cross-category changes semantic content as well as its official label; this is a category-context manipulation, not a pure label intervention.
- JSON order conditions: `{"valence":v,"arousal":a}` and `{"arousal":a,"valence":v}`. Values, one-decimal grid, prompt, model, temperature/greedy settings and parser remain fixed; parsed predictions are always canonical `[valence, arousal]`.
- Generate both decoders under both orders for each of three maps and both donor conditions. Reuse Experiment 051's exact valence-first own-review baseline outputs on the selected recipients; generate the arousal-first own-review baseline anew. Total new generations: 4,498 (4,152 donor-context outputs and 346 arousal-first own-review outputs).
- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS, 24-GB Apple Silicon laptop. Finite-grid uses the 6,561 registered outputs; free greedy has a 40-token cap and strict parser. No retries, repair, or regeneration of invalid outputs.

## Estimands and analysis

For each output order, VA coordinate, donor condition and decoder, calculate review-context gain as `RMSE(own masked review) - RMSE(donor masked review)`; positive means the recipient's own review is better. Then calculate the decoder interaction as finite-grid context gain minus free-greedy context gain. The topic-match effect is same-category interaction minus cross-category interaction (same minus cross is positive when category matching preserves a larger finite-grid advantage).

For each VA coordinate separately, define order moderation as `topic-match effect(arousal-first) - topic-match effect(valence-first)`. **Primary endpoint:** arousal order moderation minus valence order moderation. A positive value in the direction seen in 062 means moving arousal to the first JSON position changes the category-by-decoder interaction more than the analogous valence-coordinate interaction. This coordinate × order × donor-category × decoder contrast is tested with 10,000 recipient-cluster bootstrap draws, seed `20260963`, retaining every condition, decoder, coordinate and fixed map within each sampled ID.

Secondary outcomes: the two coordinate-specific order moderations with intervals; the all-VA same-minus-cross topic-match effect under each order; assignment-specific estimates; and invalid-output counts by cell. Bootstrap intervals are descriptive for secondary outcomes. If more than 2% of free-greedy donor-context outputs are invalid, withhold all score contrasts. Otherwise analyze only complete recipient IDs and report coverage. A successful result would be replication on a different domain within the same benchmark release/model family, not a second corpus or evidence of generality.

## Compute, provenance and limits

Preflight must complete without loading the target model and verify source checksums, sample counts, category/polarity constraints, map integrity, hashes, token boundaries, candidate count and prompt lengths. The runner writes resumable private JSONL and a manifest under ignored `.context/`; the public bundle contains aggregates, hashes and this protocol only. The laptop result in 062 was hypothesis-generating and its coordinate audit post hoc. Restaurant aspect categories and valence labels are public benchmark annotations; exposure to pretraining is possible. Donor-category differences also change words and meaning. No broad causal or model-general claim follows.

- Parent sample/model baseline: [Experiment 051](../../results/restaurant-domain-decoder-transfer-v1/README.md).
- Follow-up signal: [Experiment 062](../../results/output-key-order-topic-match-v1/README.md).
- Benchmark: [DimABSA paper and release](https://aclanthology.org/2026.acl-long.1881/).
