# Experiment 064 — Independent-release test of output order × category match

**Status:** Laya-selected, preregistered cross-release follow-up before new target-model inference. Laya's probabilities are uncalibrated agenda triage only.

## Motivation and literature boundary

Experiments 062–063 varied only the order of numeric JSON fields while crossing decoder with same- versus cross-category review context. The 063 coordinate-specific endpoint was inconclusive (-0.204; 95% interval [-0.487, +0.067]); the aggregate all-VA order moderation was -0.528 ([-0.808, -0.254]). A post-hoc laptop/restaurant comparison found estimates of -0.258 and -0.528 and an uncertain difference (-0.271, [-0.613, +0.068]). Both samples were from DimABSA 2026 and Qwen2.5-3B.

SIGHAN 2024 is a separately curated Chinese restaurant-review release with aspect-level categories and continuous VA annotations ([Lee et al., 2024](https://aclanthology.org/2024.sighan-1.19/); [official data repository](https://github.com/NYCU-NLP/SIGHAN2024-dimABSA)). It has already been used for a decoder-by-evidence comparison in Experiment 052, but not for the same/cross-category donor × output-order contrast. General prompt/order sensitivity is known ([Guan et al., 2025](https://arxiv.org/abs/2502.04134)); this test asks whether the narrower aggregate context-match interaction transfers to another release/language under the same small model.

## Source, sample and donor maps

- Use SIGHAN 2024 Simplified Chinese Task 2+3 test input and gold files at repository commit `d43a482039a41fbc26087ef5b26e94ee772d0fb6`; SHA-256 values are `5053608647d06ad88ce21a2aecbef0e29dc58a8220f921561d39d4e9cf86fe28` and `6ecc6949d5f77ea34a09826115a0f91327313ae33824a5c373d7322bdbb17108`. Apply the frozen Experiment 052 parser and first-eligible-target selection, yielding 1,916 eligible IDs.
- Exclude neutral valence. Choose 90 negative and 90 positive IDs using within-cell SHA-256 ranking `exp064-sighan-v1|case_id`, with these fixed category quotas:
  - Negative: `食物#品质` 45; `食物#份量与款式` 21; `食物#价格` 14; `饮料#品质` 7; `餐厅#杂项` 3.
  - Positive: `食物#品质` 45; `食物#份量与款式` 45.
  All selected polarity/category cells have at least two IDs; both polarity groups have at most half their recipients in any one category, allowing one-to-one cross-category derangements.
- For each polarity/category cell, create three same-category, same-polarity donor derangements. For each polarity group, create three cross-category, same-polarity one-to-one donor maps. Use the Qwen2.5-3B tokenizer on all-opinion-masked Chinese text to minimize absolute donor/recipient token-length differences, with deterministic hash tie-breaking. Require a full permutation, no self-donors, category agreement in same-category maps, and category disagreement in cross-category maps. Reuse the same maps under both JSON orders and both decoders.
- Reuse Experiment 052's own-review `opinion_masked` valence-first baseline on the selected IDs. Generate the arousal-first own-review baseline anew. Keep raw text, IDs, donor maps, generations and per-item values only in ignored `.context/` files.

## Conditions and generation

- Context: same official category versus different official category, with polarity held fixed. Cross-category review text also changes lexical and semantic content; it is not a pure category-label intervention.
- Serialization: valence-first `{"valence":v,"arousal":a}` versus arousal-first `{"arousal":a,"valence":v}`. Use the same review, target aspect, English instruction, numeric values, 6,561 one-decimal grid, parser, and greedy generation settings. Canonicalize every parsed result to `[valence, arousal]`.
- For each donor condition, order, map and recipient, generate both finite-grid and free-greedy scores. Generate arousal-first own-review baseline for both decoders: **4,680 new generations** total (4,320 donor-context + 360 own-review).
- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, local greedy MPS inference on the 24-GB MacBook Air. Free greedy has a 40-token cap, strict parsing, no retry or repair.

## Outcomes and analysis

For each order, category condition, decoder, map and VA coordinate, calculate context gain as `RMSE(own masked review) - RMSE(donor masked review)`. Calculate decoder interaction as finite-grid context gain minus free-greedy context gain. The same-minus-cross difference is the topic-match effect. The **primary endpoint** is the aggregate two-coordinate topic-match effect under arousal-first minus its value under valence-first. Negative values follow the direction observed in 062–063.

Bootstrap source sentence IDs with replacement, retaining both orders, donor conditions, decoders, coordinates and all fixed maps. Use 10,000 draws, seed `20260964`, percentile 95% interval. Secondary endpoints are coordinate-specific order moderations, their arousal-minus-valence difference, map-specific estimates, and invalid-output counts. These are descriptive and not separate confirmatory claims.

If more than 2% of donor-context free-greedy outputs are invalid, withhold all score contrasts. Otherwise analyze recipients with complete cells and report coverage. A result on SIGHAN would provide cross-release/cross-language replication for this model, not model-general evidence; SIGHAN 2024 and DimABSA 2026 are both public test releases and pretraining exposure cannot be ruled out.

## Compute and provenance

Run source-hash, exact sample-quota, category-group, token-boundary, donor-map, prompt-length and candidate-grid checks before loading the target model. The runner must write resumable JSONL and a run manifest under ignored `.context/`. Publish aggregates and hashes only. The expected workload is 4,680 generations; Experiment 063's 4,498 outputs took about 126 minutes on the same laptop/model size.

- Parent own-review outputs: [Experiment 052](../../results/sighan-chinese-decoder-transfer-v1/README.md).
- Prior output-order results: [Experiment 062](../../results/output-key-order-topic-match-v1/README.md), [Experiment 063](../../results/output-key-order-restaurant-replication-v1/README.md).
