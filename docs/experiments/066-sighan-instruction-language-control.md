# Experiment 066 — Does instruction language change the SIGHAN order interaction?

**Status:** Laya-selected after Experiments 064–065; preregistered before new
Qwen2.5-3B inference. Laya's probabilities are uncalibrated agenda triage only.

## Question

In SIGHAN Chinese reviews, does translating the scoring instructions from English
to Chinese change the aggregate interaction among output-field order,
same-/cross-category context, and finite-grid/free-greedy decoding?

Experiment 064 did not replicate the negative order moderation seen in two English
DimABSA splits using the same English instruction on Chinese input. Experiment 065
found no detectable model-size difference on that Chinese cohort. A post-hoc
three-cohort bootstrap found an SIGHAN-minus-weighted-English difference of +0.311
(95% interval [+0.003, +0.610]), but this interval barely excludes zero and
language, domain, release and cohort are confounded. This experiment tests one
plausible confound directly by changing only the language of the instruction on
the same selected Chinese reviews and exact donor maps.

General output-order sensitivity is already studied, including reason-before-score
scoring and schema field design ([literature boundary](064-literature-boundary-audit.md)).
This is a narrow instruction-language moderation test, not a new generic order
effect claim.

## Cohort and fixed donor maps

- Reuse a deterministic subset of 88 recipients from Experiment 064's 180-case
  SIGHAN cohort. Rank IDs within the stated polarity/category cells by
  SHA-256 of `exp066-sighan-zh-instruction-v1|case_id` and take the fixed quotas:
  - Negative: `食物#品质` 22, `食物#份量与款式` 11, `食物#价格` 7,
    `饮料#品质` 4.
  - Positive: `食物#品质` 22, `食物#份量与款式` 22.
  This yields 44 negative and 44 positive recipients in six non-empty strata.
- Reuse Experiment 064's three same-category and three cross-category donor maps
  exactly for those recipients, even when a donor lies outside the 88-person
  recipient subset. Each map remains polarity-matched and one-to-one; same versus
  cross category is verified from Experiment 064's saved donor metadata. The
  subset maps are serialized as sorted-key JSON and SHA-256 pinned here:
  - Same-category permutations 1–3:
    `b98728e19e3bb99f3f32f1c5a9d288b2820fb0f89202a8810a009fcf5b7a7d38`,
    `ec4210e0429c2a7213c953a7f534609454edb6d5ec3d7223d39c13ce11f0c401`,
    `80a79c567e72c307bcac221291be06f0cd3bd6a695cb2b4c3e730561b874cc75`.
  - Cross-category permutations 1–3:
    `7cb29764333dc08e07f29675f542d5f36fa5e20c94b4f82da78ba6fe00ca408a`,
    `9ce3da37ae573e3d1256bdf4882c58314d7c0dceb6a6f989b3787b600120e830`,
    `a69470883da6fb38a00e3e023bb2fa41f613b97607aed320cef5264c4ffc857b`.
- Inputs and gold labels must match the pinned SIGHAN 2024 Simplified source used
  in Experiments 052 and 064. The English-condition outputs come from the frozen
  Experiment 064 artifact and Experiment 052 valence-first own-review baseline.

## Intervention and generation

Keep the review text, target aspect, opinion masking, numeric interpretation,
candidate grid, output key order, decoder and decoding settings fixed. Replace only
the natural-language prompt text with a direct Chinese translation. Keep the JSON
keys themselves in English because they are part of the output contract.

The Chinese instruction template is:

> 评估作者对指定方面表达的情感倾向。效价（valence）范围从 1（极度负面）到 9（极度正面）。唤醒度（arousal）范围从 1（非常平静）到 9（非常激动或强烈）。仅使用所提供的证据。只返回一个 JSON 对象，其中包含数值键“valence”和“arousal”，两项均为 1.0 至 9.0，以 0.1 递增，且恰好保留一位小数。\n评论文本：{review}\n目标方面：{aspect}\n与目标相关的观点短语：[NOT PROVIDED]

For arousal-first serialization, reverse the two keys in the JSON instruction only,
matching Experiment 064. Review and aspect strings remain verbatim from the source;
the masked opinion text uses the same `[MASKED]` token as before. Preserve the
English `Valence` then `Arousal` definition order across both output orders, just
as the existing English prompt does.

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy MPS inference on the 24-GB
  MacBook Air; no cloud inference.
- New Chinese-instruction generations: both JSON orders and both decoders for own
  masked-review baselines (88 × 2 × 2 = 352), plus both orders, both decoders, both
  donor conditions and all three fixed maps (88 × 2 × 2 × 2 × 3 = 2,112), for
  **2,464 new outputs** total.
- Free greedy has a 40-token cap and strict parsing with no retries or repairs.
  Save resumable private JSONL and a manifest under ignored `.context/`.
- Before loading the target model, verify protocol/source/parent hashes, the exact
  88-ID selection, category/polarity strata, six donor-map extracts, map identity,
  output-cell counts, candidate token uniqueness and prompt token bounds.

## Registered analysis

For each instruction language and JSON order, calculate the same-minus-cross
difference in the finite-grid-minus-free-greedy context-gain interaction. Context
gain is `RMSE(own masked review) - RMSE(donor masked review)`, with RMSE aggregated
over both VA coordinates.

The **primary endpoint** is Chinese-instruction order moderation minus
English-instruction order moderation, where order moderation is the arousal-first
value minus valence-first value of that same-minus-cross interaction. Recompute the
English estimate on the exact 88 recipients and fixed maps from Experiment 064;
do not compare against the full-cohort estimate. Bootstrap recipient IDs with
replacement, retaining all cells and both languages together. Use 10,000 draws,
seed `20260966`, percentile 95% interval.

If either instruction language's donor-context free-greedy invalid-output rate
exceeds 2%, withhold the primary score contrast. Otherwise use IDs with complete
cells in both languages and report invalid counts and coverage. A primary interval
excluding zero would be evidence that prompt language moderates the registered
interaction in this selected cohort and model. It would not show that prompt
language caused the earlier English/Chinese cohort difference, because dataset,
language, domain and release still differ together outside this within-SIGHAN test.

## Compute and provenance

The expected job count is 2,464. The English comparator is bound to Experiment
064's prediction SHA-256
`48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7` and
Experiment 052's baseline SHA-256
`76671a3330e35fddfd915f15c86b49b90f8113af74041651162b7a81051765fe`.
SIGHAN input and gold hashes are
`5053608647d06ad88ce21a2aecbef0e29dc58a8220f921561d39d4e9cf86fe28` and
`6ecc6949d5f77ea34a09826115a0f91327313ae33824a5c373d7322bdbb17108`.
Publish aggregate values and hashes only. This run is within one model family and
one Chinese review release, and public-test pretraining exposure cannot be ruled
out.

- Local Laya choice trace: `.context/laya-research-triage-066.json` (ignored).
- Prior results: [064](../../results/sighan-output-key-order-replication-v1/README.md),
  [065](../../results/sighan-order-model-size-replication-v1/README.md).
