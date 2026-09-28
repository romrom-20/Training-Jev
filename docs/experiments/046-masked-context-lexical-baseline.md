# Experiment 046 — Can sparse lexical learning explain masked-context VA gains?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 045. Commit this protocol, runner, analyzer, and tests before computing predictions on the held-out targets. This selection is agenda triage only; its probabilities are not calibrated.

## Question and literature boundary

Experiments 043–045 found that Qwen2.5-3B's opinion-masked context substantially beat an aspect-only prompt on 217 held-out DimABSA IDs, while a same-token shuffle slightly improved 3B RMSE and had almost no effect for Qwen2.5-1.5B. Can ordinary sparse lexical learning, trained only on the official training split, reproduce any of the aspect-only-to-masked-context gain without pretrained language representations?

DimABSA already defines aspect-level continuous valence–arousal prediction and reports prompted and fine-tuned systems. Word-order robustness and bag-of-words reliance have also been studied in sentiment and broader language-understanding benchmarks. This experiment makes no novelty claim for TF-IDF regression or the general idea that lexical cues predict sentiment. It tests a narrower explanation for this project's post-opinion-mask result on its frozen multilingual held-out slice. See [the DimABSA dataset paper](https://aclanthology.org/2026.acl-long.1881/), [Sinha et al. (2021)](https://aclanthology.org/2021.acl-short.27/), and [Phan and Ogunbona (2020)](https://aclanthology.org/2020.acl-main.293/).

## Data and split

- Use Russian, Ukrainian, and Tatar Track A restaurant Subtask 2 training files and test files at repository revision `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- Training SHA-256: Russian `a81c2e737a0ed8e3fcfee48864403a3339ab388b7ed545436f45a11ae878b92a`; Ukrainian `6c1e4843fb9af233a1103c751b56b99596f724ef9309ff6156f4041e4560bff1`; Tatar `1f374e08dfd3fac61d9ca6935e185958f889a24a6690dacec6ff6497d4d54917`. Test files retain the frozen Experiment 041 hashes.
- Recreate exactly the 217 IDs and target indices selected by Experiment 043. Verify the 043 parent output hash and its gold VA pairs. Exclude any training row whose review ID or exact text overlaps a held-out test item; record exclusions.
- Each training aspect/opinion/VA annotation is one example. For the masked-context condition, replace all annotated opinion spans in its sentence with `[MASKED]`, merging overlapping spans and masking every occurrence of an annotated phrase. The same masking rule applies to every target example from a sentence. The aspect-only condition contains only the target aspect string.
- Train one model per language and condition; do not pool languages or use pretrained embeddings. Keep all text, IDs, and per-item predictions in ignored `.context/`.

## Model selection and outcomes

- Use a word unigram/bigram TF-IDF plus character-within-word 2–5-gram TF-IDF feature union, followed by multi-output ridge regression for valence and arousal.
- Select ridge alpha independently for each language and condition from `[0.01, 0.1, 1, 10, 100]` by five-fold `GroupKFold`, grouped on the review ID (the ID prefix before `:`). Select only by training-fold VA RMSE, then refit on all eligible training examples. All vectorizers are fit inside each fold.
- **Primary diagnostic contrast:** `RMSE(aspect_only) - RMSE(masked_context)` on the fixed 217 source-ID clusters, preserving all three languages and both dimensions. Positive values mean masked context helps the sparse model. Use 10,000 source-ID cluster bootstrap draws with seed `20260946`.
- Report the estimate and percentile 95% interval, overall and per language. A reproducible lexical-context gain in this sample requires an estimate of at least 0.25 VA points and a lower interval bound above zero. Compare its magnitude with Experiment 043's Qwen gain descriptively; no equivalence claim is made.
- There is no generation/parser failure gate: predictions are finite ridge outputs. If data hashes, split-disjointness, frozen IDs, or pair completeness fail, stop before scoring.

## Limits

This is a benchmark diagnostic, not a new model or an independent corpus replication. It uses one public restaurant-review release, one selected slice, and opinion spans whose annotation may be incomplete. TF-IDF can exploit residual aspect words, topic, punctuation, or annotation regularities; a positive gain would show predictive lexical information, not establish causal sufficiency. A negative result would not establish that only pretrained models use context. The result is conditional on this feature map, per-language training set, and validation procedure.

## Provenance

- Prior aggregate results: `results/expanded-opinion-mask-repair-v1/README.md`, `results/opinion-mask-word-order-v1/README.md`, and `results/opinion-mask-order-model-size-v1/README.md`.
- Private Laya agenda trace: `.context/laya-research-triage-046.json` (not committed).
- Runner/analyzer: `scripts/run_masked_context_lexical_baseline_046.py`, `scripts/analyze_masked_context_lexical_baseline_046.py`.
