# Experiment 041: Does aspect sentiment survive annotated-opinion masking?

**Status:** preregistered before any Qwen judgments.  
**Question:** In semantically aligned Russian, Ukrainian, and Tatar review sentences, does the sentence context still improve continuous aspect-level valence–arousal prediction after every annotated opinion phrase is hidden?

## Why this test

DimABSA makes continuous valence and arousal judgments available alongside aspect and opinion spans. Its standard regression input is text plus aspect, and its paper reports prompted and fine-tuned model benchmarks. Earlier aspect-based sentiment work has long studied how a model links an aspect to opinion expressions, including selection of an aspect-specific opinion snippet. Those literatures establish that opinion terms matter; they do not answer how much target-specific affect remains inferable from the rest of a sentence when all *annotated* opinion terms are removed, especially on matched multilingual versions.

During source audit, the Russian, Ukrainian, and Tatar test files were found to share 630 IDs, equal ordered triplet counts, and identical ordered VA annotations for every shared ID. The sentence texts differ for 629 of 630 IDs. We treat these as aligned versions based on IDs and annotation identity; the repository does not explicitly certify that every pair is a translation. A review-cluster bootstrap keeps the three versions of an ID together.

The novelty claim is deliberately narrow: a targeted search found no direct test of all-annotated-opinion masking on these matched versions with continuous target VA output. Opinion-word ablation and multilingual ABSA are established directions; this is not a claim that sentiment masking or cross-lingual ABSA is new in general.

## Data and sample

- Source: official DimABSA repository, pinned at commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`.
- Files: Russian, Ukrainian, and Tatar restaurant `subtask_2` test sets. SHA-256: `rus=912013b49db2bd387076f63ee9df016350180fbac621449121a27dc262d5459b`, `ukr=05fdf7e2dca235261060b785f969315385f962021f171abb85e45638bbadb036`, `tat=c4f1fb5c21f8f06f598c87e489e7adce1a953d4446ad14a60432ae2a13b85ce6`.
- The public gold test labels have been released. This is an evaluation of a frozen model, not a blind test of whether the model has seen the underlying review text.
- Keep source sentences, stimuli, and per-item model outputs in ignored `.context/`; the dataset repository has no root license file, so this repository will publish aggregate results and source provenance only, not review text or item-level derivatives.
- Include a sentence ID only if all three language versions have the same ordered triplet count and exact ordered VA labels; the chosen target triplet has a non-NULL aspect and opinion; its aspect and opinion strings each occur exactly once; every non-NULL annotated opinion occurs exactly once; and no annotated opinion span overlaps the target aspect span.
- Choose at most one target triplet per shared sentence ID: the first eligible triplet in source order.
- Stratify by target valence: 55 cases below 4.5, 10 cases from 4.5 through 5.5 inclusive, and 55 cases above 5.5. Within each stratum, sort by SHA-256 of `exp041|20260941|<ID>|<triplet-index>` and take the first quota. Selection uses labels and spans only, before inference.
- Expected sample: 120 shared sentence IDs, 360 language-specific target instances.

## Conditions

Use one Qwen2.5-3B-Instruct checkpoint, greedy decoding, and one fixed English instruction template for all languages. The named aspect remains in its native language. Each prompt has the same fields; only the available evidence changes.

1. `aspect_only`: no review sentence and no opinion phrase; the target aspect is shown.
2. `aspect_opinion`: no review sentence; the target aspect and its annotated opinion phrase are shown.
3. `full_text`: the complete source sentence and target aspect are shown.
4. `opinion_masked`: the complete sentence is shown with every distinct, non-NULL annotated opinion span replaced by `[MASKED]`; the target aspect must remain intact.

Return one JSON object with valence and arousal estimates from 1 to 9. Do not provide examples, demonstrations, retries, translation, or manual correction. Each sampled item receives all four conditions in each of the three languages: 1,440 outputs total.

## Outcomes and analysis

- **Primary outcome:** pooled two-dimensional RMSE, `sqrt(mean((Vhat - V)^2, (Ahat - A)^2))`, over the three language versions of each sampled ID.
- **Primary contrast:** `RMSE(aspect_only) - RMSE(opinion_masked)`. A positive value means the non-opinion sentence residue improves VA estimates over the aspect prior.
- Resample the 120 shared sentence IDs with replacement 10,000 times, keeping language versions and VA dimensions together. Report the percentile 95% interval. No item is treated as an independent translation replicate.
- **Registered evidence threshold:** call residual context signal supported only if the primary estimate is at least 0.25 points and its 95% interval is strictly above zero. If this gate fails, report the estimate and interval without upgrading it to a positive claim.
- Secondary, descriptive analyses: full-text versus opinion-masked RMSE; aspect-plus-opinion versus aspect-only; separate valence and arousal RMSE; and language-specific masking penalties. Report intervals, but do not treat these secondary comparisons as confirmatory.
- Invalid or unparsable model outputs count as missing for that condition and are reported. Do not retry or selectively repair outputs. If more than 2% of outputs are invalid, stop interpretation and report a protocol execution failure.

## Limits

The three language versions share labels and IDs and may share source semantics, so this is one aligned set, not three independent replications. Annotated spans do not cover every possible implicit or unannotated evaluative cue; `[MASKED]` therefore means annotated-opinion masking, not sentiment-free language. English instructions can introduce unequal instruction comprehension. A single small model cannot establish a general property of multilingual LLMs. Public test labels, source-review pretraining, annotation noise, and translation quality limit generalization. Results describe this model on this released sample only.

## Provenance and code

- Dataset: [DimABSA official repository](https://github.com/DimABSA/DimABSA2026), pinned as above.
- Dataset paper: [Lee et al., ACL 2026](https://aclanthology.org/2026.acl-long.1881/).
- Related opinion-span work: [Hu et al., CoNLL 2019](https://aclanthology.org/K19-1091/).
- Runner: `scripts/run_opinion_mask_crosslingual_dimabsa_041.py`.
- Analysis: `scripts/analyze_opinion_mask_crosslingual_dimabsa_041.py`.
- Raw source sentences and predictions remain local in `.context/dimabsa/` and `.context/exp041-private-predictions.jsonl`.
