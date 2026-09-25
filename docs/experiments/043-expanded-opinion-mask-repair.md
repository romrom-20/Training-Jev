# Experiment 043 — Expanded multilingual opinion-mask repair

**Status:** protocol and runner committed before any Experiment 043 target-model judgments.
**Selection:** Laya's pinned local typed-choice model selected a larger, stricter-decoding rerun of the 041 question. Its selection probability is not calibrated evidence of scientific value.

## Question

On a disjoint set of aligned Russian, Ukrainian, and Tatar restaurant-review IDs, does the remaining sentence context improve Qwen2.5-3B's continuous target-level valence–arousal (VA) estimates after every uniquely annotated opinion span has been masked, relative to an aspect-name-only prior?

The question is a focused execution-corrected rerun of Experiment 041's primary contrast. It is not a new claim that opinion masking or multilingual aspect-based sentiment is novel. Experiment 041's overall invalid-output rate exceeded its frozen 2% stop threshold, so none of its VA scores were analyzed. Experiment 042 showed that adding an explicit `insufficient` answer fixed syntax but led to near-universal abstention; this run therefore requires a numeric estimate and constrains only its output form.

## Literature boundary

DimABSA (Lee et al., ACL 2026) benchmarks text-plus-aspect to continuous VA outputs across six languages and reports prompted and fine-tuned baselines. Earlier aspect-based sentiment work studies linking target aspects to their opinion spans. A recent causal metacognition study tests confidence-guided abstention on factual multiple-choice questions. These establish the neighboring literatures, but do not by themselves answer whether non-opinion sentence residue improves continuous VA estimation on these matched language files. This is a narrowly scoped search result, not an exhaustive novelty claim.

- [DimABSA dataset paper](https://aclanthology.org/2026.acl-long.1881/)
- [Hu et al., target-specific opinion phrase selection](https://aclanthology.org/K19-1091/)
- [Kumaran et al., confidence-driven abstention](https://www.nature.com/articles/s42256-026-01293-x)

## Data and sample

- Use the official DimABSA Russian, Ukrainian, and Tatar restaurant test files pinned in Experiment 041 at commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`; verify the exact three SHA-256 hashes in the 041 runner.
- Apply the same aligned-ID, ordered-annotation, target-span, and opinion-span eligibility rules as Experiment 041. Include at most one target per shared ID.
- Exclude all 120 IDs selected for Experiment 041. The source audit found 574 eligible aligned IDs (158 low, 21 near-neutral, 395 high). After that exclusion, only 103 low and 11 near-neutral IDs remain. Select all 103 low-valence IDs and all 11 near-neutral IDs, plus 103 high-valence IDs by SHA-256 order using seed `20260943`. This yields 217 balanced non-neutral clusters and uses every remaining near-neutral cluster.
- The 217 IDs are disjoint from Experiment 041's selected IDs but remain within the same public test release. This is a held-out-item rerun, not an independent dataset replication or a blind test.
- Keep all source text, aspect strings, prompts, and item outputs in ignored `.context/`. The upstream repository has no root license file; publish only aggregates and source provenance.

## Conditions and model

For each selected ID and language, ask for VA under exactly two evidence conditions, with the same prompt except for evidence fields:

1. `aspect_only`: show the target aspect, and no review text or opinion phrase.
2. `opinion_masked`: show the full review with every distinct, non-null annotated opinion span replaced by `[MASKED]`, and preserve the target aspect.

Run the frozen `Qwen/Qwen2.5-3B-Instruct` revision from the task-ladder config, greedy, locally on MPS. Use a finite grammar of every JSON pair with valence and arousal from 1.0 to 9.0 in 0.1 increments (6,561 candidates). There is no abstention candidate. Both conditions use the same candidates and decoder. Candidate outputs must be token-boundary verified before inference. A response that does not parse to the exact registered schema is invalid; do not retry it. An invalid rate above 2% blocks score analysis as in Experiment 041.

The grammar discretizes predictions to tenths; it can guarantee response syntax, not estimate quality. Prompt constraints may change answer content. The paired contrast is therefore interpreted as an effect under this fixed constrained-decoding protocol.

## Outcomes and analysis

- **Primary:** pooled two-dimensional RMSE difference `RMSE(aspect_only) - RMSE(opinion_masked)` over all three language versions of complete selected IDs. Positive values favor residual sentence context.
- Resample the 217 shared IDs with replacement 10,000 times, preserving all three language versions and both VA dimensions within a sampled cluster. Use bootstrap seed `20260943`.
- The registered support gate is an estimate of at least 0.25 VA points and a percentile 95% interval strictly above zero. If the invalid rate exceeds 2%, do not calculate VA contrasts. If it is at or below 2%, calculate the score contrast on complete aligned IDs only, and report how many clusters remain.
- Descriptive only: valence and arousal contrasts separately and the primary contrast within each language. These are not additional confirmatory tests.
- Report completion, invalid count/rate, output grid, model/data/protocol hashes, MPS runtime, and gate outcome. No item-level predictions, IDs, review text, or aspects are published.

## Limits

The three files share IDs and annotation labels; they are aligned versions, not independent language replications, and translation status is not certified by the source repository. Annotated-opinion masking leaves implicit and unannotated cues. The released test labels may be present in pretraining data. One model, one restaurant domain, one public test release, and constrained one-decimal output do not support a general claim about multilingual VA reasoning or confidence. This experiment does not evaluate abstention or internal confidence. LessWrong remains deferred until completed results and a worthwhile empirical finding exist.

## Provenance

- Source commit, paths, and hashes: `scripts/run_opinion_mask_crosslingual_dimabsa_041.py`.
- Parent execution failure and parser amendment: `results/opinion-mask-crosslingual-dimabsa-v1/README.md` and `docs/experiments/041-analysis-amendment.md`.
- Adaptive technical audit: `docs/experiments/042-structured-output-feasibility.md` and `results/structured-output-feasibility-v1/README.md`.
- Runner: `scripts/run_expanded_opinion_mask_repair_043.py`.
- Analyzer: `scripts/analyze_expanded_opinion_mask_repair_043.py`.
