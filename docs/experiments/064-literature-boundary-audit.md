# Experiment 064 — Literature-boundary audit

This audit was added after Experiment 064's protocol had been frozen and the
generation run had started. It does not change the sample, prompts, contrasts,
or analysis.

## Closest prior work found

- Chen, Chu, and Nakayama (2024), [“LLM as a Scorer: The Impact of Output Order on
  Dialogue Evaluation”](https://arxiv.org/abs/2406.02863), vary prompt structure,
  including the sequence in which reasons and scores are requested. Their abstract
  reports that reason-first produces more comprehensive dialogue evaluations.
  This establishes that output sequence can change scoring behavior. It does not
  cross the order manipulation with same-/cross-category review context and
  finite-grid versus free generation for valence-arousal regression.
- Lin (2026), [“Your Prompt Is Not the Only Prompt: How Much Do LLMs Weight
  Structured-Output Schema Descriptions?”](https://arxiv.org/abs/2608.08254), tests
  schema-description placement across ten model configurations. The abstract
  reports model-dependent effects, including accuracy changes from adding an
  intermediate reasoning field before a classification label. It studies
  instruction placement and reasoning-field design, rather than reversing the
  order of two continuous numeric VA fields or testing category-match moderation.
- Mittal's [Constrained Sensitivity Lab](https://github.com/Vaibhav701161/constrained-senstivity-lab)
  reports artifact-backed paired results on Qwen2.5-7B/GSM8K where reasoning-first
  versus answer-first JSON output changes accuracy while validity is held fixed.
  This is close empirical overlap for JSON field order affecting semantics, but
  uses math answers and a reasoning field, not affective dimensions, donor context,
  or the present decoder interaction.

## Claim boundary

Experiment 064 cannot support a broad claim that output-field order is a new
phenomenon. Its possible contribution is much narrower: an interaction between
the order of two continuous affective scores, context-category match, and decoding
method, tested across two DimABSA splits and a separately released Chinese SIGHAN
dataset on one local model family. The prior work also makes a positive result
less surprising. A null or inconsistent SIGHAN result would be informative about
the limited scope and transfer of the earlier interaction, rather than a general
claim about JSON or structured outputs.

## Follow-up question if the interaction replicates

The next discriminating test should separate coordinate-order effects from a
general instruction-order effect. On a small fixed cohort, compare joint
two-coordinate generation in both orders with independent valence-only and
arousal-only scoring, using identical review evidence and no donor substitution.
Predefine whether the second-emitted coordinate has larger absolute error or a
signed residual coupled to the first coordinate, and include a reversed-axis
instruction control. This is feasible with local 3B inference and directly tests
whether the first generated score alters the second. It is a candidate, not a
registered follow-up; decide after 064's independent-release outcome.
