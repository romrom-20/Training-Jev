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

## Literature update after Experiment 066 (2026-09-29)

The initial candidate above needs a tighter claim boundary. Hryhoryeva et al.'s
[UKP_Psycontrol SemEval-2026 system](https://arxiv.org/abs/2604.21534) directly
compares predicting numeric valence and arousal jointly in one prompt versus
separately in two prompts, alongside a label-to-grid condition. That makes a
generic joint-versus-separate performance comparison established prior work.
Their longitudinal essay task and large hosted models differ from our aspect
reviews and local Qwen, but the basic format contrast is not a novelty claim.

Prompt-language sensitivity is also established in adjacent tasks. Behzad,
Zeldes, and Schneider's [Findings of EMNLP 2024 study](https://aclanthology.org/2024.findings-emnlp.916/)
reports that changing prompt language changes multilingual grammaticality-QA
performance. Nguyen et al.'s [August 2026 preprint](https://arxiv.org/abs/2608.26186)
finds prompt-language effects on generated length and lexical realization across
English/Norwegian prompt-response conditions. Experiment 066's wide interval
does not contradict those results; it found no resolved effect for this specific
VA order-moderation endpoint on this sample and model.

A narrower follow-up can ask a mechanistic question: when a causal assistant
output prefix fixes the first VA coordinate to a low versus high value, how much
does the model's continuation change its second coordinate for the same review?
This is related to, but distinct from, generic field-order tests and joint versus
separate scoring. Recent work on [commitment order in diffusion LMs](https://arxiv.org/abs/2608.05687)
studies when a final answer is emitted relative to reasoning in a non-autoregressive
architecture; it does not answer this local autoregressive numeric-coordinate
intervention. This is only an initial literature scan: do not claim priority or
novelty without a broader search.
