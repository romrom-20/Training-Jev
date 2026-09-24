# Experiment 025: does answer formatting create score/generation agreement?

## Motivation

Experiments 023–024 found near-perfect or perfect agreement between positive/negative
candidate-pair decisions and greedy generated labels on prompts that explicitly
requested a one-word answer. This paired ablation removes that output constraint to
test whether the agreement is a consequence of forced-choice formatting.

## Frozen protocol

- Use the same 233 SemEval aspect prompts (112 sentences) on Granite 3.1 2B,
  Qwen2.5-1.5B, and SmolLM2-1.7B at pinned revisions.
- Evaluate two within-item prompt formats: (A) the original explicit instruction to
  reply with exactly one word, positive or negative; (B) the same review and aspect
  question with the output-format instruction removed. Greedily generate up to eight
  tokens in both formats. Collect the positive-minus-negative next-token margin.
- Save only margins, exact one-word validity, whether a unique positive/negative word
  appears in the generated answer, that extracted polarity, and correctness. Save no
  review or generated text.
- Primary endpoint: the open-question minus forced-choice difference in the fraction
  of parseable generated labels agreeing with the candidate-pair prediction, paired
  within sentence and model. Use a 10,000-draw sentence-cluster bootstrap, seed
  20261026. Also report extraction coverage; if open-question coverage is below 90%
  for any model, report the pooled contrast descriptively and make no cross-family
  agreement claim.
- Report format compliance, gold accuracy, pair/generation agreement, and model-level
  effects. This is observational and does not test intervention causality.

The open-question output may not contain a polarity word. Such answers are marked
unparseable; no semantic classifier is fitted after seeing outputs.
