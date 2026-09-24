# 027 — validate a local judge before using it on free-form sentiment answers

## Question

Can a small, cached instruction-tuned model reliably recover aspect sentiment from
human-annotated restaurant reviews and then classify another model's unconstrained
answer? This is an evaluator validation study. It is not a test of activation
steering or a novelty claim.

## Frozen source-label gate

- Use the 233 positive/negative SemEval 2014 restaurant aspect items and the exact
  frozen source XML and item filter used by experiments 023–025.
- Use `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, already cached locally.
- Ask for the sentiment of the named aspect from the original review; require the
  exact one-word answer `positive` or `negative`. Greedy decode, maximum 4 tokens.
- Primary validation is exact-label accuracy; also report negative recall and
  parseability. Continue only if accuracy is at least 90% and negative recall is at
  least 80%. No threshold tuning or item removal is allowed.
- If the source-label gate fails, stop. Do not score Experiment 025 generations with
  this judge.

## Conditional free-answer evaluation

Only if the source-label gate passes, regenerate the open-question condition from
experiment 025 on the same 233 prompts for Granite 3.1 2B, Qwen2.5-1.5B and
SmolLM2-1.7B, at their frozen revisions, greedily with 8 new tokens. Retain answer
strings in process memory only. Ask the frozen judge to classify the generated
answer's polarity toward the named aspect without showing it the original review or
gold label. Save only item ID, judge label/parseability and the experiment 025
candidate-pair prediction/margin. Report judge-label accuracy against the review
gold as a proxy for whether the answer communicated the correct polarity, and
candidate-pair/judge agreement. Make no claim that source-label accuracy proves
judge validity on generated text; that remains an explicit limitation.

No text, prompts or review content are written to the result bundle. Do not call a
LessWrong post warranted from this experiment alone.

## Provenance

This protocol is committed before the Qwen2.5-3B judge call. It uses local cached
weights and the 24 GB MacBook Air; no API or cloud inference is part of the design.
