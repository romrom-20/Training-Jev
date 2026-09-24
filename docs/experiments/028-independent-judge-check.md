# 028 — independent-family judge check for open answers

## Question

Experiment 027 found large, model-dependent gaps between open-prompt candidate-pair
accuracy and a Qwen2.5 judge's reading of generated answers. Because source-review
accuracy does not establish answer-judging accuracy, test whether a second model
family reaches the same labels.

## Frozen judge and source gate

- Judge: `microsoft/Phi-3-mini-4k-instruct`, revision
  `f39ac1d28e925b323eae81227eaba4464caced4e`, already cached locally.
- Use the same 233 SemEval source-review/aspect items and prompts as experiment 027.
- Run greedy generation, at most 4 new tokens, on the local MPS device in float16.
- Continue only if exact-label accuracy is at least 90% and negative recall at least
  80%. Count malformed labels as incorrect and retain all items.

## Conditional answer-label replication

Only if the source gate passes, regenerate the experiment 025 open-question outputs
for Granite 3.1 2B, Qwen2.5-1.5B and SmolLM2-1.7B, at their frozen revisions, with
8 greedy new tokens. The Phi-3 judge sees only the generated answer and aspect name;
it does not see the review text, the gold label or the Qwen judge's prediction.
Persist only item IDs, aspect, gold, Phi-3 polarity, the already frozen candidate
margin/prediction, and the Qwen 027 judge label. Do not store text.

Report source gate metrics, per-target Phi-3 accuracy versus review gold, and
sentence-clustered Qwen/Phi agreement with a fixed 10,000-draw bootstrap (seed
20261028). Agreement is descriptive: neither judge has been validated on human-
labeled generated answers. Disagreement leaves the result unresolved; agreement is
not a substitute for human validation. No LessWrong post decision follows from this
experiment.

Use only the local MacBook Air; no remote or paid inference.
