# 030 — local Laya decision-engine audit of generated polarity

## Question

Do the open-answer labels from experiments 027–028 change when a small, open-
weights typed-decision model replaces autoregressive LLM judges? This is a
cross-engine robustness check. Laya outputs are not human labels and do not resolve
whether any judge is correct on generated text.

## Frozen engine and prompt

- Use `convaiinnovations/laya`, model revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, through local `laya` package
  version `0.3.20`; this is the Apache-2.0 English base checkpoint
  ([model card](https://huggingface.co/convaiinnovations/laya),
  [package](https://pypi.org/project/laya/0.3.20/)). No hosted Jev API, account,
  or remote inference is used.
- Use Laya's typed `choice` decision. Do not use its `noul` primitive: the model
  card warns that yes/no outputs can follow the true/false option wording. Each
  question has four choices with neutral keys, and a seeded per-item permutation
  of keys: communicates positive polarity, communicates negative polarity,
  communicates both, or unclear/no aspect sentiment. Descriptions are fixed in
  the runner. A shared permutation is used for the source review and all three
  generated answers for an item.
- Input only the aspect name and text being evaluated. For the source screen, text
  is the review; for generated-answer evaluation, text is only the model answer.
  Do not provide source gold, target model identity, candidate score, or Qwen/Phi
  predictions to Laya.
- Score all 233 frozen SemEval items as an unfiltered source-review capability
  anchor, and all 60 already generated answers from experiment 029. Run every
  phase regardless of source-screen accuracy; report the screen as context, not
  as a pass/fail gate selected after seeing results.

## Outcomes

For the source screen, report four-way output counts, clear positive/negative
coverage, accuracy and negative recall against the source labels. For generated
answers, report the four-way output distribution and per-target coverage, binary
polarity-label accuracy against review gold, Laya/Qwen/Phi agreement, and
candidate-pair agreement. Review gold is only a benchmark proxy for the generated
answer's meaning; it is not a valid answer-level ground truth. Report Laya's
per-choice probabilities descriptively, without calling them calibrated on this
domain. Bootstrap generated-answer proportions and binary accuracies by the 20
sampled source sentence IDs (10,000 draws, seed `20261030`); the three target
answers stay together in each draw. Treat intervals as descriptive because the
sample is small and stratified.

## Privacy and compute

Run entirely on the MacBook Air with local open weights. Keep raw answer text only
in ignored `.context/`. Publish IDs, labels, probabilities, and model provenance;
never publish review or generated-answer text. A mismatch with Qwen/Phi is a
reason to inspect the task definition, not evidence that Laya is right. No
LessWrong decision follows from this experiment alone.

## Prior limitations checked before the run

The model card reports that base Laya is weak on an unrelated typed-decisions
benchmark and cautions against trusting raw confidence without domain calibration.
Those limitations motivate the full 233-item source screen, neutral keys, the
explicitly limited interpretation above, and preservation of all predictions.
