# 032 — Do independent judges also change their read of longer answers?

## Status

Post-result robustness follow-up to experiment 031, frozen before the new judge
outputs are collected. It is exploratory, not an independent confirmation: the
prompt set and answer texts overlap exactly with 031.

## Question

Experiment 031 found that Laya labeled many 8-token answers mixed/unclear and fewer
of their 32-token continuations that way. Do two independent local judge families
(Qwen2.5-3B and Phi-3 Mini) also change their polarity judgments on those same
paired answer texts, and does any change move toward the review-level sentiment
label?

Prior literature has reported verbosity/length bias in LLM preference judges (e.g.
[MT-Bench](https://arxiv.org/abs/2306.05685) and
[length-bias decomposition](https://arxiv.org/abs/2407.01085)). This follow-up
tests whether the 031 cap intervention also affects forced binary aspect-polarity
readouts. It makes no novelty claim and does not measure preference between
answers.

## Frozen design

- Reuse all 233 exact prompts and the paired 8/32 generated texts from experiment
  031 for each of its three target checkpoints. Verify the private-generation file
  against the 031 run manifest before inference. Do not regenerate or edit answers.
- Load Qwen2.5-3B at the revision frozen in experiments 027–028 and Phi-3 Mini at
  the revision frozen in experiment 028. Run one judge at a time on local MPS. Give
  each judge the same fixed wrapper used in experiment 027, changing only the
  embedded generated answer. Greedy decode, maximum 4 output tokens. Parse only an
  exact `positive` or `negative` answer; retain unparseable as null.
- Do not show judge the target-model identity, review-level gold, Laya outputs, or
  the other judge's response. Same item and judge wrapper are used at both answer
  lengths.
- Primary descriptive outcomes: for each judge and target, strict accuracy change
  against review-level gold (unparseable counts incorrect), paired 32-minus-8
  label-flip direction, and sentence-cluster bootstrap 95% interval for paired
  accuracy difference (10,000 draws; seed 20260932). Report coverage/parseability.
  Report judge agreement at each length and the exact 8-token label reproduction
  rate against experiments 027/028 as a protocol check.
- All model answers at 8 tokens hit the cap in 031. The source review label remains
  an imperfect proxy for what the generated answer says; an apparent accuracy gain
  is not proof of correctness. Do not use a p-value threshold or make a confirmatory
  claim.

## Privacy and interpretation

Raw review/answer text stays only in ignored `.context/`. Tracked artifacts may
contain IDs, judge labels, metrics, and model provenance, never answer text. Any
cross-judge effect would establish evaluator sensitivity to answer length under
this protocol, not human-grounded semantic accuracy. A null or inconsistent result
would bound how general the Laya pattern is. No LessWrong decision follows from
this follow-up alone.
