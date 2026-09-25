# 031 — Does a short answer cap create apparent mixed sentiment?

## Question and rationale

Experiment 030 found many mixed/unclear Laya labels on 8-token open answers,
especially Qwen2.5-1.5B. A competing explanation is that some answers are simply
right-censored before they finish. Test that explanation with a paired generation
intervention: same target, prompt, greedy decoding, and item; change only
`max_new_tokens` from 8 to 32. Laya classifies each resulting answer with the same
four-way typed-choice question and per-item option mapping.

Prior work documents LLM-judge preference for longer responses, including
[Explaining Length Bias in LLM-Based Preference Evaluations](https://arxiv.org/abs/2407.01085).
That is adjacent evidence, not this intervention: this experiment holds the answer
prefix/prompt fixed, varies a generation cap, and asks whether the cap changes a
separate engine's *semantic category* for the answer. We make no priority or novelty
claim.

## Frozen design

- Run all 233 items in the frozen SemEval subset for all three target checkpoints
  used by experiment 025/030: Granite 3.1 2B, Qwen2.5 1.5B, and SmolLM2 1.7B.
- Generate independently at budgets 8 and 32, greedy (`do_sample=False`) with the
  exact `open_question` prompt template frozen in experiment 025. Keep each target
  model loaded alone on MPS; never store answer text in tracked outputs.
- Record generated token count, whether EOS occurred before the cap, and whether the
  answer was capped. For cap-hit pairs, assert the 8-token output is an exact token
  prefix of the 32-token output. This checks the intervention actually was truncation.
- Classify both texts with `convaiinnovations/laya` revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, package `laya==0.3.20`, local MPS,
  and experiment 030's four-way choice wording. Use the same seeded neutral A-D
  mapping for both budgets on each item. Laya sees only aspect and generated answer.
- Freeze the primary contrast to cap-hit pairs: change in the share classified
  `mixed` or `unclear` at 32 tokens versus 8. Report the paired four-way transition
  matrices and all-category contrasts as secondary descriptive outcomes. Report
  uncapped-at-8 pairs separately as a negative-control stratum.
- Report paired differences overall and by target, with source-sentence cluster
  bootstrap percentile 95% intervals (10,000 draws; seed 20260931). Do not treat
  the three model outputs per sentence as independent. No p-value threshold or
  confirmatory claim is specified; this is a bounded diagnostic experiment.
- Compare the regenerated eight-token Laya labels with experiment 030 on its 60
  matching rows as a reproducibility check. Do not use experiment 030's outputs to
  alter this protocol.

## Interpretation and release

A lower `mixed/unclear` rate after uncensored continuations would show that the
8-token generation cap contributes to Laya's ambiguity judgments on this task; it
would not establish that the 32-token interpretation is correct. No change, or a
change restricted to answers that had already ended, weakens the truncation
explanation. Laya is an automated evaluator with uneven aspect-level performance,
so any effect remains specific to this evaluator until human-coded answer labels
validate it.

Raw generated text and token IDs stay in ignored `.context/length-censoring-031/`.
The tracked result may contain item IDs, token counts, EOS/cap flags, Laya labels and
probabilities, and aggregate analysis only; it must not contain answer/review text.
Use CPU only if MPS is unavailable and record the device. No LessWrong decision
follows from this experiment alone.
