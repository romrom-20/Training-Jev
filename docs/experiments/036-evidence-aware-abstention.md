# Experiment 036 — Can a judge abstain when a prefix lacks evidence?

## Preregistration

This is a post-result follow-up to experiment 035. Freeze this protocol, runner,
analyzer, and tests before running the changed judge prompts. It reuses only exact
direct good/bad stimuli from the frozen 035 set; the ambiguous positive “not bad”
condition is excluded, and the forced-binary negative “not good” condition is
excluded to keep the matched gold set balanced and construction-simple.

## Question and hypothesis

When the visible prefix ends before the polarity clause, does explicitly offering
`insufficient` improve the correctness of an evaluator's *decision policy* while
preserving accurate polarity labels when the clause is visible? Experiment 035
showed that Qwen often guessed negative and Phi often failed the one-word format
on identical no-cue prefixes. It also showed that early visible positive/negative
clauses were easy for both judges. This experiment tests an abstention option as
a direct remedy, not the general effect of truncation.

This is an evaluator-protocol experiment over synthetic short text, not a natural
answer-quality study. Human validation of generated answers remains necessary.

## Frozen sample and gold

From `docs/experiments/035-cue-position-stimuli.json`, select exactly rows where
`form == direct`. This yields 192 stimuli: 96 positive, 96 negative, and 48
frame × aspect scaffold clusters. Reuse the exact eight- and twelve-word visible
prefixes and the same frame/aspect labels. Do not alter text or sample.

Expected decision is deterministic by visible evidence:

- late position, 8 words: `insufficient` (the visible text contains only the
  seven-word neutral frame and the first word of the hidden sentiment clause);
- early position, 8 words: the known positive/negative clause is visible, so the
  expected decision is its polarity;
- either position, 12 words: the full clause is visible, so the expected
  decision is its polarity.

This makes 96 insufficient-evidence cases and 288 cue-visible cases per judge.
No answer text, sentiment gold, condition name, or expected decision is shown to
the judges.

## Procedure

Run local MPS with the exact Laya, Qwen2.5-3B and Phi-3 Mini revisions from
experiment 035. Qwen/Phi receive the same prompt except for the allowed response
set: `positive`, `negative`, or `insufficient`. The prompt explicitly says to
choose `insufficient` if the visible excerpt contains no evidence about sentiment
toward the named aspect. Use greedy decoding, maximum four new tokens, and one
model loaded at a time.

Run Laya's existing four-way `positive`/`negative`/`mixed`/`unclear` evaluator
again on the same prefixes. Use a deterministic key-to-label map keyed by
`aspect × exact visible text`, so identical judge inputs always get the same map.
For the predeclared appropriateness score, map Laya `unclear` to `insufficient`;
`mixed` remains a substantive decision and does not count as abstention.

As the matched baseline, use the exact Qwen/Phi binary labels already collected
for these same direct stimulus IDs in experiment 035. Reuse their results; do not
rerun the forced-binary wrappers.

## Endpoints and decision rule

**Primary endpoint:** paired change in appropriate-decision accuracy from the
035 forced-binary wrapper to the new abstention-enabled wrapper, pooled equally
over Qwen/Phi and the exact same stimulus × prefix rows. Count a row correct only
when the judge returns the deterministic expected decision above. A paired
scaffold-cluster bootstrap resamples the 48 frame × aspect clusters, retains all
labels, positions, prefixes and judges within a sampled cluster, and uses 10,000
draws with seed `20260936`.

Report the same paired change by judge, abstention recall on late-8 no-cue rows,
false-abstention rate on cue-visible rows, polarity accuracy and coverage on
cue-visible rows, and Laya's four-way outcomes. Parse failures count incorrect
for binary wrappers and are reported separately.

The abstention option is a promising operational remedy if pooled appropriate-
decision accuracy improves by at least 5 percentage points with a cluster 95%
interval above zero, while cue-visible polarity accuracy falls by no more than
5 points. This is a decision threshold for this controlled set only, not a
generalization or publication gate. Report all outcomes whether it passes or
fails.

## Provenance and limits

Expected new factorial: 192 stimuli × 2 prefixes × 3 judges = 1,152 judgments.
Verify the reused 035 prediction and stimulus hashes, record all new model hashes,
revisions, runtime and parse coverage, and audit exact paired joins before
analysis. Keep private run files in ignored `.context/evidence-aware-abstention-036/`.
Public output may include the short synthetic texts, label-only outcomes, and
aggregate analysis. Since no-cue abstention gold follows a deliberately
constructed incomplete prefix, this does not validate abstention calibration on
open-ended natural answers. LessWrong remains undecided.
