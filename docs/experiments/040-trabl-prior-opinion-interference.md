# Experiment 040 — Does an earlier opposite opinion leak across aspects?

## Status and question

**Preregistered before any target-model judgments.** Experiments 037–038 found
that Qwen2.5-3B and Phi-3 Mini often predicted a review's full polarity from a
natural prefix ending before the target's annotated opinion phrase. Experiment
039 found that a small cross-subset lexical model did not reliably transfer this
signal. This follow-up asks a more targeted question: when the same review has
one human-agreed positive aspect and one human-agreed negative aspect, does a
model carry the earlier aspect's opposite-polarity opinion onto the later
aspect, before the later aspect's own opinion phrase appears?

The direction was selected by the repository's pinned local Laya typed-choice
engine from four candidate follow-ups. Laya selected independent-corpus
replication (choice probability 0.284); the private decision trace is
`.context/laya-research-triage-040.json`. This is agenda triage, not evidence
for the scientific hypothesis.

## Literature and scope

Aspect-level sentiment work has long studied separating target-specific
contexts, including reviews containing opposite polarities for different
targets. The MAMS resource deliberately includes sentences with multiple
aspects and different sentiments. TRABL (WWW 2026) adds aspect, opinion,
sentiment and supporting-snippet annotations for travel reviews. This protocol
does not claim that mixed-aspect classification or cross-aspect interference is
new. The narrower question is the *sequential pre-opinion* case, tested by
deleting only the earlier, annotator-agreed opinion phrase for an opposite
sentiment target, on a newly released travel-review test split.

Primary references:

- [Tang et al. (2019), A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://aclanthology.org/D19-1654/), introducing MAMS.
- [TRABL (Madmon et al., WWW 2026)](https://doi.org/10.1145/3774904.3792835), a travel-domain ABSA task with opinion spans and evidence snippets.
- [Tan et al. (2019), Recognizing Conflict Opinions in Aspect-level Sentiment Classification](https://aclanthology.org/D19-1342/), on conflicting opinions in aspect-level classification.

## Data and frozen selection

Use only `test.jsonl` from
[`Booking-com/absa-dataset`](https://huggingface.co/datasets/Booking-com/absa-dataset),
dataset revision `e9329d05a722000eb7cabaab97c59049d52b94fd`, file SHA-256
`3632412d3e8003364ed67e6877777bb51a474f30917037d36a8f7cedbd1e7884`.
The dataset card describes 900 annotator rows covering 450 paired reviews,
with hotel, attraction and destination reviews. It declares CC BY-SA 4.0 and
non-commercial research use. Raw text stays in ignored `.context/trabl/` and is
never written to tracked artifacts.

Pair annotators by `id_source`; require their review text to match exactly.
Consensus is an exact match after trim/case-fold of aspect term, category,
opinion span, sentiment and supporting snippet. Keep only explicit positive or
negative consensus tuples whose aspect term and opinion phrase each occur
exactly once (case-insensitive) in the review, with the aspect preceding the
opinion, and whose snippet occurs in the review. For each review with both
polarities, select the earliest eligible positive tuple and earliest eligible
negative tuple; require distinct aspect terms and non-overlapping opinion
spans. Order this pair by opinion position. These rules yield 48 review
clusters / 96 aspect trials. The exact IDs, source row numbers, labels and
character offsets are frozen in `040-stimuli.json`; no review text or source
identifier is published.

For each of the 96 target trials, construct:

1. `aspect_only`: no review words, with the target aspect named.
2. `natural_prefix`: review text strictly before this target's annotated
   opinion phrase.
3. `opinion_visible`: the same prefix through the complete target opinion
   phrase.

For only the later target in each 48-review pair, also construct
`prior_opinion_deleted`: the natural prefix with the earlier, opposite-label
target's exact annotated opinion phrase deleted. Do not alter any other text.
This is a direct lexical deletion with a possible fluency side effect; it does
not isolate polarity from all other semantics.

## Models, prompts, outcomes

Run local pinned Qwen2.5-3B-Instruct and Phi-3 Mini with greedy generation and
the same forced-binary and explicit-abstention wrappers as Experiment 037. Run
the pinned Laya checkpoint using a typed four-way positive/negative/mixed/unclear
choice, randomizing answer-key order by the aspect-and-visible-text key. Run
all three models sequentially on local MPS; no API calls, finetuning, or model
training are used. Freeze the runner and tests before inference.

**Primary endpoint:** on the 48 later-target trials, Qwen forced-binary
*opposite-prior copy rate* under `natural_prefix` minus that under
`prior_opinion_deleted`. A copy is a prediction equal to the earlier target's
gold polarity (which is opposite the current target's gold label). Bootstrap
review clusters 10,000 times with seed `20260940`. The diagnostic gate passes
only if the difference is at least 5 percentage points and its two-sided 95%
cluster interval is above zero. Missing/unparseable generations are reported
and do not count as copies; report coverage beside the estimate.

Secondary outcomes: paired target accuracy on all 96 trials by condition and
judge; abstention/coverage; Laya four-way label frequencies; accuracy with the
target opinion visible; paired review rate at which both opposite aspects are
classified correctly; and opposite-prior copy rates for Phi and Laya. Report
all outcomes, not only the gate. Use the same 48 review clusters for bootstrap
intervals. Laya and Phi are descriptive/replication checks, not extra primary
tests.

## Limits and safeguards

The 48 clusters are a selected subset of a single test split, not a broad sample
of travel reviews. The TRABL card reports that annotation began from LLM
suggestions followed by human correction and gives overall two-annotator F1 of
about 67%; exact tuple agreement is used here to reduce, not eliminate, label
noise. The opinion phrase is an annotation marker, not proof that earlier text
cannot support the sentiment. Deleting a phrase can reduce fluency. A model may
have seen source reviews during pretraining even though the dataset was
released in 2026. Results will not support a general claim about metacognition
or causal reasoning. LessWrong remains deferred until the experiment runs and
the result is both interesting and robust enough to merit an independent check.

## Reproduction and attribution

Download the pinned file to ignored `.context/trabl/test.jsonl`, then run:

```bash
python scripts/run_trabl_prior_opinion_interference_040.py
python scripts/analyze_trabl_prior_opinion_interference_040.py
python scripts/plot_trabl_prior_opinion_interference_040.py
```

The derived stimulus metadata and result data are attributed to Madmon et al.,
TRABL, and shared under the dataset's CC BY-SA 4.0 terms. The code remains
under the repository's code license. Raw review text is not part of the bundle.
