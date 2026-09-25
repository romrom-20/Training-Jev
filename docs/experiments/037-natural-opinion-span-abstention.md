# Experiment 037 — Do judges track annotated opinion evidence in natural text?

## Motivation and scope

Experiment 036 found that offering `insufficient` improved decisions on prefixes
where controlled sentiment cues had been removed. That result could depend on the
highly regular synthetic templates. This follow-up uses natural SemEval review
sentences and ASTE triplets, whose annotations link a target aspect, an opinion
span, and sentiment. The task is narrower than general abstention: does a judge's
decision policy change when a human-annotated opinion phrase becomes visible in
its natural sentence context?

ASTE is established work, not a new task. General abstention on unanswerable
questions is also well studied. A targeted literature search did not locate this
specific paired manipulation of target-linked opinion-span visibility in an
aspect-sentiment judge. Treat any apparent gap as provisional; this is a small,
local replication idea, not a novelty claim. References and search scope are listed
below.

## Data and frozen selection

Use the `test_triplets.txt` files in ASTE-Data-V2-EMNLP2020 at upstream commit
`d0df6600b259b6114de23cc5047c7e776cd89750`, from 14res, 14lap, 15res, and 16res.
The files derive from SemEval restaurant/laptop review datasets and encode each
sentence as whitespace tokens plus `(target-token-indices, opinion-token-indices,
sentiment)` triplets. Keep raw review text and per-example prompts private under
ignored `.context/`; release source URLs/hashes and label-only model outcomes, not
the review sentences. Upstream dataset license metadata is absent, so do not
redistribute its text.

Include only sentences with exactly one triplet, polarity `POS` or `NEG`,
contiguous target and opinion spans, the target entirely before the opinion, and
at least two whitespace tokens before the opinion starts. These restrictions
create a natural prefix ending immediately before the only annotated opinion
phrase and a paired prefix ending immediately after that full phrase. The earlier
prefix can still contain implicit or unannotated evidence; operationally, the
primary “no annotated opinion span visible” condition is not a claim that a human
would find the excerpt wholly unanswerable.

Within each dataset, keep every eligible negative item and sample the same number
of positive items uniformly without replacement. Use Python `random.Random`
seed `20260937`, datasets in this fixed order: 14res, 14lap, 15res, 16res. Freeze
the selected line indices and token spans in `037-stimuli.json`. The source audit
found 27, 29, 39, and 22 eligible negative items by dataset, respectively, yielding
**234 balanced sentence/aspect items** (117 per polarity) and 234 source-sentence
clusters. The selected identifier/span file, which contains no review text, has
SHA-256 `22dea039b2b235f8cfeb09b3f104d811f7097176f4a640d37b6fc3b9ef27d2d6`.
No model outputs may be inspected before this protocol and selected-ID file are
committed.

## Judging procedure

For every selected item, ask about the named target aspect in two exact contiguous
prefixes: (A) sentence tokens strictly before the gold opinion span begins; (B)
sentence tokens through the end of the full gold opinion span. Do not show the
gold polarity, span offsets, condition name, or the rest of the sentence.

Run Qwen2.5-3B and Phi-3 Mini locally on MPS with greedy decoding, at the same
pinned revisions as experiment 036. Each judge receives both prompts on both
prefixes: a forced-binary `positive`/`negative` wrapper and an abstention-enabled
wrapper that permits `insufficient` when the visible excerpt lacks an explicit
opinion expression supporting the named aspect. Run Laya 0.3.20 once per prefix
with its pinned four-way `positive`/`negative`/`mixed`/`unclear` choice. Map `unclear`
to abstention for summary metrics; preserve `mixed` as a distinct, non-abstaining
outcome. Hold Laya's randomized choice mapping fixed for identical aspect × visible
text inputs. Load models one at a time; publish no free-form generations.

The expected operational decision is `insufficient` before the annotated opinion
span and the annotated POS/NEG polarity after it. This uses the annotation to mark
when its opinion phrase is visible. It cannot prove that all evidence is absent
before the span or that the phrase alone is sufficient for a human reader.

## Outcomes and frozen decision rule

**Primary endpoint:** paired change in appropriate-decision accuracy from the
forced-binary wrapper to the abstention-enabled wrapper, equally weighting both
prefix states and both generative judges. An answer is appropriate when it matches
the operational expected decision above. Bootstrap paired outcomes by source
sentence, retaining both prefixes and both wrappers/judges together; use 10,000
draws and seed `20260937`.

Report per-judge changes; appropriate accuracy by wrapper and prefix state;
abstention recall before the opinion span; false-abstention rate and polarity
accuracy after it; strict parse coverage; Laya's mapped outcomes; and per-dataset
results. Forced-binary decisions cannot abstain, so their score on the pre-span
condition is reported transparently rather than described as a calibrated baseline.

The operational rule passes only if the pooled gain is at least 10 percentage
points with a cluster-bootstrap interval above zero, and cue-visible polarity
accuracy under the abstention wrapper is no more than 5 points below the paired
forced-binary accuracy for either model. Report all outcomes regardless. This gate
does not establish human evidence sufficiency, broad model-family generalization,
or a new field-level result.

## Provenance and analysis

The runner verifies all four raw-file hashes and the frozen protocol hash, creates
private prompt text from the source files, and writes only line/span identifiers
and label outcomes into the portable result bundle. Record model revisions, local
device, runtimes, parse coverage, source hashes, stimulus hash, outcome hash, and
audits for selection balance and complete paired joins. Keep all generation-time
artifacts containing review text in ignored `.context/`.

## Literature checked

- Xu et al. (2020), *Position-Aware Tagging for Aspect Sentiment Triplet
  Extraction*, defines ASTE triplets as target, sentiment, and opinion span:
  <https://arxiv.org/abs/2010.02609>.
- Kirichenko et al. (2025), *AbstentionBench: Reasoning LLMs Fail on Unanswerable
  Questions*, evaluates broad unanswerable-question abstention and reports that
  system prompting can improve abstention without resolving it:
  <https://arxiv.org/abs/2506.09038>.
- Wen et al. (2025), *A Survey of Abstention in Large Language Models*, provides
  a broad taxonomy of abstention settings and methods:
  <https://aclanthology.org/2025.tacl-1.26/>.
- ASTE-Data-V2 source/schema and attribution:
  <https://github.com/xuuuluuu/SemEval-Triplet-data>.

Search scope on 2026-09-25: targeted searches for LLM abstention combined with
opinion span, aspect sentiment, evidence span, ASTE, and judge/evaluator. This
does not constitute an exhaustive systematic review.
