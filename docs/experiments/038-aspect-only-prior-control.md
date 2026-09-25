# Experiment 038 — Aspect prior versus natural-prefix signal

## Status and motivation

This is an **exploratory, preregistered follow-up after seeing experiment 037**; it
is not an independent confirmatory study. In 037, forced-binary Qwen2.5-3B and
Phi-3 Mini predicted the full-review polarity above 73% on balanced prefixes that
ended before the only ASTE-annotated opinion phrase. The same experiment cannot
tell whether that predictive accuracy came from words in the natural prefix, a
prior associated with the named aspect, or pretraining exposure to these
2014–2016 review sentences.

Experiment 038 removes every review word while keeping the exact target aspect,
full-review polarity, and response prompt. This estimates the aspect-only baseline
for the already observed natural-prefix predictions. A positive paired gain from
the natural prefix would show that preceding review context adds predictive signal
beyond the target name alone. It would still not rule out benchmark memorization.

## Frozen sample and control

Use all 234 balanced sentence/aspect IDs in `037-stimuli.json`; do not resample or
change polarity labels. The source data remain the four pinned ASTE-Data-V2 test
splits from experiment 037 and are read only from ignored `.context/aste14res/`.

The sole new input is a fixed aspect-only control. Preserve the exact 037 prompt
template and aspect field, but replace the visible review excerpt with the literal
`[no review text provided]`. Do not show the remaining sentence, opinion span,
gold polarity, or dataset name. This textual placeholder explicitly tells the
judge that the review is absent; it is not a natural sentence and should be
interpreted as a prior baseline rather than a new review judgment.

Run Qwen2.5-3B and Phi-3 Mini locally on MPS with the same pinned revisions and
both 037 wrappers (forced binary and abstention-enabled). Run the pinned Laya
four-way typed choice once on each aspect-only prompt, with deterministic choice
mapping keyed by aspect and the exact placeholder text. Load models one at a time.
Reuse 037's frozen natural-prefix predictions as the paired comparator; collect
only the new aspect-only judgments. Do not publish review text or free-form model
outputs.

## Outcomes and decision rule

**Primary endpoint:** paired difference in forced-binary polarity accuracy between
the natural prefix before the annotated opinion phrase (existing 037 outcomes) and
the new aspect-only condition, equally weighting Qwen/Phi and all items. Bootstrap
by 037 sentence cluster, retain each judge's pair together, and use 10,000 draws
with seed `20260938`.

Report changes by judge; aspect-only accuracy and parse coverage for each wrapper;
the abstention rate on the text-free control versus the natural pre-opinion prefix;
Laya's four-way label counts and `unclear` rate; and results by dataset. The
diagnostic gate passes if the pooled forced-binary natural-prefix advantage is at
least 5 percentage points and its 95% cluster-bootstrap interval is above zero.
Report all outcomes regardless. Since this is selected after 037 and the dataset
is old enough for contamination, passing would indicate added visible-context
signal on these items, not broad natural-language generalization.

## Provenance

Commit this protocol, runner, tests, and the 037-parent hash before collecting
038 outputs. Verify the 037 prediction hash, protocol hash, and unchanged frozen
stimulus list. Store only label-only predictions, source IDs/spans, model revision
and runtime hashes, and audit checks in the portable bundle. Private review text
remains under ignored `.context/`.

## Literature carried forward

- ASTE asks for target/opinion/sentiment relations, so opinion-span visibility is
  an established annotation, not a new task:
  <https://arxiv.org/abs/2010.02609>.
- Spurious correlations are an existing ABSA concern:
  <https://aclanthology.org/2023.findings-emnlp.193/>.
- Training-data contamination is a known benchmark risk and cannot be excluded in
  these old review texts: <https://arxiv.org/abs/2406.04244>.
