# Experiment 039 — Does pre-span context generalize without pretrained LMs?

## Status and motivation

This is an exploratory follow-up to experiments 037–038. The natural prefix before
the sole annotated opinion span predicts full-review polarity at 73–78% for the two
local generative judges, and it exceeds an aspect-only prompt by 31 points pooled.
Because all reviews come from legacy SemEval sets, pretrained completion or
memorization remains possible. This low-compute baseline asks whether lexical
context is learnable across the available dataset subsets without using a
pretrained language model.

## Frozen data and split

Use the same 234 balanced IDs and full-review polarity labels in
`docs/experiments/037-stimuli.json`. Keep the four data subsets as indivisible
folds in the fixed order 14res, 14lap, 15res, 16res. For each fold, train only on
the other three subsets and predict every item in the held-out subset. Fit all
text vectorizers only on the three training subsets. Do not retune or select a
model using held-out outcomes.

Create three fixed feature variants from local raw text, which remains ignored and
private:

1. `aspect_only`: target text only (`aspect: {target}`).
2. `masked_context`: the natural prefix strictly before the opinion span, with
   target-aspect token indices replaced by the single token `ASPECT`.
3. `context_plus_aspect`: named aspect plus the unmasked natural prefix.

Fit a separate `TfidfVectorizer` for each variant and fold with lowercase,
Unicode accent stripping, word unigrams/bigrams, sublinear term frequency,
L2 normalization, and `min_df=1`. Fit binary `LogisticRegression(C=1.0,
solver="liblinear", max_iter=2000, random_state=20260939)`. Do not use pretrained
embeddings, extra lexicons, hyperparameter search, or data outside the three
training folds. Emit only per-ID out-of-fold labels, not source text or vectorizer
vocabularies.

## Outcomes and rule

**Primary endpoint:** paired out-of-fold accuracy difference between `masked_context`
and `aspect_only`, averaged over the 234 examples. Bootstrap sentence IDs 10,000
times with seed `20260939`, retaining all three variant predictions per resampled
sentence.

Report all three pooled accuracies and paired differences; per-held-out-subset
metrics; class-balanced chance (50%); and macro accuracy across the four held-out
subsets. The diagnostic gate passes if masked context improves by at least 5
percentage points over aspect-only and the sentence-bootstrap 95% interval is
above zero. This indicates lexical signal transfers across these four small
benchmark subsets; it does not show broad domain transfer or exclude overlap
between source collections. Include context-plus-aspect as a secondary upper bound,
not a second primary test.

## Provenance and limits

Freeze this protocol, code and tests before fitting on the selected labels. Verify
the 037 protocol, selected-ID file, and four source-file hashes. Keep raw sentence
text and all vectorizer vocabulary private under ignored `.context/`. Publish only
IDs/spans, gold/parsed labels, fold names, metric outputs, fit-library versions,
hashes and audits. The ASTE source repository has no explicit license metadata.
The test subsets are small and related: 14res/15res/16res are all restaurant
reviews, with only 14lap representing a different product type. A positive result
helps distinguish ordinary lexical context from behavior requiring LLM pretraining,
but does not eliminate data overlap or establish human sufficiency.

## Literature carried forward

- [ASTE definition and source task](https://arxiv.org/abs/2010.02609).
- [Spurious correlations in ABSA](https://aclanthology.org/2023.findings-emnlp.193/).
- [Benchmark contamination survey](https://arxiv.org/abs/2406.04244).
