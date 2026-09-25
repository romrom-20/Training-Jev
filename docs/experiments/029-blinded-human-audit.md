# 029 — blinded human audit of generated-answer polarity

## Question

Do the large, model-dependent gaps between candidate-pair scores and local LLM
judges in experiments 027–028 reflect real polarity errors in the generated
answers, or weaknesses in automated judges? This is a small adjudication audit,
not a confirmatory estimate of general model quality.

## Frozen sample and coding

- Use the same 233 positive/negative SemEval 2014 restaurant aspect items used by
  experiments 023–028.
- Select 20 unique source sentences with seed `20260925`, stratified by aspect and
  source polarity: food positive/negative 4/3, service 3/4, and price 3/3. If
  duplicate sentence IDs arise across aspect strata, skip the duplicate and take
  the next eligible shuffled item in that stratum. The sample therefore has 10
  positive and 10 negative source labels.
- Regenerate all three frozen experiment-025 open-question systems on those same
  20 items, greedily with at most 8 new tokens, using the existing prompt and
  revisions. Pairing the models on the same prompt is part of the design.
- Present 60 answers in a seeded random order. Hide system identity, source review,
  source gold polarity, candidate-pair scores, and both LLM judges' labels. Show
  only the aspect and generated answer. Keep the mapping and text in ignored
  `.context/human-coding-029/`; do not commit them or place answer text in results.
- The annotator codes the polarity communicated about the named aspect as
  `positive`, `negative`, `mixed`, or `unclear` (including no answer, irrelevant
  content, and truncation). Use the answer as written; do not infer from the
  hidden source review. One annotator is available, so no inter-rater reliability
  claim is possible.

## Analysis, fixed before coding

After annotations are exported, first report completion and the four code counts.
For the primary descriptive comparison, among `positive`/`negative` human codes,
report each target's accuracy against the human code for (a) its frozen candidate-
pair prediction, (b) experiment 027's Qwen2.5-3B judge, and (c) experiment 028's
Phi-3 Mini judge. Report coverage (clear human codes and parseable judge labels)
alongside accuracy; do not silently drop malformed or mixed/unclear items. Also
report human-to-review-gold agreement and the paired per-item disagreements.
Provide sentence-cluster bootstrap 95% intervals with 10,000 draws and seed
`20261029`, resampling the 20 review sentence IDs and retaining all three model
answers within each selected sentence. Because the sample is small and stratified,
these are descriptive intervals, not population-representative confidence claims.

No significance or novelty claim follows from this audit alone. Decide whether a
larger multi-annotator study is warranted only after inspecting disagreement
patterns. Do not write a LessWrong post unless the human audit yields a robust,
interesting finding and the follow-up eliminates plausible artifacts.

## Compute and privacy

Use only the local MacBook Air and cached model weights; no paid or remote
inference. The tracked runner can recreate the blinded local form, while generated
answers, blind mapping and annotations remain under ignored `.context/`.
