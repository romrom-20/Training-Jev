# 015 — Do frozen aspect directions select the right side of a conflicting review?

**Prospective protocol, 24 September 2026.** Experiment 014 showed that synthetic
aspect directions transfer to SemEval restaurant reviews as strong positive-minus-
negative logit shifts, but their paired aspect-specificity effect was practically
negligible. Its most informative limitation was that just eight eligible reviews had
opposing food/service/price labels. I want to test the same question on MAMS, which was
created because ordinary ABSA test sets often let sentence-level sentiment stand in for
aspect sentiment. Jiang et al. define MAMS so each sentence has at least two aspects
with different polarities ([EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1654/)); the
authors' [public dataset repository](https://github.com/siat-nlp/MAMS-for-ABSA) provides
both term-level and category-level XML.

This experiment asks whether a direction trained on synthetic food, service, or value
reviews selectively changes the score for the corresponding natural review aspect when
the same sentence also expresses another polarity. It is a focused transfer test, not a
new ABSA task or a claim of a new steering method.

## Frozen data and mapping

Use only the MAMS-ACSA test split (`data/MAMS-ACSA/raw/test.xml`) from repository
revision `cddcdb0f423b3fdb2c76b70744737abed3a00d17`, SHA-256
`cc0dc3c5b711653daa9ac886c6517d3ebf8752954b73715692d7765fec4a57f5`. Keep the XML in
ignored `.context/datasets/`; do not copy review text into this repository's result
bundle. The test split has 400 sentences. Restrict target aspects to the prespecified
semantic crosswalk: `food` and `menu` map to the frozen food direction; `service` and
`staff` map to the frozen service direction; `price` maps to the frozen value direction.
Ask about the merged category group in the prompt (food/menu, service/staff, or price)
and use that group's gold polarity. If two included aliases in a sentence map to the
same group but disagree in polarity, drop that group for that sentence rather than
arbitrarily choosing one label. Keep positive/negative labels only; require at least two
distinct mapped groups in the sentence. A deterministic audit of the frozen file yields
35 eligible sentences / 71 category queries, including 18 sentences with opposing
polarities across mapped groups. Abort if any count differs.

## Capture and endpoint

Use the exact directions, layer 24, dose, random vector and target versions from
experiments 008/014. On each eligible sentence/group prompt, capture unmodified
positive and negative next-token logits, then apply each native direction, the shared
component and the norm-matched random component. Repeat shared and random controls under
each source label as in 014. Score the change in `logit(positive) - logit(negative)`
relative to the identical prompt's baseline. Record strict baseline and steered accuracy
and valid-label rate. Do not tune prompts or choose directions based on MAMS outcomes.

The primary endpoint is the sentence-clustered `native diagonal - native off-diagonal`
specificity contrast minus the same contrast for the norm-matched random vector across
all eligible test sentences. Bootstrap the original sentence IDs 5,000 times using the
fixed seed `20260924`. Report models separately. Also report shared-vector
specificity, class balance, per-category effects and the same specificity contrast on
the 18 polarity-conflict sentences. The conflict subset is a prespecified key
stratification, but do not treat a small or model-discordant effect as confirmed.

## Interpretation

Call this a replication of practical natural aspect selectivity only if the native-minus-
random interval is above zero in both models and the effect is large enough to matter
relative to the generic margin shift (at least 5% of that shift in each model). A tiny
precisely nonzero logit difference is not practical selectivity. If the generic shift is
large but that criterion fails, conclude that this fixed direction recipe transfers as
a sentiment-score bias on MAMS, not as reliable target-aspect control. Only interpret
generated-choice changes if both models exceed chance strict accuracy on the unsteered
test prompts.

Run the cached Qwen2.5-1.5B and SmolLM2-1.7B sequentially in float32 on the 24 GB Air,
batch size eight. Hash the XML, source directions, code and protocol in manifests and
retain resumable checkpoints. Because the benchmark code repository has an Apache-2.0
license but this does not by itself clarify separate rights for every review text, keep
the raw dataset local and exclude all text from published artifacts.
