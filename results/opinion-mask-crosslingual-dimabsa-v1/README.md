# Experiment 041: opinion-mask cross-lingual DimABSA

## Execution outcome

The preregistered comparison was **not analyzed** because 48/1440 outputs were invalid (3.33%), exceeding the 2% stop threshold. Invalid counts by condition: aspect_only: 0, aspect_opinion: 47, full_text: 0, opinion_masked: 1. There were 77 fully parseable aligned sentence clusters, but the protocol's coverage gate failed, so no RMSE, confidence interval, language contrast, or score-based claim is reported.

Some responses did not provide parseable numeric JSON, concentrated in the aspect-plus-opinion-only condition. Those responses remain invalid; they were not converted to scores or retried. This is a prompt/output feasibility failure for the registered design, not evidence for or against residual context effects.

## Limits and files

The public test split and labels are not blind. The three versions share IDs and gold labels and are treated as aligned clusters, not independent replications. Annotated-opinion masking leaves implicit and unannotated evaluative cues. This is one Qwen2.5-3B checkpoint and does not establish a property of LLMs generally. No source sentences or item-level derivatives are redistributed here.

- Aggregate execution record: `summary.json`
- Aggregate hardware/model/runtime provenance appears in `summary.json` when a private run-metadata file is available.
- Protocol: `docs/experiments/041-opinion-mask-crosslingual-dimabsa.md`
- Parser amendment: `docs/experiments/041-analysis-amendment.md`
- Per-item prompts and model outputs remain local in ignored `.context/`.
