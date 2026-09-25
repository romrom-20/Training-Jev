# Experiment 042: structured abstention versus free generation

**Purpose:** technical follow-up to Experiment 041's output-coverage failure. This is not a replication of its scientific VA comparison.

## Question

On the same prompts where free Qwen2.5-3B generation did or did not return a parseable VA object, does finite-choice token-constrained decoding preserve an explicit abstention option, and how much do its numeric outputs agree with free generation when free generation was parseable?

## Motivation and literature

Experiment 041 produced 48 invalid responses among 1,440 outputs, mostly in its aspect-plus-opinion-only condition. Its preregistered score analysis was correctly withheld. Laya selected a small constrained-output audit as the next engineering question; this selection is prioritization only, not evidence.

This is established methods territory, not a novelty claim. Structured-generation benchmarks distinguish schema compliance from output quality, and recent controlled work reports that constraints can change the content of instruction-tuned models even as they improve format reliability. We therefore measure syntax and paired answer stability separately and do not treat a valid JSON object as a better VA judgment.

## Sample and prompt

- Reuse the exact Experiment 041 prompts and pinned Russian/Ukrainian/Tatar data source.
- Include all 48 Experiment 041 outputs that the amended parser marked invalid.
- Add 48 controls: 16 per language, selected by SHA-256 order from parseable `aspect_opinion` outputs in Experiment 041. Selection uses language and prior parse status only, not VA gold or predicted values.
- This is an adaptive diagnostic sample selected on the first run's formatting outcome. It is not an independent sample and has no confirmatory population interpretation.
- For both decoders, replace only the final output-format instruction with the same instruction: return either `{"status":"estimate","valence":N,"arousal":M}` for N,M in [1,9], or `{"status":"insufficient"}` when the supplied evidence does not support an estimate. All evidence fields, aspect strings, and text remain unchanged.
- Compare ordinary greedy generation with a finite-trie constrained greedy decoder over exactly 82 candidate JSON objects: 81 integer VA pairs and one explicit insufficient response. The constrained candidate grammar is intentionally small and is not a general JSON-schema engine.
- Run the frozen Qwen2.5-3B-Instruct revision locally on MPS. Use no gold scores in prompts or output analysis.

## Outcomes

- Valid response rate under free and constrained decoding, using the two exact schemas above (an optional single outer `json` code fence is accepted by both parsers).
- On the 48 previously invalid prompts, count constrained outputs choosing `estimate` versus `insufficient`.
- On the 48 prior-valid controls, report exact VA-pair agreement between free and constrained numeric estimates and the average absolute change in each dimension. If either output is invalid or insufficient, report coverage and omit that pair from numeric agreement.
- Show these counts by prior-validity group, source condition, and language. Do not use VA gold labels or compute accuracy/RMSE/correlation against gold.
- No hypothesis test or field-level claim is attached to this adaptive engineering sample.

## Limits

Constrained choices change the decoding distribution; integer candidates discretize the scores. A constrained `insufficient` choice is not human-validated evidence sufficiency. The control sample is selected by prior parser success, and the invalid group by prior parser failure. Any status differences are descriptive and cannot establish calibrated uncertainty or general effects of constrained decoding. The reused dataset is public and not blind. Keep all prompts, IDs, and outputs in ignored `.context/`; publish only aggregate counts and model/data provenance.

## Provenance

- Dataset: [DimABSA official repository](https://github.com/DimABSA/DimABSA2026), same pinned revision and file hashes as Experiment 041.
- Related benchmark: [JSONSchemaBench](https://arxiv.org/abs/2501.10868).
- Structured-decoding quality effects: [Schall and de Melo (2025)](https://aclanthology.org/2025.ranlp-1.124/).
- Runner: `scripts/run_structured_output_feasibility_042.py`.
- Analyzer: `scripts/analyze_structured_output_feasibility_042.py`.
