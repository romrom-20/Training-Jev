# 009 — Does an explicit one-word constraint rescue target behavior?

**Status: frozen before collection; run on 23 September 2026.** This follow-up is motivated by the completed 008 behavior results. In format 1, SmolLM2-1.7B emits a non-label first token on every final-test row, while its conditional positive-vs-negative score remains informative. The experiment distinguishes a response-policy failure caused by omitted answer-format instruction from a task-comprehension failure.

## Question and design

Use the same four tasks, contexts, aspects, label combinations, cue variants, split groups, models, revisions, and label-token scoring as protocol 008. Cross each prompt with three paired prompt conditions:

- **A (reference):** protocol 008 format 0, including “Reply with exactly one word”.
- **B (replication):** protocol 008 format 1, “Choose just its sentiment” without the explicit one-word constraint.
- **C (constraint control):** protocol 008 format 1 wording, with the exact one-word constraint added at the end: “Reply with exactly one word: positive or negative.”

Only instruction wording differs between B and C; context and requested aspect stay identical. Collect all 64 groups for Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct, on the locally available MPS device, using the revisions from 008. This is a target-behavior diagnostic; do not fit probes or use 008 activations.

## Endpoints and decision rules

The primary endpoint is final-test strict greedy answer accuracy (first generated token is exactly the correct label), paired by prompt. Also report (i) answer-token compliance (first token is either label), (ii) conditional accuracy among the `positive`/`negative` logits, (iii) total softmax mass on those two label tokens, and (iv) paired group-bootstrap 95% intervals for B-to-C accuracy changes. Report every task, format, aspect, both models and both selector/final-test splits. No post-hoc task selection.

The response-format explanation is supported if C improves SmolLM2 final-test strict accuracy over B by at least 50 percentage points on isolated clauses and neutral distractors, reaches at least 90% on each of those simple-task/aspect cells, and reduces conditional accuracy by no more than 3 points. If C does not meet that rule, the output-policy explanation is not supported by this control. Mixed-review and keyed-record results remain separate diagnostics: they can fail even when simple task behavior is rescued. The threshold is a decision rule for this follow-up, not a general claim about prompt robustness.

## Compute and reporting

The run is 11,520 prompts per model (23,040 total), in batches of eight, sequentially on the MacBook Air M5 with 24 GB unified memory. No 3B scale run is included: protocol 008's optional Qwen 3B run caused sustained paging and was stopped at 4,232/7,680 prompts. Save complete target predictions and input rows with model revision, source/protocol/dataset hashes and runtime; exclude activation caches. Stop if sustained paging or thermal/resource pressure makes the run impractical, preserving completed model outputs and marking incomplete models explicitly. Never treat partial runs as results.

Protocol 008 remains the independent capability ladder; its 90% strict-generation gate was not cleared, so no probe study is authorized by this follow-up.
