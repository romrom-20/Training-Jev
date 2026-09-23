# Controlled task ladder 008: behavior gate not passed

**Run 23 September 2026 on the MacBook Air M5, 24 GB RAM.** Two local instruction-tuned model families completed the frozen 7,680-prompt target-behavior screen: Qwen2.5-1.5B and SmolLM2-1.7B. The screen crossed four task types, three queried aspects, two prompt formats, eight label combinations where applicable, four held-out lexical cue variants, and disjoint group splits. The exact protocol is [`docs/experiments/008-controlled-task-ladder.md`](../../docs/experiments/008-controlled-task-ladder.md).

**No task cleared the predeclared 90% exact-generation gate on both core models, so no probe was fitted.** On final-test prompts, mean strict accuracy across aspects was:

| Task | Qwen format 0 | Qwen format 1 | Smol format 0 | Smol format 1 |
| --- | ---: | ---: | ---: | ---: |
| Isolated clause | 100.0% | 83.3% | 100.0% | 0.0% |
| Neutral distractors | 100.0% | 58.3% | 100.0% | 0.0% |
| Mixed review | 76.4% | 47.9% | 69.4% | 0.0% |
| Keyed record | 97.2% | 23.6% | 69.4% | 0.0% |

Format 0 explicitly says to reply with exactly one word. Format 1 asks the model to choose the sentiment but does not include that exact wording. Under format 1, SmolLM2's first generated token was not a label on any final-test prompt (usually “the”). Yet its positive-vs-negative conditional ranking accuracy was 100% on the two simple tasks and 70.5% on mixed reviews. Qwen's corresponding format-1 conditional accuracy was 100%, 100%, 77.1%, and 93.1%. Label-token probability mass was low under this format (about 2% for SmolLM2 and 18–28% for Qwen), so conditional ranking does not mean the model was likely to emit a valid label. This points to an output-policy/label-ranking dissociation worth a controlled follow-up; it is not evidence of an activation readout or a novel mechanism.

The mixed-review endpoint remains difficult even when the explicit one-word constraint is present. Under format 0, Qwen matched the overall three-aspect majority label on 76–78% of mixed-review queries, but answered the queried aspect correctly on only 46–54% of cases where the queried label conflicted with that majority. SmolLM2 reached 69–76% overall-majority matching and 38–46% queried-aspect accuracy on those conflict cases. These are descriptive outcomes on balanced synthetic templates, not natural-review estimates or proof that majority-following causes the errors.

All four tasks failed the selector gate; final-test gates also failed. The follow-up choice rule was applied only after both core captures completed. The optional Qwen2.5-3B scale check was interrupted at 4,232/7,680 prompts after sustained paging on this 24 GB machine; its partial run is excluded from analysis. No probe stage ran.

The compact bundle contains complete target outputs/labels (`dataset.json.gz`), manifests, cell summaries, specificity diagnostics, and SHA-256 checksums. Activation arrays (hundreds of MB) are excluded. Original local run directories are ignored by Git. To reproduce the analysis from them:

```bash
uv run python scripts/task_ladder.py analyze --root runs/task-ladder-v1
uv run python scripts/package_task_ladder.py runs/task-ladder-v1 results/task-ladder-v1
```

Experiment 009 prospectively tests whether adding an explicit one-word constraint to format 1 rescues exact output on simple tasks: [`docs/experiments/009-output-constraint-control.md`](../../docs/experiments/009-output-constraint-control.md). No LessWrong decision should be made until that follow-up has run and the full evidence has been reviewed.
