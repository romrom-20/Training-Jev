# Local judge on open sentiment answers (experiment 027)

The Qwen2.5-3B judge first passed its frozen source-review screen at 96.1%
accuracy and 98.2% negative recall. It then classified 699 unconstrained answers,
without seeing the source review or gold label. Raw answers were held only in
memory and are not included in this bundle.

The judge's estimated accuracy against the review's aspect label was 53.6% for
Granite 3.1 2B, 81.1% for Qwen2.5-1.5B, and 87.1% for SmolLM2-1.7B. All three
open-format positive/negative candidate-pair predictions reproduce experiment 025
exactly and have accuracy 93.6%, 94.8%, and 87.1%, respectively. The paired
sentence-bootstrap interval for candidate-score accuracy minus judge accuracy was
[+32.6, +47.2] percentage points for Granite, [+8.6, +19.0] for Qwen, and
[-6.4, +6.0] for SmolLM2. Candidate/judge agreement among parseable judge outputs
was 51.9%, 79.4%, and 80.1%.

This is a measurement lead, not a validated finding about generation. Passing on
human-labeled source reviews does not prove that a judge can interpret model-written
answers. The observed gap could come from generated-answer errors, judge distribution
shift, or both. The result warrants an independent-judge robustness test, not a
LessWrong post or a claim that candidate scores predict correct free-form behavior.

The per-item bundle contains only stimulus IDs, labels, margins, and judge outputs.
It contains no review or completion text. The frozen procedure is in
[experiment 027](../../docs/experiments/027-local-judge-validation.md).
The capture-time runner hash is recorded in `analysis.json`; a reproduction output
path option was added afterward without changing the model or scoring operations.

Recompute from the repository root:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/local_open_judge.py \
  --output /tmp/local-open-judge-reproduction
PYTHONPATH=scripts .venv/bin/python scripts/analyze_local_open_judge.py \
  --root /tmp/local-open-judge-reproduction
```
