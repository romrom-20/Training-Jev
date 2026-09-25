# Experiment 042: structured-output feasibility

## Diagnostic result

The audit covers 48 prompts that failed the Experiment 041 response parser and 48 parseable aspect-plus-opinion controls, each generated once with free greedy decoding and once with a finite 82-output constraint. Free decoding returned invalid objects on 0/96 prompts; constrained decoding returned invalid objects on 0/96.

For the prior-invalid prompts, free generation yielded 0 estimate objects, 48 explicit insufficient objects, and 0 invalid responses. Constrained generation yielded 0 estimates, 48 insufficient objects, and 0 invalid responses. On 1 paired numeric controls, exact VA-pair agreement was 1/1 (100.0%); mean absolute change was 0.000 valence points and 0.000 arousal points.

The JSON summary also reports response counts by source condition, language, and prior-validity group. Those are descriptive cells from this selected audit, not inferential comparisons.

## Interpretation

This is an adaptive technical audit selected on Experiment 041 parser outcomes. It does not measure VA accuracy and does not establish that an `insufficient` response is correct. The constrained decoder can alter numeric content; the comparison is descriptive and no score-versus-gold analysis was performed. The constrained choices use integer VA candidates and are not a general structured-generation benchmark.

No review text, aspect strings, IDs, or item-level responses are redistributed. Raw material remains in ignored `.context/`.

- Aggregate counts: `summary.json`
- Protocol: `docs/experiments/042-structured-output-feasibility.md`
- Runner/analyzer: `scripts/run_structured_output_feasibility_042.py`, `scripts/analyze_structured_output_feasibility_042.py`
