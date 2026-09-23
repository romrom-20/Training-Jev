# 012 — Does the shared positive-score shift transfer across task structures?

**Prospective extension frozen after 011 and before new task-structure outcomes are collected.** Experiments 010–011 found that the small prompt-effect readout failed its gate, the three learned aspect directions were aligned, and their shared component reproduced almost all of the broad positive-minus-negative logit shift on mixed reviews. Experiment 011 did not establish that this shared shift was task-selective. This extension asks whether it transfers to the three other already-frozen 008 task structures: isolated clauses, neutral distractors, and keyed records.

Related work already covers activation-steering side-effect forecasting and answer-encoding sensitivity ([Ong et al. 2026](https://arxiv.org/html/2608.11227v1); [Gao et al. 2026](https://arxiv.org/html/2608.22985v1)). This is a small local robustness test on fixed candidate-token scores. It does not add a new general steering algorithm or measure generated behavioral outcomes.

## Frozen setup

Use Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct; block 24; format-0 final-test prompts from all four 008 task structures; and the fixed 5% median training activation-norm dose. Recompute the three unit source directions from mixed-review training groups only. Let `u` be the unit-normalized mean of those directions and `c` their mean projection onto `u`. Construct one deterministic random control (seed 20260927), orthogonal to `u` and matched to `c`'s norm.

On every prompt, measure the change in queried positive-minus-negative next-token log-odds after adding each native source direction (`d_food`, `d_service`, `d_value`), the shared component `c`, or the random control `r`. The directions and control are fixed across prompts and task structures. The shared-versus-random comparisons are primary for the three task structures not used to discover the shared shift: isolated clause, neutral distractors, and keyed record. Mixed-review outcomes are a descriptive bridge to experiments 010–011. The total design is 7,200 interventions per model.

## Primary rule

Within each primary task and model, estimate the mean paired log-odds effect difference `shared − random` and a two-sided 95% interval from 5,000 bootstrap resamples of that task's 24 scenario groups (seed 20260928). Evidence that the shared shift transfers beyond mixed reviews requires the interval to lie entirely above zero in **all three primary tasks in both models**. Do not loosen this all-cells rule after seeing outcomes. Report unpaired task-level means, all source-by-target cells for native directions, and target selectivity descriptively; no sign-accuracy endpoint is used.

Because the same benchmark and final-test group construction recur, this is a held-out-task-structure extension, not an independent replication. The score is still a finite intervention effect on synthetic candidate labels. Passing the rule would motivate replication on natural aspect-sentiment data with remapped answer encodings; it would not support a broad sentiment-control claim by itself.

## Compute and integrity

Reuse only the saved 008 activation cache and local model weights. Run models sequentially on the 24 GB MacBook Air, batch at eight, and checkpoint outcomes by unique intervention ID. Save complete per-row outcomes, protocol/code/source hashes and manifests. Keep binary activations and model weights out of Git; preserve results 010 and 011 unchanged.
