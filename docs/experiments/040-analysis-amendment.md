# Experiment 040 — secondary-interval reporting amendment

Date: 2026-09-25.

After the first preregistered analysis pass, I found that the analysis code
reported the secondary accuracy and both-target-correct rates as point estimates
only, although the frozen protocol says to use review-cluster bootstrap
intervals for these secondary outcomes. The code now adds those already-specified
intervals, along with paired natural-versus-aspect-only and opinion-visible-
versus-natural accuracy contrasts. It does not change the stimuli, prompts,
primary endpoint, diagnostic gate, or primary estimate. No new judgments were
run. The 10,000-replicate intervals use the already-frozen review clusters and
seed. This is a reporting correction to carry out the frozen secondary analysis,
not a new confirmatory test.
