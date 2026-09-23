# Latent Decisions — seed research brief

**Request:** support an eight-week feasibility study of small, calibrated semantic
readouts for language-model activations. Working budget: **US$15,000**, a planning
proposal rather than a priced grant application: $12,000 researcher support, up to
$2,000 replication compute, and $1,000 independent reproduction/review. Adapt to the
applicant's circumstances and the funder's eligible costs. No compute has been
purchased or committed.

## The problem

Generative activation decoders offer a convenient interface for asking questions of
model internals, but cost, decoder reasoning, calibration and text-reconstruction
confounds complicate their use. Separate probes are cheap and competitive, but require
property-specific supervision. We want to measure when a shared, tiny readout earns
its additional complexity.

## The research bet

Train a question-conditioned head on a frozen small model's activations. Evaluate
probability quality on unseen scenarios and property families; separately test whether
its activation directions influence the corresponding model behavior. The output is
a bounded probability estimate about a labeled property, with an explicit scope of
validity. It is not an explanation generator or a general deception detector.

This is a **provisional empirical contribution**, not a claim to invent activation
oracles, constrained-choice decoding, calibration or steering. The closely related
2026 calibration-oracle work makes the small-head and cross-property comparison
particularly important. See the [primary-source review](prior-art.md).

## Evidence already available

A reproducible local pilot uses a frozen Qwen2.5-0.5B-Instruct model on 416 controlled
prompts, three extraction layers and three probe seeds. At the validation-selected
layer, the shared head reaches approximately 95% accuracy; independent probes are
slightly stronger. Accuracy falls to approximately 79% on a held-out prompt template.
Shuffled activations and swapped questions bring accuracy near chance. A privileged
full-prompt text baseline solves this constructed task. This is engineering feasibility
and a useful failure signal, **not evidence of superiority or real-world safety utility**.

The code, per-example results, local runtime measurements, calibration diagnostics
and target-model intervention records are included. The original model is never
trained; no external model API is required. The project has not yet produced
unseen-property transfer or cross-model replication results.

## Why the current hardware is enough to start

The pilot runs on a 24 GB Apple Silicon MacBook Air. Cached activations and small CPU
readouts make supervision/rank/seed sweeps inexpensive. Initial funding should buy
research time and independent evaluation. Cloud compute is reserved for a specific
replication that exceeds local constraints; large decoder training is not assumed.

## Deliverables and gates

1. **Weeks 1–3:** a leakage-resistant property-family benchmark, matched-budget baselines,
   and a report of where query sharing helps or fails.
2. **Weeks 4–5:** a controlled test of behavioral specificity with natural patching,
   random controls and explicit collateral effects.
3. **Weeks 6–8:** second-model replication, independent rerun and a public technical report,
   including negative results.

The [research plan](research-plan.md) defines proposed continuation and stop criteria.
A useful outcome can be a negative result establishing that cheap independent probes
are the stronger default, plus reusable evaluation infrastructure.

## Scope and readiness

Repository status: local research prototype with a completed controlled pilot and tests.
No publication, external validation, partnership, user traction or funding is claimed.
Before submitting, the applicant should add their own relevant background and schedule,
choose a funder, and tailor the budget and application format. The technical claims
should remain tied to the included evidence.

## Follow-up evidence, 23 September 2026

The subsequent [research bundle](../results/followup/README.md) includes failed target-competence
checks and a partially successful score-correction result that failed its broader
continuation criterion. These results support a scoped evaluation/diagnosis proposal,
not a claim of a novel or superior activation-oracle architecture. Any application
should include them alongside the original pilot.
