# Prior art and the actual research gap

Literature check: **22 September 2026**. Primary abstracts were inspected in ego-browser;
Torrielli et al.'s v2 paper and the original Activation Oracles article were also read.
This is a targeted search, not an exhaustive novelty review. Publication dates below
refer to first public versions unless specified. Reported results belong to their authors.

## The original idea has substantial precedent

**[LatentQA — Pan, Chen & Steinhardt](https://arxiv.org/abs/2412.08686)**
(December 2024; revised March 2026, ICLR 2026) trains a decoder to answer
natural-language questions about activations and demonstrates activation reading and
steering. Asking semantic questions of hidden states and testing control are therefore
not new contributions here.

**[Activation Oracles — Karvonen et al.](https://alignment.anthropic.com/2025/activation-oracles/)**
(December 2025) develops generalist activation decoders, including binary classification
questions, and reports transfer to auditing tasks. It explicitly identifies excessive
decoder expressivity, inference expense, and the distinction from mechanistic explanation
as limitations. Our motivation is closely aligned with those stated limitations.

**[Confidence and Calibration of Activation Oracles — Torrielli et al.](https://arxiv.org/html/2605.26045v2)**
(May 2026; v2 August 2026) is the closest prior art to the supplied notes. It compares
confidence readouts on a secret-word task across four oracles and finds that explicitly
scoring candidate answers improves accuracy and confidence discrimination. It also
studies post-hoc calibration. Consequently, constrained choices, oracle confidence,
and temperature scaling cannot be presented as our novelty. Its trained LLM-oracle
readout differs from the tiny direct activation head explored here; its results do not
establish that such a head transfers across unseen semantic properties.

**[Building Better Activation Oracles — Bauer et al.](https://arxiv.org/abs/2606.02609)**
(May 2026; revised June 2026) introduces AObench and addresses vagueness,
hallucinations, and text-inversion confounds. Its evaluation approach is a relevant
next comparison: recovering an explicit setting in our pilot might simply recover
input text. This is intentionally a smoke test, not a solution to that confound.

## The remaining ingredients also have precedents

**[Tuned Lens — Belrose et al.](https://arxiv.org/abs/2303.08112)** (March 2023)
uses affine probes to read vocabulary distributions from intermediate layers and
includes causal experiments. Layerwise readout trajectories and probing plus intervention
are established ideas. Here the readout is about a specified property rather than
next-token vocabulary prediction. Layerwise decodability does not date the emergence
of a belief or decision.

**[Patchscopes — Ghandeharioun et al.](https://arxiv.org/abs/2401.06102)** (January 2024)
unifies a range of hidden-representation inspection methods through patching and
language-model readouts. Interventions are part of an existing methodological landscape,
not an automatic novelty claim or proof that the readout identifies a mechanism.

**[Designing and Interpreting Probes with Control Tasks — Hewitt & Liang](https://arxiv.org/abs/1909.03368)**
(September 2019) studies whether probe accuracy reflects representations or the probe's
own ability to learn. It motivates explicit capacity comparisons and controls. Our
shuffled-label check is a simple diagnostic, not a reproduction of their word-type
control-task construction.

**[CORAL — Miao, Cho & Ungar](https://arxiv.org/abs/2602.06022)** (February 2026)
uses regularized activation probes for calibration-aware inference-time steering and
reports transfer on multiple-choice benchmarks. Combining calibration, probes, and
steering is already explored. Our focus would need to be transfer across query-defined
properties and the conditions under which small heads fail.

**[Beyond Linear Probes — Oldfield et al.](https://arxiv.org/abs/2509.26238)**
(September 2025; revised April 2026) introduces truncated polynomial classifiers for
adaptive-cost activation monitoring. Cheap structured readouts and monitoring cost
tradeoffs are also established research directions.

**[Certified Interventional Fidelity — Asiaee](https://arxiv.org/abs/2607.08349)**
(July 2026) makes intervention distributions and causal estimands explicit and gives
statistical certification tools. This argues for carefully specified intervention
claims. Our small fixed-sample bootstrap intervals are not CIF certification and are
not valid for adaptively stopping when a desired result appears.

**[Jev announcement — TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**
(September 15, 2026) motivates typed probabilistic decisions and describes RLCD. The
announcement is a vendor account, not evidence that Jev can consume model activations.
The inspected material does not provide an implementable architecture or complete RLCD
objective. This repository implements neither Jev nor RLCD and is unaffiliated with
TypeSafe. Schema validity alone does not guarantee factual accuracy or calibration.

## A defensible, provisional direction

**Under a fixed small-head budget, when does sharing a question-conditioned readout
improve sample efficiency across properties, and when do apparently good probabilities
fail under prompt and intervention shifts?**

The potential contribution is a measured answer, a reusable evaluation protocol, and
possibly a useful small-head training recipe. It is not a claim to have invented
semantic activation decoding. We did not locate a primary source that settled this
exact combination of budget, cross-property transfer, calibration and behavioral
specificity; a targeted search cannot establish its absence.

The interesting possibility is that a cheap head becomes an **auditable test of a
bounded hypothesis family**, with an explicitly measured domain of validity. The
interesting negative possibility is that semantic query sharing adds little beyond
separate linear probes, especially when the number of properties is small. Both
outcomes are publishable only with substantially stronger evaluation than this pilot.

## Corrections to the supplied notes

- A proper scoring rule encourages honest probabilities in expectation; finite-sample
  training does not guarantee empirical calibration or calibration under shift.
- Brier loss is not exclusively a calibration penalty: it also captures resolution
  and uncertainty. Report raw and calibrated scores separately.
- A high refusal/deception score does not identify a belief or intent. Specify an
  observable label and its collection process before assigning semantic meaning.
- A successful gradient intervention shows influence under that intervention. It does
  not establish necessity, on-manifold behavior, or natural causal use.
- A low-rank bilinear head has no inherent ability to answer arbitrary unseen queries.
  Frozen query embeddings and supervision determine which transfer is even possible.
- An SAE comparison is complementary and task-dependent. No SAE experiment is included
  yet, so this repository makes no claim of outperforming or replacing SAEs.

## Follow-up search, 22–23 September 2026

The deeper search substantially narrows the novelty claim again:

**[What Does Activation Steering Control? — Gao et al.](https://arxiv.org/html/2608.22985v1)**
(August 2026) is direct prior art for answer-remapping controls. It holds interventions
fixed while changing answer encodings, distinguishes semantic from identifier following,
and locates much of one effect in output-sensitive components. Our Experiment 002 is
a small-probe investigation in that existing evaluation direction, not its invention.
The source was found after freezing our local protocol and while its run was underway.

**[How Reliable are Causal Probing Interventions? — Canby et al.](https://arxiv.org/html/2408.15510v3)**
(2024; inspected February 2025 version) evaluates completeness and selectivity of
interventions and compares method families. It reinforces the need to test collateral
changes and explains why a direction's behavioral effect is insufficient on its own.

**[An Interpretability Illusion for Subspace Activation Patching — Makelov et al.](https://arxiv.org/abs/2311.17030)**
(2023) demonstrates that control and faithful feature attribution can diverge. Matched
donor interventions improve our controls but do not bypass this conceptual limitation.

**[Diagnosing Correctness Probes under Self-Judgement Confounding — Lu](https://arxiv.org/abs/2607.16799)**
(July 2026) constructs conflict cases where correctness and self-judgment separate,
showing that transfer alone does not establish the intended semantics of a readout.
Its controlled-disagreement logic is relevant to the interpretation of our probes.

**[Forecasting Side Effects of Activation Steering — Ong et al.](https://arxiv.org/abs/2608.11227)**
(2026) studies structured cross-effects and forecasts them from unsteered representations.
The idea of predicting steering side effects is therefore also established prior art,
not an unexplored novelty claim for our roadmap.

**[Calibrate Before Use — Zhao et al.](https://proceedings.mlr.press/v139/zhao21c.html)**
(ICML 2021) corrects prompt-dependent answer bias using content-free calibration inputs.
Our score-transport experiment instead uses unlabeled source/target score means, but
prompt-format bias correction is a longstanding idea. An additive correction working
on this dataset would be an empirical diagnosis, not a new calibration algorithm.

The remaining interest is in a reproducible failure profile of small activation
readouts: separating target competence, semantic ranking, threshold placement,
calibration and intervention effects. A targeted search still does not establish
novelty of that combination, and a useful local result need not merit a research post.
