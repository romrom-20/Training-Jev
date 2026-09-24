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

## Aspect-sentiment and compositionality literature, 23 September 2026

Aspect-based sentiment is a well-developed benchmark domain. Zhou et al.'s
[comprehensive LLM ABSA evaluation](https://arxiv.org/abs/2412.02279) predates this
project, and Liskowski & Jankowski's [reasoning-infused ABSA models](https://arxiv.org/abs/2601.03940)
show continued task-specific work in 2026. Zhong et al.'s ACL 2024 paper
[benchmarks compositional generalization in multi-aspect text generation](https://aclanthology.org/2024.acl-long.351/),
including held-out combinations of attributes. A small synthetic aspect-sentiment
benchmark alone would therefore be a weak novelty claim.

These sources motivate the next controlled task ladder: isolate whether failures
come from selecting a requested property, adding neutral context, resolving
conflicting property cues, or following explicit keyed records; hold out cue
wording and use a second model family. This is an experimental design choice, not
a claim that the task ladder or aspect-based sentiment is new.

## Output compliance and task accuracy, 23 September 2026

The response-constraint control (Experiment 009) also sits inside an established
benchmark area. [FOFO](https://aclanthology.org/2024.acl-long.40/) evaluates
format-following across real-world output structures and reports that format ability
can differ from content quality. [LLMs Are Biased Towards Output Formats](https://arxiv.org/abs/2408.08656)
explicitly separates task accuracy under format compliance from accuracy regardless
of compliance, across multiple formats and tasks. The 2026 [MOSAIC instruction
compliance benchmark](https://aclanthology.org/2026.eacl-long.62/) studies constraint-
specific adherence, interactions, and positional effects. Our tiny two-model paired
result—that an explicit one-word constraint restores valid-label emission for SmolLM2
but does not restore correct aspect sentiment—fits this existing framing. It is a
useful diagnostic, not a new account of instruction-following or activation behavior.

## Prompt-specific intervention-effect forecasts, 23 September 2026

Ong et al.'s [Forecasting Side Effects of Activation Steering](https://arxiv.org/html/2608.11227v1)
(2026) is now the closest causal-prediction comparison. It learns behavioral probes
and a propagation map from unsteered executions, then predicts held-out behavior-pair
cross-effects without running those interventions. Its cross-effect labels pool over
fixed prompt contexts and its evaluation holds out source or target behaviors. This
motivates Experiment 010's distinct unit: a *prompt-specific* finite treatment effect
within a fixed source/target behavior pair, tested on held-out cue groups. The local
study is intentionally much smaller (two sub-2B checkpoints and three synthetic
attributes), so it is an extension pilot and replication signal, not a claim to
supersede that paper. A prompt-level effect forecast would matter because a benign
average effect can conceal a subset of prompts with strong or reversed responses.

## Intervention geometry and local outcome

The follow-up search also found two close geometric precedents that constrain the next
interpretation. [A Geometric Account of Activation Steering through Angle-Norm
Decomposition](https://arxiv.org/abs/2606.06735) separates angular and radial changes
to hidden states and evaluates steering across seven models. [Pre-Intervention
Prediction of Sparse Autoencoder Steering Side Effects](https://arxiv.org/abs/2606.08365)
forecasts feature-level side effects from pre-intervention statistics. Experiment 011
uses a different decomposition: it separates overlap shared among three task-derived
directions from each direction's orthogonal residual, then directly intervenes on those
components in a tiny synthetic task. That is a local controlled diagnostic, not a new
general theory of steering geometry or side-effect forecasting.

Experiment 010 has now tested its frozen prompt-specific forecaster. It misses its
predeclared 10% improvement rule in both models: the small head is worse than the
pair-average on Qwen and only modestly better, with an interval including zero, on
SmolLM2. The first-order gradient reference is much more accurate in both. This is a
useful cost/accuracy result for this setup.

In 011, the three training directions had pairwise cosine similarities from 0.85 to
0.96. The norm-preserving shared component reproduced almost the entire mean
positive-logit increase on both models (about +2.46 and +1.47), while the task
residuals had near-zero mean effects. The preregistered residual-specificity rule
failed because the residual did not beat its norm-matched random control in SmolLM2.
For this synthetic task and dose, the measured candidate-label shift mostly follows a
direction shared by all three task vectors. It does not establish that model behavior
itself is controlled by a general sentiment mechanism; the endpoint remains a
next-token logit difference.

Experiment 012 then passed its frozen task-structure transfer rule: in all six
non-mixed-task/model comparisons, the shared component produced a larger candidate
positive-minus-negative shift than the norm-matched random control. The average shift
was similar across isolated clauses, neutral distractors and keyed records. Native
source-direction means remained almost exactly the same as the shared-vector means,
while target selectivity was small. This increases confidence that the mixed-review
finding was not isolated to that one task structure; it still measures token logits
on a shared synthetic benchmark and reuses its scenario split.

Gao et al.'s 2026 [cross-encoding steering evaluation](https://arxiv.org/html/2608.22985v1)
is direct prior art and the most relevant interpretation check: they freeze steering
directions while changing answer mappings, and show that score gains can follow answer
identifiers rather than semantic labels. Experiment 013 is a smaller-scale replication
of this necessary control. Until a naturalistic test shows robust aspect selectivity, the
011–012 pattern is an interesting local candidate-score effect, not a field-level claim.
## Answer-remapping and natural-review results

Experiment 013 directly tested the interpretation suggested by Gao et al.'s
[Cross-Encoding Steering Evaluation](https://arxiv.org/html/2608.22985v1). On 144
isolated-clause prompts, Qwen2.5-1.5B solved both A/B semantic mappings perfectly
before steering. Under the fixed intervention, the A-minus-B score change stayed
positive after the meanings of A and B were swapped, so the semantic positive-minus-
negative change flipped sign. This is identifier following on this task, not a semantic
sentiment effect. SmolLM2 showed a consistent preference for B, but its reversed-map
unsteered accuracy was 87.5%, below the prespecified competence gate; its semantic
interpretation remains unresolved.

Experiment 014 tested whether the frozen synthetic aspect directions selectively change
the score for the matching aspect in human-annotated SemEval-2014 restaurant reviews.
The official task description defines aspect-category polarity and includes examples
where category labels disagree ([Pontiki et al. 2014](https://aclanthology.org/S14-2004/);
[official task description](https://alt.qcri.org/semeval2014/task4/index.php)). In this
local gold-test slice, 233 queries from 112 multi-category sentences passed the models'
baseline behavior checks. Native directions caused broad positive-minus-negative shifts
(+2.28 logits in Qwen, +1.38 in SmolLM2), but their paired within-sentence specificity
contrast was near zero in Qwen (−0.00114; 95% interval [−0.00323, +0.00136]) and only
+0.00070 in SmolLM2 ([+0.000515, +0.000872]). That small SmolLM2 estimate is about
0.05% of its generic shift. The subset with opposing food/service/price polarities had
only eight sentences and cannot support a confident conflict-specific conclusion.

This extends the earlier synthetic result to natural reviews, but it does not establish
useful aspect-specific control. The benchmark slice is positive-heavy (76%), which also
makes the slight overall accuracy increase under a positive direction sensitive to class
balance. The clear local pattern remains broad candidate-logit movement, not reliable
target-specific behavior. Existing answer-remapping work already establishes the
identifier confound ([Gao et al. 2026](https://arxiv.org/html/2608.22985v1)); this result
does not yet add a sufficiently surprising mechanism or a strong naturalistic effect.
**The current evidence does not justify a LessWrong post.** Reconsider after an
independent dataset supplies many more same-review polarity conflicts and a replicated
specificity effect or a compelling failure mechanism.

## Conflict-rich transfer and intervention dose

Jiang et al.'s [MAMS benchmark](https://aclanthology.org/D19-1654/) deliberately
constructs multi-aspect reviews whose aspect polarities can disagree, making it a
stronger stress test for aspect binding than the small conflict slice in SemEval-2014.
Experiments 015–016 transfer the already-trained directions to its category-level test
annotations; they do not train on MAMS text. On 35 eligible reviews, the native-minus-
random diagonal/off-diagonal score contrast is positive in both small models, including
the 18 conflict reviews, but the full-slice estimate is only 0.08%–0.38% of the broad
positive-score movement. The five-dose follow-up finds that this fraction declines
with dose in both models and remains far below the preregistered practical threshold.
At the largest dose, answer accuracy also falls sharply for SmolLM2. The direction and
size of these estimates are specific to the mapped food/service/price categories, the
two instruction checkpoints and the next-token polarity-score endpoint.

Layer choice is a plausible mechanism to test next, but is not itself a novelty claim.
Stoehr et al.'s [Activation Scaling for Steering and Interpreting Language
Models](https://arxiv.org/abs/2410.04962) explicitly studies intervention points across
layers and optimizes where to intervene. The local open question is narrower: whether
the *fraction of a steering effect that is target-aspect-specific* varies by layer on
human-annotated conflicting reviews, after matching dose to each layer's training
activation norm. Layers 8, 16 and 24 are already cached for both current models. A
layer-wise result would be exploratory until it survives a new benchmark and model
family; selecting whichever layer looks best on MAMS would be post-selection.
