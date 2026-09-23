<p align="center"><img src="docs/assets/banner.svg" alt="Latent Decisions — small probes, testable claims" width="100%"></p>

# Latent Decisions

**Can a tiny question-conditioned probe read a language model's activations reliably—and tell us something about what drives its behavior?**

Latent Decisions is a working, laptop-scale research prototype. It pairs a frozen
language model with a small probabilistic readout, evaluates calibration and transfer,
and tests interventions on the original model. The code and a completed local pilot
are included. **There is no demonstrated advantage over independent linear probes yet.**

[Results](results/pilot/README.md) · [Research proposal](docs/funding-brief.md) ·
[Prior art](docs/prior-art.md) · [Protocol](docs/protocol.md) · [24 GB compute guide](docs/compute.md)

## Latest research update — 23 September 2026

The [follow-up studies](results/followup/README.md) add two model sizes, answer-remapping
controls, unlabeled score correction and four fresh prompt formats. The causal task
failed its competence checks; an initially promising correction did not pass the
fresh-format continuation criterion on both models. These results narrow the project
and are included with the positive findings. No LessWrong research-post draft has
been prepared.

The next capability-gated run found that Qwen2.5-1.5B handles simple sentiment
classification but misses the predeclared threshold when asked about service in
mixed-aspect reviews. The probe phase stopped before fitting. See the [aspect
sentiment results and audited outputs](results/aspect-sentiment-v2/README.md).

## What has actually run

A frozen **Qwen2.5-0.5B-Instruct** model on an **Apple M5 MacBook Air with 24 GB RAM**:
416 controlled prompts, three activation layers, three probe training seeds, and no
paid compute. The rank-four query-conditioned head has **8,065 trainable parameters**.

- **About 95% held-out accuracy**, with ordinary linear probes slightly better.
- **About 79% on a new prompt template**: a visible transfer and calibration failure.
- **Near chance after shuffling activations or swapping questions**, supporting
  dependence on both inputs.
- **A larger red/blue steering effect for the color direction than the eight sampled
  random directions**, an exploratory behavioral-specificity signal.

The task reads three explicit instruction settings. A full-prompt text baseline solves
it. It does **not** establish hidden-intent detection, unseen-property generalization,
or a uniquely identified mechanism. See the [complete results and limitations](results/pilot/README.md).

![Measured pilot results: Brier by layer, reliability, and steering effects](results/pilot/overview.png)

## Why this project exists

The inspiration was Jev's structured probabilistic decision interface. This repository
is **not Jev, not an RLCD implementation, and not affiliated with TypeSafe**.

Natural-language activation queries already exist in
[LatentQA](https://arxiv.org/abs/2412.08686) and
[Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/).
A [2026 paper](https://arxiv.org/abs/2605.26045) already studies oracle confidence,
constrained candidate scoring and calibration. Probability outputs alone are not novel.

Our narrower research question is whether **small shared readouts improve sample
efficiency across properties, under a fixed capacity budget**, and whether their
probabilities and behavioral specificity survive meaningful shifts. That remains
untested here. The [literature review](docs/prior-art.md) explains the overlap; the
[research plan](docs/research-plan.md) states the next experiments and stop criteria.

## Run it locally

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a CPU or Apple Silicon
Mac. The pinned environment was tested with Python 3.12. Other GPU adapters are not
included. First use without cached weights downloads the pinned target model.

```bash
uv sync --frozen --extra dev --python 3.12
uv run pytest -q
uv run latent-decisions all --run runs/my-first-pilot
open runs/my-first-pilot/report.html  # macOS; otherwise open this file in a browser
```

Use `--offline` when the model is already cached. No credentials or paid API are
needed. Results directories are preserved: choose a new `--run` for each experiment.

Run each stage separately if preferred:

```bash
uv run latent-decisions collect --offline --run runs/experiment-01
uv run latent-decisions train --run runs/experiment-01
uv run latent-decisions intervene --offline --run runs/experiment-01
uv run latent-decisions report --run runs/experiment-01
uv run python scripts/audit_run.py runs/experiment-01
```

`configs/macbook.toml` pins the target revision, layers, split sizes, rank and seeds.
The report is a standalone local HTML page with interactive layer, evaluation and
calibration selectors. GitHub does not render its HTML as a website: download the
repository and open `results/pilot/report.html`. No server is necessary.

## What the implementation does

```text
Frozen target ── block-output activation h ──┐
                                           ├─ rank-constrained head ── P(setting | h, q)
Frozen question encoder ── query vector q ──┘
                                                       │
                          held-out scores + calibration + interventions on target
```

For a fixed query, the head is affine in the activation. The implementation includes:

- Fully crossed factors and scenario-level train/validation/calibration/test separation.
- Independent linear, query-only, shuffled-label and privileged text baselines.
- Paraphrase, shuffled-activation, swapped-query and mean-activation diagnostics.
- Raw and temperature-scaled Brier, NLL, ECE, AUROC and per-property scores.
- Fixed-dose target interventions, zero sham, random directions, vocabulary KL and token mass.
- Saved predictions, configuration and data hashes, source fingerprint, and an artifact audit.

Temperature scaling uses calibration data only. Layer selection uses validation only.
Tests cover split isolation, proper-score edge cases, query-dependent signal recovery,
activation-gradient scaling, padding, final-token logits and intervention hooks.

## Repository map

```text
configs/macbook.toml       Small-model experiment, pinned revision
src/latent_decisions/     Data, capture, probes, evaluation, interventions, report
scripts/audit_run.py      Recompute reported metrics from per-example predictions
tests/                   Scientific and adapter correctness tests; no downloads
docs/prior-art.md        Primary-source review and limits of novelty
docs/protocol.md         Labels, splits, estimands and known limitations
docs/research-plan.md    Eight-week plan, continuation gates and failure conditions
docs/funding-brief.md    Seed-funding proposal with an illustrative budget
docs/compute.md          Measured pilot and practical 24 GB memory constraints
results/pilot/           Reviewable results, data, manifests and offline report
```

## Contributing and support

We welcome careful reproductions, harder property families, matched-budget baselines
and negative results. See [CONTRIBUTING.md](CONTRIBUTING.md). The funding brief is a
technical starting point for an application; it does not claim external validation,
traction or an existing grant. Researcher background and funder-specific requirements
still need to come from the applicant.

MIT-licensed code. Model weights retain their upstream license. Citation metadata is
in [CITATION.cff](CITATION.cff).
