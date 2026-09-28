# Experiment 059 — category-matched context control at 1.5B

**Status:** preregistered before any new 1.5B judgments. Laya's pinned local
triage choice selected the model-size transfer (choice C; uncalibrated top
probability 0.286). This is an empirical replication at a second scale, not a
new independent sample.

## Question

Does Experiment 057's attenuation of the finite-grid/free-greedy matched-review
interaction after matching donor polarity and official aspect category also
appear in Qwen2.5-1.5B on the same recipients and donor maps?

## Design

- Use the same 184 fresh English laptop recipient IDs from Experiment 057 and
  the same three donor maps, verified against its private output and manifest.
  This avoids changing the sample or assignments while changing model size.
- Produce the 1.5B `opinion_masked` own-review baseline for every recipient under
  both decoders (368 generations), plus both decoders for each of the three
  category/polarity-matched donor contexts (1,104 generations): 1,472 total.
- Preserve recipient aspect and gold. Donors have the same official DimABSA
  aspect category and negative/positive gold-valence polarity. No neutral cases
  are available in this fresh sample. No items are dropped after output review.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy local MPS inference.
  Keep Experiment 050 prompt wording, 6,561-value one-decimal finite grid,
  free-greedy parser, 40-token cap, and no-retry rule.
- Primary outcome: for each donor map, compute
  `RMSE(category/polarity-matched donor) - RMSE(own opinion-masked review)`
  separately for finite-grid and free-greedy, then finite-minus-free. Report
  the mean over the three fixed maps and a 10,000-draw recipient bootstrap
  (seed `20260959`), retaining all outputs for each sampled ID. The interval
  conditions on the maps. Withhold score contrasts if more than 2% of free
  donor outputs are invalid.
- Secondary scale comparison, frozen before 1.5B outputs: compare this primary
  1.5B interaction to Experiment 057's already observed 3B interaction on the
  same 184 IDs and same donor maps, with a paired recipient bootstrap. Label it
  a cross-size diagnostic; use 10,000 draws with seed `20260960`. The models
  share a family and the single sample does not establish population-level scale
  moderation.

## Interpretation limits

This changes model size while holding cases and donor maps fixed. It does not
test whether category matching itself caused the 057 attenuation, whether exact
aspect identity or specific words matter, or whether the result generalizes to
other corpora. The 3B-versus-1.5B comparison is not an independent replication.
All raw text, donor maps, IDs, and item-level predictions stay in ignored
`.context/`; publish only aggregates and provenance.

The design follows the broader DimABSA evidence base, where context denoising
and synthetic sentiment descriptions have already been studied. It tests the
portability of this particular local decoder interaction rather than claiming
that context sensitivity is novel.

- Prior result: [Experiment 057](../../results/category-matched-context-swap-v1/README.md)
- Runner/analyzer: `scripts/run_category_match_size_transfer_059.py`,
  `scripts/analyze_category_match_size_transfer_059.py`
