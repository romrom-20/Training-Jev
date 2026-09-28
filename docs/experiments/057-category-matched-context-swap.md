# Experiment 057 — polarity- and category-matched context swap

**Status:** preregistered before inference. Laya selected an aspect-family donor
control with low, uncalibrated confidence. We implement the family using official
gold categories from the benchmark's parallel quadruplet test file, avoiding a
hand-curated phrase taxonomy.

## Question

Experiment 056 showed that the decoder-specific matched-review advantage remains
after donor and recipient share the same negative/positive gold-valence polarity.
Could the effect still be explained by broad aspect-topic cues, such as both
reviews mentioning battery quality? Experiment 057 also matches the official
aspect category while replacing each recipient's review with another case's
review.

## Data and sample

- Use the pinned English laptop Task 2 triplet file at DimABSA commit
  `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256
  `08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543`.
- Join each fresh Experiment 055 case to the same-review Task 3 quadruplet file
  at that revision, `task-dataset/track_a/subtask_3/eng/eng_laptop_test_task3.jsonl`,
  SHA-256 `600aeba51fb53f6bbae7f669f03796999b4609cb5eaddfef171ce390d2ff8786`.
  The paired Task 2/3 files use different ID prefixes, so join by exact review
  text and exact target `(Aspect, Opinion, VA)` tuple. The frozen audit found one
  unique category for all 217 fresh cases.
- Keep only cases whose `(gold-valence polarity, official category)` cell contains
  at least two cases. This yields 184 recipients across 45 cells; 33 singletons
  are excluded because no non-self donor exists. No model output is consulted for
  selection. Reuse the exact Qwen2.5-3B own-review `opinion_masked` outputs from
  Experiment 050 after verifying SHA-256
  `890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f` and gold.
- For each eligible polarity-category cell, tokenize each opinion-masked review
  with the pinned Qwen2.5-3B tokenizer and sort cases by token length, then ID.
  For each of three donor assignments, map each case to the case at a deterministic
  cyclic offset in that sorted list. Use the first three distinct offsets from
  `[1, n-1, 2, n-2, ...]` for cell size `n`, wrapping when a cell has fewer than
  three possible derangements. Therefore donor and recipient share gold polarity
  and official aspect category; their exact review/aspect pair differs. In cells
  of size two, all three assignments necessarily use the same swap. Record unique
  assignment count rather than calling these independent random samples.
- Preserve each recipient's target aspect and gold. Produce both decoders for all
  184 recipients and three assignments: 1,104 generations. Review text, category
  labels, IDs, donor maps, raw generations, and item predictions stay in ignored
  `.context/`.

## Model and outcomes

- `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS inference.
- Keep Experiment 050's prompt, one-decimal score grammar, 6,561-candidate finite
  grid, and free-greedy parser settings fixed. Free output maximum is 40 tokens;
  no retry or repair. Prompts are identical across decoder arms.
- Per decoder and assignment, compute matched-review advantage as
  `RMSE(category/polarity-matched donor context) - RMSE(own opinion-masked
  review)` over both VA dimensions. The primary diagnostic is finite-grid
  advantage minus free-greedy advantage. Report each assignment and their mean;
  report the range only across unique mappings and the number of unique mappings.
- Bootstrap 184 recipient IDs with replacement, keeping all three assignments,
  both decoders, own-review outputs, and VA dimensions; 10,000 draws with seed
  `20260957`. The mean-interaction interval conditions on the fixed mappings.
- If more than 2% of the 552 free outputs are invalid, withhold all score
  contrasts; otherwise report complete coverage and invalid counts.

## Limits

This test controls gold polarity and the official broad category, but not
fine-grained aspect identity or which contextual words are relevant. The 184-case
sample excludes category-polarity singletons. It remains one public laptop split,
one model family, and no neutral-valence cases; public pretraining exposure is
possible. The experiment tests whether the finite-decoder gap survives a stronger
donor control, not whether context masking or context selection is novel in itself.

The parallel label file is part of the public DimABSA release; see the
[DimABSA benchmark paper](https://aclanthology.org/2026.acl-long.1881/).
The local Laya agenda trace is ignored at `.context/laya-research-triage-058.json`.

- Runner/analyzer: `scripts/run_category_matched_context_swap_057.py`,
  `scripts/analyze_category_matched_context_swap_057.py`.
