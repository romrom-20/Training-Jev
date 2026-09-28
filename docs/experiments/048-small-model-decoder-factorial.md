# Experiment 048 — Does constrained decoding amplify masked-context gains in Qwen2.5-1.5B?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 047. Commit this protocol, runner, analyzer, and tests before scoring new held-out outputs. This reuses the existing sample and is not an independent data replication. Laya's selection is agenda triage; its probabilities are uncalibrated.

## Question and literature boundary

In Experiment 043, Qwen2.5-3B under a finite one-decimal VA grammar showed an aspect-only minus opinion-masked RMSE gain of +1.845. Experiment 047 used ordinary greedy decoding but also removed the one-decimal/increment instruction; the free-decoding gain was +0.175. A post-hoc same-ID comparison showed a large difference, but it could not separate decoder constraints from numeric-format wording. Experiment 048 asks whether the decoding method changes the context gain in Qwen2.5-1.5B when both arms receive exactly the same prompt.

Prior work already studies semantic accuracy under constrained generation and effects of output-token conventions, including recent results on small-model scales ([Schall & de Melo, 2025](https://aclanthology.org/2025.ranlp-1.124/); [Hamilton & Mimno, 2026](https://aclanthology.org/2026.gem-main.18/); [Chavan, 2026 preprint](https://arxiv.org/abs/2609.23742)). This experiment does not claim novelty for constrained decoding. It checks one aspect-conditioned continuous VA contrast and one local model-size boundary.

## Sample and factorial conditions

- Reconstruct exactly the 217 held-out IDs, target indices, and Russian/Ukrainian/Tatar rows used in Experiments 043–047. Verify the pinned test source hashes, 043 protocol hash, and 043 private output hash `1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8`; verify all gold VA pairs.
- Use the same two text conditions as 043: `aspect_only` and `opinion_masked`. For each ID/language/condition, create a `finite_grid` arm and a `free_greedy` arm, for 217 × 3 × 2 × 2 = 2,604 judgments.
- The prompt text must be byte-identical across the two decoder arms and must retain the 043 instruction: exactly one JSON object with valence and arousal from 1.0 to 9.0 in increments of 0.1, exactly one decimal place.
- Keep prompts, IDs, text, raw generations, and per-item predictions in ignored `.context/`.

## Model and decoding

- Use `Qwen/Qwen2.5-1.5B-Instruct` at the task-ladder pinned revision on local MPS, greedy.
- In `finite_grid`, use the exact 6,561-candidate one-decimal VA grammar from 043. In `free_greedy`, use the same identical prompt without a candidate trie, `do_sample=False`, and `max_new_tokens=40`.
- The free parser accepts one JSON object with exactly numeric `valence` and `arousal`, finite and in [1, 9]. A single fenced JSON object is accepted. No retries. If more than 2% of the 1,302 free outputs are invalid, classify the run as a protocol-execution failure and withhold all score contrasts.
- Assert identical prompt strings across decoder arms for every item, verify every grid candidate's token boundary, and verify both arms cover the complete paired design.

## Outcomes and rule

- For each decoder, compute the paired context gain `RMSE(aspect_only) - RMSE(opinion_masked)`. Positive values mean masked context improves VA error.
- **Primary contrast:** `context_gain(finite_grid) - context_gain(free_greedy)`. Positive values mean finite-grid decoding amplifies the apparent value of masked context relative to free generation, under identical wording.
- Bootstrap the 217 source IDs with replacement, preserving three languages and both VA dimensions; use 10,000 draws with seed `20260948`. Report both per-decoder context gains and the primary interaction with percentile 95% source-ID cluster intervals.
- Evidence for a practically meaningful decoder interaction in this sample requires a primary estimate of at least +0.25 VA points and a lower interval bound above zero. Failing this rule is inconclusive; it does not prove decoder equivalence.
- Report invalid coverage, descriptive language-specific gains, runtime, model/data/protocol hashes, and output validity. Do not publish item-level outputs or text.

## Limits

This uses the same public benchmark, selected held-out IDs and Qwen family as 043–047. The model size differs from the 3B decoder-sensitivity test. A positive interaction would not establish that all constrained decoding inflates accuracy; it would show that this particular finite candidate grammar changes this model's masked-context contrast. The free and finite arms share wording, but differ in output-space restriction by design. Dataset annotations are not fresh human ratings.

## Provenance

- Parent outcomes: `results/expanded-opinion-mask-repair-v1/README.md`, `results/opinion-mask-order-model-size-v1/README.md`, and `results/free-decode-opinion-mask-sensitivity-v1/README.md`.
- Private Laya agenda trace: `.context/laya-research-triage-048.json` (not committed).
- Runner/analyzer: `scripts/run_small_model_decoder_factorial_048.py`, `scripts/analyze_small_model_decoder_factorial_048.py`.
