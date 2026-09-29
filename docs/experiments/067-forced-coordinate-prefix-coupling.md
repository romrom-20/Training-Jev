# Experiment 067 — Does a forced first VA score pull the second score?

**Status:** Laya-selected after Experiment 066; preregister before inference.
Laya's choice scores are uncalibrated agenda triage only.

## Question and motivation

When Qwen2.5-3B is made to continue a JSON answer after a supplied first
valence-arousal (VA) coordinate, does changing that first coordinate from 2.0 to
8.0 shift the model's second coordinate for the same review?

Experiment 066 did not resolve the prompt-language contrast. The older field-
order result could arise partly because, in an autoregressive model, the first
generated numeric coordinate becomes conditioning context for the second. This
experiment manipulates only that first emitted value through an assistant output
prefix, leaving the review and user prompt fixed. The forced values are
intentionally counterfactual; this estimates conditional continuation coupling,
not the quality of normal affect ratings.

Generic output-order effects are established. The [UKP_Psycontrol SemEval-2026
system](https://arxiv.org/abs/2604.21534) also compares joint and separate
numeric VA prompts, so this is not a generic joint-versus-separate performance
study. Initial literature scan only; no novelty or priority claim is made.

## Cohort and prompts

- Select 64 recipients from Experiment 066's deterministic 88-ID SIGHAN subset,
  using SHA-256 ranking of `exp067-forced-coordinate-prefix-v1|case_id` within
  the following fixed polarity/category cells:
  - Negative: `食物#品质` 16, `食物#份量与款式` 8, `食物#价格` 5,
    `饮料#品质` 3.
  - Positive: `食物#品质` 16, `食物#份量与款式` 16.
  This yields 32 negative and 32 positive recipients and includes all six cells.
  Sorted selected IDs joined by newline have SHA-256
  `ddd1584d1f5151b4e29cf5de1dfbe5bfc6657f3c90599d494f919193b884a621`.
- Reuse the Chinese instruction wording and opinion-masked review/aspect inputs
  from Experiment 066. No donor context is used.
- Model: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy MPS inference on the 24-GB
  MacBook Air; no cloud inference.

## Intervention and generation

For each recipient, run four free-greedy completions. Keep the user message byte-
identical within each order, and prepend one of these partial assistant outputs
after the ordinary Qwen chat-generation marker:

| First field order | Forced first-field value | Assistant output prefix |
|---|---:|---|
| valence, then arousal | 2.0 | `{"valence": 2.0, "arousal": ` |
| valence, then arousal | 8.0 | `{"valence": 8.0, "arousal": ` |
| arousal, then valence | 2.0 | `{"arousal": 2.0, "valence": ` |
| arousal, then valence | 8.0 | `{"arousal": 8.0, "valence": ` |

The supplied prefix is continuation context, not text added to the user's
instruction. The model must generate the second numeric field and close the JSON
object. Use deterministic free-greedy decoding with a 40-token cap, the existing
strict parser, no retries and no repairs. Save the prefix, raw continuation and
parsed prediction only in ignored `.context/` artifacts. Total new generations:
64 × 2 output orders × 2 forced values = **256**.

Before loading the target model, verify the protocol, source, Exp066 parent
output, model revision and selected IDs; verify all four cells for every
recipient and the assistant-prefix/chat-template boundary. Freeze code and
protocol in git before starting inference.

## Registered analysis

For each recipient and output order, compute the second-coordinate change
`prediction(second | first=8.0) - prediction(second | first=2.0)`. The primary
endpoint is the equal-weight mean of these changes across both output orders and
complete recipients. A positive estimate means a higher forced first value
raises the second-coordinate continuation; a negative estimate means the reverse.
The 95% interval is a percentile bootstrap over recipient IDs with replacement,
10,000 replicates and seed `20260967`, retaining the four cells per recipient.

Report the order-specific changes and the unforced free-greedy second scores
from Experiment 066 on the overlapping IDs as secondary context. They do not
alter the primary endpoint. If any forced-value × output-order arm has an
invalid-output rate above 2%, withhold the primary score contrast. Otherwise,
report invalid counts and analyze only IDs with valid outputs in all four new
cells. If the primary interval excludes zero, interpret it only as evidence of
forced-prefix sensitivity in this model, prompt and cohort; it does not establish
that naturally generated scores anchor one another or explain the prior
field-order result.

## Compute and provenance

The preceding Exp066 Chinese-instruction outputs are
`.context/exp066-private-predictions.jsonl`; their SHA-256 is
`47f47317e1e609a5106d3024ebfad978c28c910ca545719a93dcdadfeb71041f`. The
source input and gold hashes remain the pinned SIGHAN hashes in Experiment 052.
Publish aggregate values and hashes only. Public-test pretraining exposure cannot
be ruled out.
