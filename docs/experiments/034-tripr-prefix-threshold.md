# 034 — Does the 8-to-12 token transition transfer to TripR?

## Status

Independent-corpus follow-up to experiments 031–033. Freeze this protocol before
generating or judging TripR completions. This changes the review corpus, not the
basic sentiment-extraction task or automated evaluators.

## Question

On SemEval restaurant reviews, Qwen/Phi agreement rose sharply between 8 and 12
tokens. Does the same pattern appear on independently collected and manually
annotated TripR-2020Large TripAdvisor review sentences?

TripR has expert aspect-polarity annotation by three researchers and was used in
prior repo work for activation-transfer tests. It is a different corpus and source
population, but still restaurant sentiment; it is not an unrelated-domain
replication. Prior work on judge length bias focuses on pairwise response preference
(see [MT-Bench](https://arxiv.org/abs/2306.05685) and
[length-bias decomposition](https://arxiv.org/abs/2407.01085)). This tests the
transfer of the observed cap/prefix sensitivity in single-answer aspect extraction,
not a priority claim.

## Frozen dataset filter

Use the pinned local release already documented in experiment 018:
`OD-TripR-2020Large/TripR-2020Large_AnnotatedReviews.xml`, repository revision
`2e3f56f8f3691019139bdec36db0b6115ed7b191`, CC BY-SA 4.0. Use exactly the frozen
018 crosswalk and eligibility rule: food quality/style → `food/menu`, service
general → `service/staff`, categories ending in `#PRICES` → `price`; retain only
positive/negative opinions, remove a sentence-group with conflicting polarity, and
keep a sentence only when at least two distinct mapped groups remain. This must
yield 187 sentences, 385 aspect queries and 29 opposite-polarity conflict sentences.
Do not rebalance, sample, or alter the filter after seeing model outputs.

## Frozen procedure

- For all 385 query rows, use experiment 018's source sentence and question wording,
  but remove its forced one-word response suffix exactly as experiment 025's
  `open_question` format does. Generate once to a maximum of 32 new tokens with
  greedy decoding on each frozen target revision: Granite 3.1 2B, Qwen2.5 1.5B,
  and SmolLM2 1.7B. Load one target at a time on local MPS.
- From each exact 32-token continuation, evaluate nested prefixes at 8, 12 and 32
  actual generated tokens. If EOS occurred before a checkpoint, reuse the completed
  text and report its actual token count. No target answer is regenerated at the
  shorter checkpoints.
- Classify every prefix with the exact Laya four-way typed-choice prompt and
  neutral per-item A-D map from experiment 030, plus the exact Qwen2.5-3B and Phi-3
  Mini binary judge wrapper from experiment 032. Use frozen revisions, greedy
  maximum 4 output tokens, one judge loaded at a time, and do not provide source
  gold, target identity, or another judge's output.
- Primary endpoint: paired 12-minus-8 change in Qwen/Phi agreement among item rows
  where both judges parse at both lengths. Report all-length coverage and agreement,
  the paired source-sentence-cluster bootstrap 95% interval (10,000 draws, seed
  `20260934`), and target-specific direction. Secondary outcomes: Laya
  mixed/unclear rate and both judges' strict accuracy against TripR polarity labels,
  with source-sentence cluster intervals. These source labels are human-annotated
  proxy labels for answer meaning, not answer-level judgments.
- Report all three prefixes and all targets regardless of outcome. The transfer
  pattern is descriptively supported if agreement increases from 8 to 12 for each
  target family and pooled Laya ambiguity declines; intervals are evidence about
  this dataset, not a universal generalization threshold. No p-value gate and no
  publication decision are specified.

## Privacy and interpretation

Keep raw TripR review text and generated answer text only in ignored `.context/`.
The tracked bundle may contain sentence/query IDs, labels, token counts, metrics and
provenance, never review text, answer text, or token IDs. TripR's annotation
agreement is reported as 0.66 in its dataset documentation, so even source polarity
labels have noise. A replicated shift would support corpus transfer within restaurant
sentiment; it would not validate all generated answers or show the threshold transfers
to unrelated tasks. A null or reversed shift is equally informative. No LessWrong
decision follows until this result is inspected alongside answer-level human
validation.
