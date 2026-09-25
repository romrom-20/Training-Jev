# Experiment 035 — Does the polarity cue arrive after the short prefix?

## Preregistration

Freeze this protocol, the deterministic stimulus generator, and the tests before
running any of the three evaluators. No model judgments have been collected for
these stimuli as of this commit.

## Motivation and literature boundary

Experiment 034 found a post-hoc negative-source accuracy gain between 8 and 12
tokens, but TripR labels describe the source review rather than the generated
answer. The class split was also discovered after inspecting the pooled outcome.
This experiment uses deterministic phrase-level gold and moves the same target
polarity clause across a fixed short-prefix boundary.

This is motivated by, and deliberately narrower than, existing results:

- LLM-judge work has already separated preference length effects from genuine
  answer completeness, including direct truncation controls
  ([Soumik, 2026 preprint](https://arxiv.org/abs/2604.23178));
- aspect-sentiment work reports that negation and contrast benefit from deeper
  aspect-conditioned reading
  ([Xia et al., ACL 2026](https://aclanthology.org/2026.acl-long.667/));
- aspect-sentiment benchmarks and LLM studies are established
  ([Bai et al., Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.460/)).

The question here is whether revealing an otherwise identical polarity clause
between short prefixes causally changes small LLM evaluators' accuracy, and
whether that effect depends on polarity or negation. This does not test natural
answer quality or human preference.

## Frozen controlled stimuli

Generate the full factorial from the constants in
`scripts/run_cue_position_polarity_035.py`:

- 16 neutral, seven-word frames × 3 aspects (`food`, `service`, `atmosphere`)
  = 48 scaffold clusters;
- balanced positive/negative gold;
- direct clauses (`The food was good/bad.`) and negated clauses
  (`The food was not bad/good.`), paired so the root adjective is shared across
  polarities;
- early position: polarity clause followed by the frame; late position: the
  same frame followed by the polarity clause.

This produces 384 complete answers (48 scaffolds × 8 polarity/form/position
conditions), 192 per gold polarity. Every direct or negated clause is fully
visible in the early eight-word prefix. In the late condition the first seven
words are neutral frame and the clause begins at word 8; each complete clause is
fully visible by word 12. The same word-delimited surface text is provided to
both binary judges. Here “8/12 prefix” means whitespace-delimited words, not
model-specific subword tokens. The full answer contains no more than 12 words.

Tests must verify the 48-cluster factorial, polarity balance, exact frame length,
and cue visibility at both boundaries. The generator must fail on any stimulus
that violates those constraints. No outputs may be inspected before this
protocol and implementation are committed. The exact frozen stimulus JSON is
[`035-cue-position-stimuli.json`](035-cue-position-stimuli.json), SHA-256
`5b68f5d4dff58953536a41612b1933a9bc9639a13f242a93c35de72b42aa5054`.

## Procedure and endpoints

For each of the 384 answers, make 8- and 12-word prefixes from the exact same
string. Judge each with the existing fixed wrappers and pinned local revisions:

- Laya four-way polarity choice, including `mixed` and `unclear`;
- Qwen2.5-3B and Phi-3 Mini binary polarity judgments.

Use greedy decoding, maximum four generated label tokens, local MPS, and one
judge model loaded at a time. Do not expose gold labels, stimulus condition,
other judges' outputs, or the complete answer when judging an eight-word
prefix. Laya's key-to-label map is deterministic from the frozen experiment
seed. Qwen/Phi use experiment 032's exact single-answer wrapper.

**Primary endpoint:** difference-in-differences in strict binary accuracy:
`(accuracy_12 − accuracy_8)_late − (accuracy_12 − accuracy_8)_early`, pooled
with equal weight over Qwen/Phi and all balanced polarity/form cells. A positive
value means the 8→12 gain is specific to cases where the cue is revealed between
the checkpoints, beyond any generic second-sentence/order effect. Use a paired
scaffold-cluster bootstrap with 10,000 draws and seed `20260935`; a scaffold is
the exact frame × aspect pair, retaining all conditions, prefixes, and both
judges together.

Report each judge's primary contrast separately; accuracy by polarity, form,
cue position, and prefix; and Laya's mixed/unclear rate by position and prefix.
Show 95% scaffold-cluster intervals. The 12-word capability check is descriptive
and predeclared: interpret the cue-timing result only if pooled binary accuracy
at 12 words is at least 90% and early-position accuracy at 8 words is at least
85%. Otherwise mark the manipulation/capability check failed and do not claim
that a null interaction refutes cue availability.

No p-value threshold, universal claim, or publication decision follows from this
experiment. A small controlled stimulus set can isolate a mechanism while
remaining far from natural language validation.

## Reproducibility and interpretation

Expected factorial: 384 stimuli × 2 prefixes × 3 judges = 2,304 judgments.
Record exact model/package revisions, hashes, seeds, coverage, parse failures,
runtime, and audit all joins before analysis. Keep private execution files in
ignored `.context/cue-position-polarity-035/`; public artifacts contain the
synthetic stimuli and label-only predictions. A robust cue-position effect would
motivate a natural-data follow-up with answer-level labels. No effect, a failed
capability check, or strong negation-specific errors will each change the next
experiment. LessWrong remains undecided until those results and validation needs
are reviewed.
