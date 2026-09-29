# Experiment 076: role and corpus controls

## Result

Paired 8−2 expected-score shifts use the normalized probability distribution over the 81 registered one-decimal scores. Intervals resample recipient sentences, not output strings. The sample is small; these are finite-set estimates.

### A. Same final request and one-key output schema

| Model | Target order | Assistant prior shift | User prior shift | Assistant − user |
|---|---|---:|---:|---:|
| qwen-0.5b | Valence first | 1.744 [1.637, 1.857] | 1.328 [1.240, 1.419] | 0.416 [0.335, 0.494] |
| qwen-0.5b | Arousal first | 1.805 [1.664, 1.946] | withheld by >2% invalid gate | withheld by >2% invalid gate |
| qwen-1.5b | Valence first | 0.625 [0.564, 0.689] | 0.931 [0.822, 1.041] | -0.306 [-0.379, -0.233] |
| qwen-1.5b | Arousal first | 0.031 [-0.008, 0.071] | -0.120 [-0.222, -0.025] | 0.152 [0.083, 0.227] |

### B. Fresh SemEval-2014 laptop sentences

| Model | Valence-first shift | Arousal-first shift | Difference |
|---|---:|---:|---:|
| qwen-0.5b | 2.327 [2.136, 2.532] | -0.457 [-0.550, -0.358] | 2.785 [2.633, 2.943] |
| qwen-1.5b | 0.872 [0.718, 1.019] | 0.308 [0.081, 0.568] | 0.565 [0.307, 0.809] |

Greedy shifts, valid probability mass, and per-arm parsing rates are in `summary.json`.
The preregistered 2% validity gate failed for Qwen-0.5B's prior-user, arousal-first
role condition: 12/24 outputs were invalid with anchor 2.0 and 5/24 with anchor
8.0. The raw failures commonly emitted an `affect` key instead of the requested
remaining dimension. The arousal-first role contrast for this model is therefore
withheld. All fresh-corpus arms and the other role-control arms had zero invalid
greedy outputs.

The fresh corpus retained a large field-order difference in both sizes, but the
0.5B arousal-first shift reversed sign: it was −0.457 here versus +0.114 on the
previous 24 DimABSA reviews. Its valence-first shift was similar (+2.327 versus
2.577). The 1.5B shifts were smaller here than on the same 24 DimABSA cases
(+0.872/+0.308 versus +2.008/+1.012). Since Part B changes corpus, sentence
style, and sample at once and has no VA gold labels, this is evidence that the
conditional effect is not obviously invariant across contexts; it does not say
which change explains the difference.

## Interpretation and limits

Finite-set paired effects in artificial forced-coordinate prompts. Part A reuses public texts and still bundles role with an acknowledgement turn; Part B uses 24 new-to-project texts without VA gold labels. The Part A user-location prompt says “the other coordinate” without naming the JSON key. Its high Qwen-0.5B arousal-first invalid rate may reflect this wording or role interaction; Experiment 077 tests the same role comparison with the target axis named explicitly.

A shift that persists in Part A would rule out the two-key-versus-one-key schema change as the sole explanation for Experiment 075, but it would not establish a pure causal role effect. A Part B pattern would be a small transfer signal; a null or interval spanning zero would leave the interaction unresolved rather than prove equivalence.

The results do not establish novelty against the broader literature on anchoring, output order, or structured-output effects. See [Experiment 076 protocol](../../docs/experiments/076-schema-and-corpus-controls.md), [Kapetanovic et al. (2026)](https://arxiv.org/abs/2608.25869), [Chen et al. (2024)](https://arxiv.org/abs/2406.02863), and [Parikh (2026)](https://arxiv.org/abs/2607.18476).

## Reproduction

Run the frozen protocol with `PYTHONPATH=scripts:src .venv/bin/python scripts/run_schema_corpus_controls_076.py`, then analyze with `PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_schema_corpus_controls_076.py`.

Raw prompts and outputs remain in ignored `.context/`.
