# Exploratory analysis — target wording × anchor source (Experiments 076–078)

This combined analysis was selected by local Laya after the three runs had
finished. It is **post hoc**, not a preregistered factorial experiment. It
reuses the frozen 24-case DimABSA role-control sample and reads outputs from
Experiments 076, 077, and 078; it adds no model inference.

## Available factors

- Wording: Experiment 076's final request (“the other coordinate”) versus
  Experiments 077–078's request that explicitly names the fixed axis, target
  axis, and required JSON key.
- Anchor source: prior assistant score versus prior user score followed by a
  neutral assistant acknowledgement.
- Model: Qwen2.5-0.5B versus Qwen2.5-1.5B.
- Field order: valence-first versus arousal-first.
- Anchor level: 2.0 versus 8.0.

All factor combinations share the same recipient IDs and the same opinion-masked
review/aspect context. Only final-request wording and model checkpoint vary
between Experiments 076 and 077/078. This is a useful within-sample contrast,
but it inherits the role/acknowledgement limitation from the frozen protocols.

## Outcomes and validity handling

For each recipient and factorial cell, compute the 8−2 change in expected
target score after normalizing the 81 canonical one-decimal numeric
continuations. Resample recipients in paired 10,000-draw bootstraps with seed
`20260979` for cell means, wording contrasts, role contrasts, and the
wording-by-role difference-in-differences.

Apply the original per-arm rule: withhold a conditional shift if either forced
anchor arm for that model × wording × order × role cell has more than 2%
invalid greedy outputs. In particular, the Qwen-0.5B generic-wording,
prior-user, arousal-first cell fails this gate (17/48 invalid responses across
anchor levels); contrasts requiring that cell are withheld. Do not treat
theoretical numeric-grid likelihoods from a failed output-format cell as a
valid generation result.

This analysis is descriptive for the fixed texts and fixed models. It does not
establish a general role effect, a causal cognitive mechanism, or novel
structured-output sensitivity. It should be used to decide whether an
independent recipient replication is worth the available local compute.
