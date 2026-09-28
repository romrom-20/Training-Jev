# Experiment 047: free-decoding sensitivity

## Result

Free-decoded aspect-only minus opinion-masked RMSE was 0.175 (95% source-ID interval [0.028, 0.321]); the preregistered free-decoding rule did not pass.
Generation took 53.5 minutes after model load.

As a post-hoc same-cluster comparison, the Experiment 043 finite-grid gain was 1.847, versus 0.175 under free decoding; the difference was 1.672 (95% interval [1.526, 1.815]). This secondary comparison was not preregistered, and the output-format wording changed along with the decoder.

This adaptive test reuses the 217 source IDs from Experiment 043 and changes the output contract from finite-choice one-decimal VA to ordinary greedy JSON generation. It is a decoder-sensitivity check, not an independent replication. The free prompt/parser can still shape which generations count as valid.

This direction is consistent with published work showing that constrained decoding and seemingly small output-format choices can change semantic accuracy, including in smaller models ([Schall & de Melo, 2025](https://aclanthology.org/2025.ranlp-1.124/); [Hamilton & Mimno, 2026](https://aclanthology.org/2026.gem-main.18/)). A recent preprint also separates structural validity from semantic accuracy across small-model sizes ([Chavan, 2026](https://arxiv.org/abs/2609.23742)). Those studies do not test this aspect-conditioned VA task. Our same-cluster decoder comparison is explicitly post hoc and changes precision wording as well as the decoding constraint.

No review text, aspects, IDs, raw generations, or per-item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/047-free-decode-opinion-mask-sensitivity.md`
- Runner/analyzer: `scripts/run_free_decode_opinion_mask_sensitivity_047.py`, `scripts/analyze_free_decode_opinion_mask_sensitivity_047.py`
- Aggregate result and provenance: `summary.json`
