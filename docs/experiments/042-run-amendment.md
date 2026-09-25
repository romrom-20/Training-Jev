# Experiment 042 run amendment

The initial constrained-decoding process stopped after saving all 96 free outputs and 56 constrained outputs. In a batched generation, Transformers appends the EOS token as padding for rows that finish before other rows. The finite-trie callback initially rejected that completed-prefix-plus-EOS-padding state and raised an exception. The failed batch was not written; the 152 saved raw responses are preserved in ignored `.context/` and will be reused.

Before resuming, the callback is amended to allow the registered terminal EOS token when a generated row already matches a complete candidate sequence. All prompt text, the 82 output candidates, greedy decoding, sample, and parser remain unchanged. The runner skips every previously saved `(ID, language, condition, decoder)` key and generates only missing constrained rows. This is an execution amendment; no completion from the first process has been inspected for its numeric values or interpreted.
