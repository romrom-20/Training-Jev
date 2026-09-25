# Experiment 041 analysis amendment

After the preregistered run began, it was paused after 60 of 1,440 judgments because the strict JSON parser marked Markdown-fenced JSON as invalid. Among those first 60 raw strings, 51 used one outer `json` code fence, 7 were bare JSON objects, and 2 were non-JSON prose. I inspected response envelopes only; I did not inspect predicted score values or compare them with gold labels.

Before resuming, the parser is amended to accept a bare JSON object or a single outer triple-backtick fence whose optional language tag is exactly `json`. The enclosed value still must be one JSON object with numeric `valence` and `arousal` fields in `[1, 9]`. Prose, extra material outside the fence, malformed JSON, missing fields, and out-of-range values remain invalid. The original raw responses are preserved unchanged; the same parser is applied to cached and future outputs. No judgment is retried or changed.

This is a post-start format-handling amendment. It is reported separately rather than silently editing the frozen protocol. The registered invalid-output threshold still applies after this narrowly defined normalization. The runner code and tests record the amended parser.
