# Working within a 24 GB MacBook Air

Start with **one frozen 0.5B target at a time**, short prompts, batch size four, and
three captured token vectors per prompt. Train readout heads on CPU after extraction.
The default run uses existing cached weights when `--offline` is supplied. Without it,
the Hugging Face loader may download the pinned model and tokenizer. No paid API or
cloud service is used by this code.

## Why this fits

The default model has approximately 494 million parameters. Float32 parameter storage
alone is about 1.84 GiB (`parameters × 4 / 2**30`); real runtime use is larger. Raw
float32 activation storage is `N × layers × hidden_dim × 4` bytes. For 416 examples,
three layers and width 896, this is about 4.27 MiB before compression, plus query vectors.
We do not save all tokens or all layers, and do not build target-model gradients or
optimizer states. The vocabulary head only runs on the last token.

The included [run manifest](../results/pilot/manifest.json) records actual wall time,
activation-cache size and capture-process peak RSS. **RSS is not total unified-memory
usage**, especially with Metal allocations. Use Activity Monitor's memory pressure
and swap readings before increasing model size. The run was measured on an Apple M5
MacBook Air with 24 GB unified memory; other Air models will have different throughput.

## Scaling order

First expand data diversity and held-out tasks, then sweep small heads over cached
activations. Next try a 1.5B Qwen2-family target with an explicitly pinned model revision
and valid layer indices; the adapter currently validates only the Qwen2 architecture.
A different architecture needs a tested adapter, not just a new model string.

Float32 parameters for 1.5B are roughly 5.6 GiB; 3B roughly 11.2 GiB; 7B roughly
26.1 GiB. Those are weight-only estimates. **The current float32 7B path is unsuitable
for 24 GB.** Quantized or half-precision backends may fit larger models, but are not
implemented or validated here. Quantization can change the activations under study,
so it is a methodological change as well as a memory optimization.

Keep the context limit at 256 until needed. The fanless Air can throttle on sustained
work; recorded short-pilot timing should not be extrapolated linearly to hour-long runs.
Checkpoint captures between runs rather than keeping multiple targets resident.

## Reproducibility

`uv.lock` pins Python dependencies. Model revision and all protocol parameters are
captured in the manifest. The code records a source SHA-256, git revision and dirty
status, dataset/cache hashes, platform, dtype and library versions. Metal floating-point
results need not be bitwise identical across hardware or versions. Compare metrics
and intervention outcomes within reasonable tolerances; do not silently reuse a
capture with a different configuration.

Run artifacts in `runs/` are local and ignored by git. The small selected results in
`results/pilot/` are intended for review; large model weights and binary activation
caches are not committed. Regenerate those caches from the pinned configuration.
