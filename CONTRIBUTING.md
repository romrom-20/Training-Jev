# Contributing

Useful contributions improve a falsifiable comparison: an independently labeled
property family, a leakage test, a baseline, an adapter with verified intervention
semantics, or a reproduction with complete artifacts. Negative results are welcome.

Install with `uv sync --frozen --extra dev`. Before proposing a change, run
`uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q`.
The tests use tiny randomly initialized models and do not download model weights.

For experiments, create a new run directory. Preserve the config, model revision,
split definitions, predictions, runtime information and failed controls. Do not
select seeds, layers or intervention doses on the test split. Changes prompted by
test results belong in a separately labeled follow-up experiment.

Keep large weights, caches and private data out of git. Respect the upstream model
and dataset licenses; the project's MIT license does not relicense them. Do not
label instruction recovery as intent detection or call a steering effect proof
of a mechanism. See [the protocol](docs/protocol.md) for definitions.
