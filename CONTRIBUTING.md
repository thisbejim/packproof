# Contributing

Thanks for helping make packed-data checks boring and dependable.

1. Create a focused issue or pull request with a synthetic JSONL fixture.
2. Keep the core dependency-free and offline.
3. Add a regression test for every new finding or parser behavior.
4. Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
   and `uv run mypy` before opening a pull request.

Diagnostics should be deterministic, stable across Python versions, and free
of raw token payloads. New format support should document the expected field
shape and how ambiguous cases are handled.
