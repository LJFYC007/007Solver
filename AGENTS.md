# Repository Guidelines

Use [README.md](README.md) for build/check commands, [solver/ARCHITECTURE.md](solver/ARCHITECTURE.md) for shared semantics and ownership, and [tests/README.md](tests/README.md) for fixture updates.

## Project constraints

- Use `resources/gtowizard-preflop/` for application and test ranges. Do not introduce custom strategy ranges or estimate missing branches. Derive subsets with `scripts/sync-preflop-fixtures.py` and regenerate expected answers with the independent oracle.
- Preserve the identity, card, amount, EV, reach and lifecycle contracts in `solver/ARCHITECTURE.md` when changing solver or bridge code.
- Keep JSON and presentation details out of core, game and engine. Coordinate protocol changes across C++ serialization, Rust envelopes and TypeScript types; Rust passes node JSON through.
- Avoid speculative backend, model or storage frameworks. Let a second real implementation establish shared interfaces.

## Changes and verification

- Use an independent subagent review for architectural changes and check documentation against the final code.
- Do not add tests unless explicitly requested. Adapt existing tests when interfaces change, preserving independent expected values and precision. Never replace external references with this solver's output.
- Run relevant README checks. Include untracked source files with `pre-commit run --files`; `--all-files` only covers the Git index.
- Reuse existing build and output directories. Create additional directories under `build/` only when necessary.
- Repeat or broaden passed checks only for relevant changes, failures or unresolved concerns.
- Rust formatting hooks must name crate roots and their edition. Inspect desktop layout changes using the current build.

## Documentation

Keep documentation to current setup, workflows and contracts that are easy to misuse. Link to code, configuration, fixtures and `build/benchmark-results/` for implementation details and measurements. Do not duplicate file contents or accumulate historical performance, migration narratives or completed design notes.
