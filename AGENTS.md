# Repository Guidelines

007 Solver is a Windows/macOS heads-up postflop solver: C++17 engine/service, Rust/Tauri bridge, React/TypeScript UI. It starts in the GTO Wizard preflop browser and solves the selected heads-up line and flop on demand. The next priority is CPU solver accuracy and performance.

See [README.md](README.md) for setup, commands and repository layout, [solver/ARCHITECTURE.md](solver/ARCHITECTURE.md) for data semantics and ownership, and [tests/README.md](tests/README.md) for reference fixtures.

## Project constraints

- Use `resources/gtowizard-preflop/` for application and test ranges. Do not introduce custom strategy ranges or fill missing source branches with estimates. Derive small test subsets with `scripts/sync-preflop-fixtures.py`; regenerate expected answers with the independent oracle.
- Keep JSON and presentation details out of core, game and engine. Coordinate protocol changes across C++ serialization, Rust envelopes and TypeScript types; Rust passes node JSON through.
- Each `NodeId` identifies one complete action and concrete-card history in one tree. `InfoSetKey` is a node plus an exact private hand. Preserve original suits and flop input order; use `Card.h` helpers.
- Amounts use tenths of a chip, with no BB conversion. Player 0 is hero and player 1 is villain on the wire.
- Keep amount-to distinct from chips committed, node EV from zero-sum utility, and input/own/joint reach weights from one another.
- Export one snapshot and release training state before evaluating exploitability. The service retains the result; the browser caches queried reports.
- Service stdout is one protocol JSON message per line; diagnostics go to stderr.
- Avoid speculative backend, model or storage frameworks. Let a second real implementation establish shared interfaces.

## Changes and verification

- Use an independent subagent review for architectural changes and check documentation against the final code.
- Do not add tests unless explicitly requested. Adapt existing tests when interfaces change, preserving independent expected values and precision. Never replace external references with this solver's output.
- Run the relevant build and check commands in README.md. Include untracked source files with `pre-commit run --files`; `--all-files` only covers the Git index.
- Reuse the project's existing build and output directories. Do not create additional build directories under `build/` for verification, logs, backups or reports, unless it is neccessary.
- After relevant checks pass, repeat or broaden verification only for new changes, failures, or unresolved concerns.
- Rust formatting hooks name crate roots and their edition explicitly; keep them aligned with Cargo targets. Inspect desktop layout changes using the current build.
