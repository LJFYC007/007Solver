# Repository Guidelines

Run the relevant [README checks](README.md#checks) and follow the [fixture update workflow](tests/README.md#updating-inputs-and-references). Preserve [solver contracts](solver/ARCHITECTURE.md) when changing solver or bridge code.

## Project constraints

- Application and test ranges must come from [the captured catalog](resources/gtowizard-preflop/). Do not introduce custom ranges or infer missing branches.
- Avoid speculative backend, model or storage frameworks. Let a second real implementation establish shared interfaces.

## Changes and verification

- Use an independent subagent review for architectural changes and check documentation against the final code.
- Do not add tests unless explicitly requested. Adapt existing tests when interfaces change, preserving independent expected values and precision.
- Reuse existing build and output directories. Create additional directories under `build/` only when necessary.
- Repeat or broaden passed checks only for relevant changes, failures or unresolved concerns.
- Time performance changes back to back against a rebuilt baseline, and GPU changes also beyond the benchmark's hand counts (see [Benchmark](tests/README.md#benchmark)).
- Rust formatting hooks must name crate roots and their edition. Inspect desktop layout changes using the current build.

## Documentation

Document current setup, workflows and contracts that are easy to misuse. Link to code, configuration, fixtures and reports under `build/benchmark-results/` for details; omit implementation walkthroughs, historical measurements and completed design or migration notes.
