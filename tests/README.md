# Solver tests

Run commands from the repository root. Use Terminal on macOS or Developer PowerShell on Windows; Windows executables have an `.exe` suffix.

## Correctness

Use the [build and test commands](../README.md#checks). CTest writes `build/release/solver-test-results.json`; routine tests use checked-in references and run offline.

Coverage and assertions live in [domain_tests.cpp](domain_tests.cpp) and [tests.cpp](tests.cpp). Scenario inputs and independent expected answers live in [fixtures/](fixtures/); each input's `rangeSource` records its source and reductions.

## Benchmark

The benchmark is separate from the default build and CTest:

```sh
cmake --preset Release
cmake --build --preset Release --target 007SolverBenchmark
./build/release/007SolverBenchmark
```

For Visual Studio CPU sampling that can name functions, use RelWithDebInfo instead. It matches Release's `/O2 /Ob2` and the Windows solver's `/arch:AVX2`, and writes a PDB next to the executable; do not reuse the Release tree:

```sh
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo --target 007SolverBenchmark
./build/relwithdebinfo/007SolverBenchmark
```

Optional flags are `--report=<new-json-path>`, `--iterations=<count>` and `--workers=<count>`. The default workload is [utg-bb-wide.json](fixtures/utg-bb-wide.json); checks and report fields are defined in [benchmark.cpp](benchmark.cpp).

Add `--convergence` to record exploitability every 200 updates while preserving the fixed budget. `--stop-at-accuracy` also stops when the benchmark's accuracy requirement is reached. Both record checkpoint wall time separately from pure training time; use the same checkpoint mode when comparing discount schedules. The final exported snapshot is evaluated independently of the checkpoints.

Reports are local artifacts under `build/benchmark-results/` by default. Existing report paths are not overwritten. Read timings, memory and accuracy directly from a completed report; `status: running` is incomplete. Builds and CTest do not refresh benchmark results.

For comparisons, rebuild the changed source, run without competing builds/solves, and hold inputs, references, iteration budget and worker count constant. Check achieved exploitability alongside timing. Identify the Git revision and any uncommitted changes in the report filename or comparison notes; the report does not record them. Windows and macOS memory metrics have different meanings.

## Updating inputs and references

All range inputs derive from [the captured GTO Wizard catalog](../resources/gtowizard-preflop/). To refresh fixtures and the CLI example after a catalog or subset change:

```sh
python3 scripts/sync-preflop-fixtures.py
```

This script updates inputs only. Regenerate independent answers for changed scenarios with the Rust oracle. Its upstream revision and dependencies are pinned in [oracle/Cargo.toml](oracle/Cargo.toml) and [oracle/Cargo.lock](oracle/Cargo.lock); the adapter is [oracle/src/main.rs](oracle/src/main.rs).

The pinned upstream code needs these lint allowances on current Rust toolchains. Set them for reference generation in your shell:

```sh
# macOS
export RUSTFLAGS='-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
```

```powershell
# Windows
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
```

Then run the required generator; `RAYON_NUM_THREADS` can limit its workers:

```sh
# Correctness references
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Benchmark reference
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

Generation fails before writing if its precision requirement is unmet. Review generated diffs and rerun the affected correctness suite or benchmark. Preserve independent expected values and tolerances; this solver's output must never replace the oracle's answers.
