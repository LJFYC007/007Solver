# Solver tests

Run commands from the repository root. Use Terminal on macOS or Developer PowerShell on Windows; Windows executables have an `.exe` suffix.

## Correctness

Use the [build and test commands](../README.md#checks). CTest writes `build/release/solver-test-results.json`; routine tests use checked-in references and run offline.

Coverage lives in [domain_tests.cpp](domain_tests.cpp) and [tests.cpp](tests.cpp); inputs and independent answers live in [fixtures/](fixtures/). GPU cases skip when no supported device is available. Check the recorded device and skipped cases in the JSON report or `ctest -V`; passing CUDA checks does not validate Metal.

## Benchmark

The CPU and GPU benchmarks are separate opt-in commands. Each builds the same executable and writes its own timestamped report under `build/benchmark-results/`:

```sh
cmake --preset Release
cmake --build --preset Release --target benchmark_cpu
cmake --build --preset Release --target benchmark_gpu
```

The GPU command selects CUDA or Metal on a supported machine and fails if no GPU backend is available.

For Visual Studio CPU sampling with symbols, use the separate RelWithDebInfo build:

```sh
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo --target 007SolverBenchmark
./build/relwithdebinfo/007SolverBenchmark --device=cpu
```

Direct invocation accepts `--report=<new-json-path>`, `--iterations=<count>`, `--workers=<count>` and `--device=cpu|gpu|auto`; it defaults to CPU, and `--workers` affects only CPU training. Workload and report definitions are in [benchmark.cpp](benchmark.cpp).

`--convergence` records periodic checkpoints; `--stop-at-accuracy` also permits early stopping. GPU stopping and final checkpoints receive CPU certification, and the exported snapshot is independently evaluated. Checkpoint time is separate from pure training time.

Reports do not overwrite existing paths. Read results under [build/benchmark-results/](../build/benchmark-results/) only after completion; `status: running` is incomplete. Builds and CTest do not refresh them.

Compare runs without competing builds/solves, holding inputs, references, update budget, checkpoint mode and CPU workers constant. Compare achieved exploitability alongside time, and record the Git revision and uncommitted changes externally. Process memory metrics differ across platforms and exclude dedicated GPU allocations; the solver estimate covers combined host/device allocations.

## Updating inputs and references

All range inputs derive from [the captured GTO Wizard catalog](../resources/gtowizard-preflop/). To refresh fixtures after a catalog or subset change:

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
