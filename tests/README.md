# Solver tests

Run commands from the repository root using the [build environment](../README.md#requirements). Append `.exe` to Windows executable paths.

## Correctness

The [correctness suite](../README.md#checks) runs offline against checked-in references and writes `build/release/solver-test-results.json`.

GPU cases skip without a supported device. Check the device and skipped cases in the JSON report or `ctest -V`; passing CUDA checks does not validate Metal.

## Benchmark

Benchmarks are opt-in; normal builds and CTest do not run them:

```sh
cmake --preset Release
cmake --build --preset Release --target benchmark_cpu
cmake --build --preset Release --target benchmark_gpu
```

`benchmark_gpu` fails without a supported CUDA or Metal device. Timestamped reports go to `build/benchmark-results/`; `status: running` is incomplete, and existing paths cannot be overwritten.

For Visual Studio CPU sampling with symbols, use the separate RelWithDebInfo build:

```sh
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo --target 007SolverBenchmark
./build/relwithdebinfo/007SolverBenchmark --device=cpu
```

Direct invocation accepts `--report=<new-json-path>`, `--iterations=<count>`, `--workers=<count>` and `--device=cpu|gpu|auto`. It defaults to CPU; `--workers` affects only CPU training. See [benchmark.cpp](benchmark.cpp) for workload and report definitions.

`--convergence` records periodic checkpoints; `--stop-at-accuracy` also permits early stopping under the [CPU certification contract](../solver/ARCHITECTURE.md#training-and-memory). The exported snapshot is independently evaluated. Checkpoint time is separate from training time.

Compare runs without competing builds/solves, holding inputs, references, update budget, checkpoint mode and CPU workers constant. Compare achieved exploitability alongside time, and record the Git revision and uncommitted changes externally. Process memory metrics differ across platforms and exclude dedicated GPU allocations; the solver estimate covers combined host/device allocations.

## Updating inputs and references

All ranges derive from [the captured catalog](../resources/gtowizard-preflop/); fixture subsets and reduced pot/stacks are recorded in each input's `rangeSource`. After a catalog or subset change:

```sh
python3 scripts/sync-preflop-fixtures.py
```

The script updates inputs only. Regenerate answers with the independent [Rust oracle](oracle/src/main.rs), pinned by [Cargo.toml](oracle/Cargo.toml) and [Cargo.lock](oracle/Cargo.lock).

The pinned upstream code needs these Rust lint allowances for reference generation:

```sh
# macOS
export RUSTFLAGS='-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
```

```powershell
# Windows
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
```

Run the required generator; `RAYON_NUM_THREADS` can limit its workers:

```sh
# Correctness references
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Benchmark reference
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

The oracle writes only after meeting its precision requirement. Review diffs and rerun the affected suite or benchmark; preserve independent expected values and tolerances, never replacing them with this solver's output.
