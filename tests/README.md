# Solver tests

Run commands from the repository root using the [build environment](../README.md#requirements).

## Correctness

The [correctness suite](../README.md#checks) runs offline against checked-in references and writes `build/release/solver-test-results.json`. CTest also runs [CLI end-to-end checks](cli_e2e.py) for solving, navigation, queries and errors, using four CPU workers.

GPU cases skip without a supported CUDA device. Check the device and skipped cases in the JSON report or `ctest -V`.

The backend parity [input](fixtures/backend-parity.json) has no independent answer; the CPU backend is its reference. It must keep splitting street regions into several GPU batches.

## Benchmark

Benchmarks are opt-in; normal builds and CTest do not run them:

```sh
cmake --preset Release
cmake --build --preset Release --target benchmark_cpu
cmake --build --preset Release --target benchmark_gpu
```

`benchmark_gpu` fails without a supported CUDA device. Timestamped reports go to `build/benchmark-results/`; `status: running` is incomplete, and existing paths cannot be overwritten.

For Visual Studio CPU sampling with symbols, use the separate RelWithDebInfo build:

```sh
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo --target 007SolverBenchmark
./build/relwithdebinfo/007SolverBenchmark.exe --device=cpu
```

Direct invocation accepts `--report=<new-json-path>`, `--iterations=<count>`, `--workers=<count>` and `--device=cpu|gpu|auto`. It defaults to CPU; `--workers` affects only CPU training. See [benchmark.cpp](benchmark.cpp) for workload and report definitions.

The [workload](fixtures/utg-bb-wide.json) uses full 6-max GG R&C UTG open / BB call single-raised-pot ranges with 5% rake capped at 3bb, pot 5.5bb, stacks 97.5bb, three-street bets of 33%/125%, raises of 50%, and 1000 player updates. `maxRaises: 2` and `allInSpr: 0` preserve larger branches. Its ranges hold 254 and 287 hands on its flop (the report's `legal_hands`); GPU kernels take other paths and batch differently for wider ranges, so also time GPU performance changes on a scenario with wider catalog ranges, such as both players' full ranges from the catalog's root node.

Passing checks the independent uniform reference (`5e-6` initial-pot tolerance), training improvement and root queries; it **does not require convergence**. Reports include the tree size, updates per second, achieved accuracy and `target_reached`.

`--convergence` records periodic checkpoints; `--stop-at-accuracy` also permits early stopping under the [CPU certification contract](../solver/ARCHITECTURE.md#training-and-memory). The exported snapshot is independently evaluated. Checkpoint time is separate from training time.

Compare runs without competing builds/solves, holding inputs, references, update budget, checkpoint mode and CPU workers constant. Compare achieved exploitability alongside time, and record the Git revision and uncommitted changes externally. Process memory metrics exclude dedicated GPU allocations; the solver estimate covers combined host/device allocations.

## Updating inputs and references

The four fixtures derive from [captured cases](../resources/gtowizard-preflop/cases/) the way the desktop replays preflop lines. The benchmark, backend parity and weighted-flop inputs use the 6-max Simple GG R&C case (5% rake, 3bb cap); raise-flop uses the rake-free cEV case. Each input's `rangeSource` records its history, fixture subsets, pot/stack reductions and the sha256 of the saved nodes it reads. Edit scenario definitions in [sync-preflop-fixtures.py](../scripts/sync-preflop-fixtures.py), then regenerate inputs:

```sh
python scripts/sync-preflop-fixtures.py
```

The script updates inputs only. Regenerate answers with the independent [Rust oracle](oracle/src/main.rs), pinned by [Cargo.toml](oracle/Cargo.toml) and [Cargo.lock](oracle/Cargo.lock).

The pinned upstream code needs these Rust lint allowances for reference generation:

```powershell
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
```

Run the required generator; `RAYON_NUM_THREADS` can limit its workers:

```sh
# Correctness references
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Benchmark uniform-policy reference (no convergence solve)
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

Correctness references require oracle convergence. The benchmark uses uniform-policy evaluation at chip scale 10 and `allInSpr: 0` to match rounding across engines. Review diffs and rerun affected checks; never replace independent answers with this solver's output.
