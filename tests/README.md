# Solver tests

Run commands from the repository root in the [build environment](../README.md#requirements).

## Correctness

The [correctness suite](../README.md#checks), including [CLI end-to-end checks](cli_e2e.py), runs offline against checked-in references and writes `build/release/solver-test-results.json`. GPU cases skip without a supported CUDA device; the JSON report and `ctest -V` show the device and skipped cases.

The [backend parity input](fixtures/backend-parity.json) has no independent answer: the CPU backend is its reference. It must keep splitting street regions into several GPU batches. The GPU benchmark repeats its comparison on the [wide parity input](fixtures/wide-parity.json), whose full ranges must keep taking the GPU's wide-range kernel paths (256-thread Terminal groups, several hand tiles per Reach or Backup item).

## Benchmark

Benchmarks are opt-in; normal builds and CTest do not run them:

```sh
cmake --preset Release
cmake --build --preset Release --target benchmark_cpu
cmake --build --preset Release --target benchmark_gpu
```

Reports go to `build/benchmark-results/`; `status: running` marks an incomplete one, and `skipped` a GPU run without a supported CUDA device. For Visual Studio CPU sampling with symbols, use the RelWithDebInfo build:

```sh
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo --target 007SolverBenchmark
./build/relwithdebinfo/007SolverBenchmark.exe --device=cpu
```

[benchmark.cpp](benchmark.cpp) defines the flags, workload and report; [CMakeLists.txt](../CMakeLists.txt) raises `benchmark_gpu` to 2000 updates, which must finish within two minutes, while `benchmark_cpu` keeps the workload's 1000 and takes several minutes. Checks that need longer than [CTest's budget](../README.md#checks) belong here. A run passes on GPU parity over full ranges (GPU runs only), on the independent uniform reference and training improvement, both evaluated on the training device, on CPU evaluation of the export agreeing with that device, and on root queries; it **does not require convergence**, so compare achieved exploitability alongside time. Process memory metrics exclude dedicated GPU allocations.

The [workload](fixtures/utg-bb-wide.json) ranges hold only 254 and 287 hands on its flop, and GPU kernels take other paths and batch differently for wider ranges, so also time GPU performance changes on wider catalog ranges, such as both players' full root ranges.

To compare runs, avoid competing builds and solves, hold inputs, update budget, checkpoint mode and CPU workers constant, and record the Git revision and uncommitted changes.

## Updating inputs and references

The five inputs derive from [captured cases](../resources/gtowizard-preflop/cases/) the way the desktop replays preflop lines; each input's `rangeSource` records its history, subsets, pot/stack reductions and the sha256 of the saved nodes it reads.

1. Edit scenario definitions in [sync-preflop-fixtures.py](../scripts/sync-preflop-fixtures.py) and regenerate the inputs with `python scripts/sync-preflop-fixtures.py`. The script does not touch answers.
2. Regenerate the affected answers with the independent [Rust oracle](oracle/src/main.rs), pinned by [Cargo.toml](oracle/Cargo.toml) and [Cargo.lock](oracle/Cargo.lock). The pinned upstream code needs the lint allowances below; `RAYON_NUM_THREADS` can limit its workers.

   ```powershell
   $env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
   # Correctness references (require oracle convergence)
   cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
   # Benchmark reference: uniform-policy evaluation, no convergence solve; chip scale 10 and
   # allInSpr: 0 match rounding across engines
   cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
   ```

3. Review the diffs and rerun the affected checks. Never replace independent answers with this solver's output.
