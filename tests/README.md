# Solver tests

Run from the repository root in Terminal (macOS) or Developer PowerShell (Windows).

## Default correctness tests

Use the shared build/test commands in the [README](../README.md#run-the-solver-tests).

Seven offline cases cover betting rules, uniform-policy best responses (BR), two reference solves, and fixed-policy node EV/reach. CTest runs one entry, `solver`, with a 60-second timeout and writes `build/solver-test-results.json`. Windows/macOS CI uses this same suite.

The weighted/raise fixtures use pot 2 with stacks 4/8 respectively and 200 DCFR full-player updates (1 and 4 workers, including continued runs), require exploitability <= `0.01`, and compare value intervals with external GT to `1e-5`. Fixed-policy expectations are stored in `fixtures/correctness-reference.json`. The local Release target is under 20 seconds for the complete correctness suite; the 60-second CTest timeout allows for slower CI machines. Larger-workload performance measurement is handled by the separate benchmark.

## Explicit wide-range benchmark

```sh
cmake --preset release
cmake --build --preset release --target 007SolverBenchmark
./build/007SolverBenchmark
# Optional: choose a new report path; existing reports are never overwritten.
./build/007SolverBenchmark --report=build/benchmark-results/sample.json
# Optional: override the fixture's DCFR budget and OpenMP worker count.
./build/007SolverBenchmark --iterations=1900 --workers=12
```

On Windows the executable has an `.exe` suffix. Run the benchmark separately from other builds/solves when comparing timings.

The benchmark is excluded from the default build, CTest, CI and pre-commit execution. It uses `fixtures/utg-bb-wide.json`: UTG/IP versus BB/OOP, flop `Ac Kh Qs`, pot 5, stacks 20 each (stack-to-pot ratio 4), no rake, half-pot/pot-sized/all-in bets, all-in raises and **1900 DCFR full-player updates by default**, retaining complete ranges and turn/river play. The benchmark harness and external oracle both read the fixture's `benchmark.betPercentages` list (`[50, 100]`) and append maximum bets/raises. This benchmark-only configuration produces 1,035,060 nodes, retaining distinct action and concrete-card histories.

`--iterations` overrides the fixture budget; `--workers` selects the OpenMP training worker count (otherwise the runtime default, configurable with `OMP_NUM_THREADS`). Snapshot evaluation and analysis are serial. The optional `--algorithm=dcfr` spelling is accepted; no other algorithm is supported. Compare time at equal exploitability and record the worker count.

It checks exact input/GT equality, uniform BRs and exploitability within `1e-5`, finite trained metrics, value-interval compatibility, the BR-average identity and exploitability in `[-1e-5, 0.0005]` (at most 0.01% of the initial pot). Root EVs, reach and probabilities must be valid; equilibrium action frequencies and per-hand EVs are not compared.

Unique JSON reports default to `build/benchmark-results/`. They include algorithm, worker count, iterations and unit, the exploitability limit, build/workload information, phase timings, metrics and process memory. Windows reports private committed bytes and peak working set; macOS reports physical footprint and peak resident bytes. These OS metrics are not interchangeable. Training state is released before trained evaluation. Failures retain diagnostics and return nonzero; `status: running` reports are incomplete.

The local Release performance target is **two to five minutes** for the complete benchmark, including reference checks and root analysis, on the documented machine with 12 training workers. Timing is recorded, not asserted: faster implementations should finish sooner, and other hardware can differ. Local JSON reports are measurements, not replacement reference answers. The older stacks-2.5 and stacks-7.5 workloads are not directly comparable to this larger tree.

Local measurements on 2026-09-09 used the same Release build configuration on an Apple M4 Pro with 24 GiB RAM: 1,035,060 nodes, 1,900 full-player updates and 12 workers. Times below are seconds.

| Implementation | Total | Training | Uniform evaluation | Snapshot export | Trained evaluation | Root query |
|---|---:|---:|---:|---:|---:|---:|
| Initial shared traversal baseline | 287.31 | 281.16 | 1.60 | 0.90 | 1.95 | 1.29 |
| Zero-tie showdown and terminal own-reach fast paths | 225.34 | 219.79 | 1.20 | 1.04 | 1.71 | 1.12 |
| Action-major backups/regret updates and dynamic terminal batches | **186.96** | **181.49** | 1.17 | 1.02 | 1.67 | 1.08 |

The latest full run passed in **3 minutes 7 seconds**, reducing total time by **17.0%** from the fast-path baseline and **34.9%** from the initial baseline. Training time fell another **17.4%**. Uniform/trained metrics and the complete root report matched the preceding baseline exactly, with the same fixture, iteration budget and independent reference. Final exploitability remained **0.00036353 chips (0.00727% of the initial pot)**, below the 0.0005-chip limit; peak resident memory remained about 6.61 GiB. The latest local reports are `action-major-20260909-204547.json` and `action-major-20260909-204547-comparison.json` under `build/benchmark-results/`. These are single-run measurements, not timing guarantees.

Rebuild and run after changing source; CTest and normal builds do not refresh benchmark reports. Check `status` before using the numbers. Reports are local, ignored artifacts and record compiler/configuration but not Git revision or dirty state. Use a unique `--report` filename containing the commit ID and timestamp, and note any uncommitted changes separately when comparing implementations.

## Independent GT generation

`fixtures/correctness-reference.json` and `fixtures/benchmark-reference.json` use [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver) at commit `9d1509fe5077d019825f833eed04b16d342dfda1`, with dependencies pinned by `oracle/Cargo.lock`. The oracle enables its existing Rayon support; set `RAYON_NUM_THREADS` to limit reference-generation workers.

On macOS, set `export RUSTFLAGS='-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'` instead of the PowerShell assignment below; the Cargo commands are otherwise identical.

```powershell
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
# Original correctness references only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Wide benchmark reference only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

Generation requires exploitability <= `1e-5` within 10,000 external iterations (each updates both players); the wide entry fails before writing GT if precision is unmet. GT generation is outside the benchmark time budget. Review generated diffs and rerun the relevant executable; never substitute this solver's output for independent answers or loosen tolerances to pass.

The 2026-09-09 wide reference reached `9.94873e-6` chips after 5,680 external iterations, using the pinned solver and 8 Rayon workers. The benchmark iteration budget is metadata and does not affect this independent solve; its recorded budget was updated after timing calibration without changing the generated reference values. The smaller correctness reference was not regenerated.

The adapter scales chips/EVs by 500 and maps external OOP/IP to villain/hero. Correctness fixtures use half-pot/maximum bets; the wide fixture uses half-pot/pot-sized/maximum bets. Both use maximum raises and no rake, automatic all-in thresholds or size merging. Current fixture sizes avoid rounding differences. Root BRs use net payoff minus initial pot/2; their average is exploitability. Node EVs use the queried node's contribution baseline. All-in runouts remain complete, with conditional chance denominators 45/44.

### Wide-range source

The fixture transcribes [RangeConverter's 9-max / 100BB Live Cash PDF](https://rangeconverter.com/downloads/9-max-100bb-Poker-Charts-No-Limit-Texas-Holdem-Cash), accessed 2026-09-05: page 3 upper-left UTG RFI opens to **3BB**; page 10 upper-left BB vs UTG contributes only green **Call**, excluding orange **13BB 3-bets**. Page 2 specifies simplified 50% increments: solid opening/calling cells become `1.0`, mixed cells `0.5`, absent cells zero, applied equally to each exact combo.

All positive cells are retained: UTG 28 classes/131 weighted combos, BB 45/178 before blockers. Weights follow the simplified cells, whose totals differ slightly from the graphic footers (10.09%/13.51%); no rescaling is applied. The source does not specify rake. These ranges are fixed inputs to reduced postflop stacks, not a full 100BB equilibrium.

PDF SHA-256: `0feb70db01ab74db6d6a8cb6e9761358f6eb04879468452f8429e027c6770b27`.
