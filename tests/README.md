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
./build/007SolverBenchmark --iterations=3000 --workers=12
```

On Windows the executable has an `.exe` suffix. Run the benchmark separately from other builds/solves when comparing timings.

The benchmark is excluded from the default build, CTest, CI and pre-commit execution. It uses `fixtures/utg-bb-wide.json`: UTG/IP versus BB/OOP, flop `Ac Kh Qs`, pot 5, stacks 15 each (stack-to-pot ratio 3), no rake, half-pot/pot-sized/all-in bets, all-in raises and **3000 DCFR full-player updates by default**, retaining complete ranges and turn/river play. The benchmark harness and external oracle both read the fixture's `benchmark.betPercentages` list (`[50, 100]`) and append maximum bets/raises. This configuration produces 922,164 nodes, retaining distinct action and concrete-card histories.

`--iterations` overrides the fixture budget; `--workers` selects the OpenMP training worker count (otherwise the runtime default, configurable with `OMP_NUM_THREADS`). Snapshot evaluation and analysis are serial. The optional `--algorithm=dcfr` spelling is accepted; no other algorithm is supported. Compare time at equal exploitability and record the worker count.

It checks exact input/GT equality, uniform BRs and exploitability within `1e-5`, finite trained metrics, value-interval compatibility, the BR-average identity and exploitability in `[-1e-5, 0.0005]` (at most 0.01% of the initial pot). Root EVs, reach and probabilities must be valid; equilibrium action frequencies and per-hand EVs are not compared.

Unique JSON reports default to `build/benchmark-results/`. They include algorithm, worker count, iterations and unit, the exploitability limit, build/workload information, phase timings, metrics and process memory. Windows reports private committed bytes and peak working set; macOS reports physical footprint and peak resident bytes. These OS metrics are not interchangeable. Training state is released before trained evaluation. Failures retain diagnostics and return nonzero; `status: running` reports are incomplete.

The local Release performance target is **within about five minutes** for the complete benchmark, including reference checks and root analysis, on the documented machine with 12 training workers. Timing is recorded, not asserted: faster implementations should finish sooner, and other hardware can differ. Local JSON reports are measurements, not replacement reference answers. The older stacks-2.5 and stacks-7.5 workloads are not directly comparable to this larger tree.

The GTO Wizard migration changes the benchmark ranges, so earlier RangeConverter timings and reference values do not describe this fixture. Compare only runs with identical scenario inputs and independent references.

On the local Apple M4 Pro (Release, AppleClang 21, 12 workers), the sourced 3-bet fixture passed at 3000 updates in **256.50 seconds**: training 252.20s, trained evaluation 1.31s, root query 0.88s, exploitability **0.000263937**. It contains 100/174 legal hero/villain hands and 15,523 compatible hand pairs. The report is `build/benchmark-results/gtowizard-2b088a1-dirty-20260910-001323.json` (uncommitted migration changes). The 1900-update calibration took 162.48s but missed the unchanged precision limit at 0.000674300; 3000 is the calibrated default.

Rebuild and run after changing source; CTest and normal builds do not refresh benchmark reports. Check `status` before using the numbers. Reports are local, ignored artifacts and record compiler/configuration but not Git revision or dirty state. Use a unique `--report` filename containing the commit ID and timestamp, and note any uncommitted changes separately when comparing implementations.

## Independent GT generation

`fixtures/correctness-reference.json` and `fixtures/benchmark-reference.json` use [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver) at commit `9d1509fe5077d019825f833eed04b16d342dfda1`, with dependencies pinned by `oracle/Cargo.lock`. The oracle enables its existing Rayon support; set `RAYON_NUM_THREADS` to limit reference-generation workers.

On macOS, set `export RUSTFLAGS='-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'` instead of the PowerShell assignment below; the Cargo commands are otherwise identical.

```powershell
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
# Correctness references only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Wide benchmark reference only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

Generation requires exploitability <= `1e-5` within 100,000 external iterations (each updates both players). Generation fails before writing GT if precision is unmet. GT generation is outside the benchmark time budget. Review generated diffs and rerun the relevant executable; never substitute this solver's output for independent answers or loosen tolerances to pass.

The regenerated correctness references reached `9.91821e-6` chips for weighted-flop and `8.78906e-6` for raise-flop. The 3-bet benchmark reference reached `9.97925e-6` after 18,860 external iterations. The Release correctness suite passed in 15.98 seconds on the local Apple M4 Pro; this is a measurement, not a timing guarantee.

The oracle enables JSON float round-tripping so copied scenario weights retain their original `f64` values. The first 3-bet reference required correcting two copied weights by one `f64` rounding step; both had identical `f32` values in the external range parser, and all computed reference metrics were preserved.

The adapter scales chips/EVs by 500 and maps external OOP/IP to villain/hero. Correctness fixtures use half-pot/maximum bets; the wide fixture uses half-pot/pot-sized/maximum bets. Both use maximum raises and no rake, automatic all-in thresholds or size merging. Current fixture sizes avoid rounding differences. Root BRs use net payoff minus initial pot/2; their average is exploitability. Node EVs use the queried node's contribution baseline. All-in runouts remain complete, with conditional chance denominators 45/44.

### GTO Wizard range source

All range-containing inputs, including the CLI example, derive from [`resources/gtowizard-preflop/`](../resources/gtowizard-preflop/). This is the captured 6-max/8-max, chip-EV, 100bb source used by the desktop. Every scenario records the catalog SHA-256, solution, action history and any retained hand classes in `rangeSource`. There are no legacy strategy-range fixtures.

The 2026-09-10 and 2026-09-11 catalog expansions from 77 to 208 matrices leaves all three numerical test scenarios unchanged. Only their catalog SHA-256 provenance was refreshed, including the copied scenario metadata in the reference files; independent expected values and the benchmark measurement above were retained.

The directory migration likewise preserves all 208 source nodes, numerical scenario inputs and independent expected values. Catalog schema 2 stores shared metadata in `catalog.json` and nodes in `6max/<actor>.json` or `8max/<actor>.json`. Its SHA-256 covers all catalog JSON files in sorted relative-path order: UTF-8 POSIX relative path, a NUL byte, exact file bytes, then another NUL byte for each file. The directory path and this digest replace the former single-file provenance in scenarios and copied reference metadata.

- `weighted-flop`: 8-max UTG open, BB 3-bet to 13, UTG 4-bet to 23.5, BB call; retain UTG AKs/QQ and BB KK/A5s.
- `raise-flop`: 6-max UTG open, BB 3-bet to 12.5, UTG 4-bet to 26.5, BB call; retain UTG KJs and BB AQs.
- `utg-bb-wide`: 8-max UTG opens to 2.5, other seats fold, BB 3-bets to 13, UTG calls; retain all 24 UTG and 45 BB classes, including very small positive source frequencies. Before blockers these contain 50.530716 and 47.433703 weighted combos respectively. The source line has pot 26.5bb and stacks 87bb; the benchmark reduces these to pot 5 and stacks 15 for runtime (SPR 3).

Weights are products of the acting player's captured conditional action frequencies. Displayed rows are normalized for 0.01 percentage-point rounding, and absent continuation cells are omitted. Test pot/stacks are reduced explicitly for runtime; these are postflop test games with sourced inputs, not full 100bb equilibrium claims. The engine applies the original exact suits and board blockers.

To refresh inputs after a reviewed catalog change:

```sh
python3 scripts/sync-preflop-fixtures.py
```

Then regenerate both independent reference files with the commands above and rerun the correctness suite and benchmark. The input generator never produces expected solver values. The fixed-policy oracle intentionally prescribes mixed and zero actions over the sourced hands to exercise own/joint reach and node EV; that policy is an analysis reference, not a replacement preflop strategy.
