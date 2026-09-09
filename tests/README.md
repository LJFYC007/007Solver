# Solver tests

Run from the repository root in Terminal (macOS) or Developer PowerShell (Windows).

## Default correctness tests

Use the shared build/test commands in the [README](../README.md#run-the-solver-tests).

Seven offline cases cover betting rules, uniform-policy best responses (BR), two reference solves, and fixed-policy node EV/reach. CTest runs one entry, `solver`, with a 60-second timeout and writes `build/solver-test-results.json`. Windows/macOS CI uses this same suite.

The weighted/raise fixtures use 3M/8M ESCFR updates and 200 DCFR updates (1 and 4 workers, including continued runs), require exploitability <= `0.01`, and compare value intervals with external GT to `1e-5`. Fixed-policy expectations are stored in `fixtures/correctness-reference.json`. Performance measurement is handled by the separate benchmark.

## Explicit wide-range benchmark

```sh
cmake --preset release
cmake --build --preset release --target 007SolverBenchmark
./build/007SolverBenchmark
# Optional: choose a new report path; existing reports are never overwritten.
./build/007SolverBenchmark --report=build/benchmark-results/sample.json
# DCFR requires an explicit iteration budget.
./build/007SolverBenchmark --algorithm=dcfr --iterations=48 --workers=8
```

On Windows the executable has an `.exe` suffix. Run the benchmark separately from other builds/solves when comparing timings.

The benchmark is excluded from the default build, CTest, CI and pre-commit execution. It uses `fixtures/utg-bb-wide.json`: UTG/IP versus BB/OOP, flop `Ac Kh Qs`, pot 5, stacks 2.5 each, no rake, default sizing and **200M ESCFR sampled updates by default**, retaining complete ranges and turn/river play.

`--algorithm` and `--iterations` select the solver and budget; DCFR requires an explicit budget, and `--workers` applies only to DCFR. Compare time at equal exploitability, since iteration counts measure different work.

It checks exact input/GT equality, uniform BRs and exploitability within `1e-5`, finite trained metrics, value-interval compatibility, the BR-average identity and exploitability in `[-1e-5, 0.025]`. Root EVs, reach and probabilities must be valid; equilibrium action frequencies and per-hand EVs are not compared.

Unique JSON reports default to `build/benchmark-results/`. They include algorithm, worker count, iterations and unit, build/workload information, phase timings, metrics and process memory. Windows reports private committed bytes and peak working set; macOS reports physical footprint and peak resident bytes. These OS metrics are not interchangeable. Training state is released before trained evaluation. Failures retain diagnostics and return nonzero; `status: running` reports are incomplete.

The local Release performance target is under five minutes; timing is recorded, not asserted. Local JSON reports are measurements, not replacement reference answers.

Seed 42 does not guarantee identical ESCFR sampled strategies across standard libraries; retain the independent reference tolerances on every platform.

Rebuild and run after changing source; CTest and normal builds do not refresh benchmark reports. Check `status` before using the numbers. Reports are local, ignored artifacts and record compiler/configuration but not Git revision or dirty state. Use a unique `--report` filename containing the commit ID and timestamp, and note any uncommitted changes separately when comparing implementations.

## Independent GT generation

`fixtures/correctness-reference.json` and `fixtures/benchmark-reference.json` use [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver) at commit `9d1509fe5077d019825f833eed04b16d342dfda1`, with dependencies pinned by `oracle/Cargo.lock`.

On macOS, set `export RUSTFLAGS='-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'` instead of the PowerShell assignment below; the Cargo commands are otherwise identical.

```powershell
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
# Original correctness references only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
# Wide benchmark reference only:
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle -- --wide
```

Generation requires exploitability <= `1e-5` within 10,000 external iterations; the wide entry fails before writing GT if precision is unmet. GT generation is outside the benchmark time budget. Review generated diffs and rerun the relevant executable; never substitute this solver's output for independent answers or loosen tolerances to pass.

The adapter scales chips/EVs by 500 and maps external OOP/IP to villain/hero. It uses half-pot/maximum bets, maximum raises, and no rake, automatic all-in thresholds or size merging. Current fixture sizes avoid rounding differences. Root BRs use net payoff minus initial pot/2; their average is exploitability. Node EVs use the queried node's contribution baseline. All-in runouts remain complete, with conditional chance denominators 45/44.

### Wide-range source

The fixture transcribes [RangeConverter's 9-max / 100BB Live Cash PDF](https://rangeconverter.com/downloads/9-max-100bb-Poker-Charts-No-Limit-Texas-Holdem-Cash), accessed 2026-09-05: page 3 upper-left UTG RFI opens to **3BB**; page 10 upper-left BB vs UTG contributes only green **Call**, excluding orange **13BB 3-bets**. Page 2 specifies simplified 50% increments: solid opening/calling cells become `1.0`, mixed cells `0.5`, absent cells zero, applied equally to each exact combo.

All positive cells are retained: UTG 28 classes/131 weighted combos, BB 45/178 before blockers. Weights follow the simplified cells, whose totals differ slightly from the graphic footers (10.09%/13.51%); no rescaling is applied. The source does not specify rake. These ranges are fixed inputs to reduced postflop stacks, not a full 100BB equilibrium.

PDF SHA-256: `0feb70db01ab74db6d6a8cb6e9761358f6eb04879468452f8429e027c6770b27`.
