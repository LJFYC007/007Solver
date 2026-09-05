# Solver correctness tests

Build `007SolverTests` with the Release preset, then run:

```powershell
ctest --test-dir build/windows-msvc-release --output-on-failure
```

The seven cases run as one CTest entry with a 60-second timeout. GoogleTest filters can select individual cases. No external solver, network, or Rust build is needed for routine tests.

[GitHub Actions](../.github/workflows/tests.yml) builds and runs this suite in Windows/MSVC Release on pushes and pull requests, with an optional manual run. CI uses the checked-in fixtures and the same 60-second CTest timeout.

| Check | Independent responsibility |
|---|---|
| Three betting-rule cases | Full/short raises, unequal effective-stack caps, and sizing bases absent from the small solve fixtures |
| Uniform-policy best responses | Calibrate exploitability and terminal payoffs on a deliberately exploitable strategy, without training |
| Weighted flop, 3 million iterations | Detect training that ignores range weights |
| Raise flop, 8 million iterations | Detect training that omits a third action or raise |
| Fixed-policy analysis | Concrete board/hand coordinates, node EV baseline, exact conditional reach, own reach and zero-reach behavior |

Both solves start on `Ks 9s 2d` and include turn/river play. The weighted fixture has two actions per decision; the raise fixture also has three-action decisions. They check that the solver's value interval `[-villain BR, hero BR]` intersects the external interval, within `1e-5` rounding tolerance, and that exploitability is at most `0.01` (0.5% of the initial pot). Equilibrium action frequencies are not unique and are not compared.

Analysis uses prescribed strategies, not approximate equilibria. On the weighted flop, OOP checks AQ 25% and QJ 100%; after a check, IP checks KQ 0% and JT 100%. All remaining decisions are uniform. Queries cover blockers, zero reach, concrete turn/river cards `Qc` / `Th`, and a bet/call followed by turn decisions in the raise fixture. Node EV tolerance is `1e-5`; reach uses relative tolerance so small river masses cannot hide a wrong chance denominator.

## Independent reference generation

`fixtures/reference.json` is generated solely by [b-inary/postflop-solver](https://github.com/b-inary/postflop-solver), pinned to commit `9d1509fe5077d019825f833eed04b16d342dfda1`. `oracle/Cargo.lock` pins its dependency graph. The generated file embeds the exact scenario inputs, which tests compare against the scenario files to prevent stale answers.

The adapter multiplies chips by 500 and divides output EVs by 500. It uses 50% pot and maximum bets, maximum raises, no rake, no automatic all-in thresholds and no size merging. These fixtures have no rounding ambiguities. External OOP/IP maps to villain/hero. Root BR values use zero-sum utilities (initial pot / 2 subtracted); node EVs use the queried node as their contribution baseline. The external flop sorting is converted back to the user's original order.

External solves stop at exploitability <= `1e-5` in fixture units, with a 10000-iteration cap; generation fails if this precision is not reached. Fixed-policy EVs are evaluated without training. External normalized hand weights are divided by the initial legal joint range weight and by 45/44 for dealt turn/river cards to obtain this application's marginal reach mass. Paths use zero-based action indices and concrete card strings; `policy` records the only nonuniform strategy entries.

Regenerate in a separate Developer PowerShell session:

```powershell
# Compatibility with newer Rust's deny-by-default lint in this pinned older dependency.
$env:RUSTFLAGS = '-A dangerous_implicit_autorefs -A mismatched_lifetime_syntaxes'
cargo run --locked --release --manifest-path tests/oracle/Cargo.toml --target-dir build/solver-test-oracle
```

Review the generated diff and rerun the C++ suite. Never copy this solver's observed values into the reference file or loosen precision merely to make a regression pass.

The suite covers the listed rule boundaries, training fixtures and fixed-policy analysis. It does not cover all possible game configurations, desktop navigation or service-process behavior.
