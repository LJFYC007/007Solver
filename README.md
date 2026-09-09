<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

<p align="center">
  A Windows desktop poker solver with a C++17 engine and a React/Tauri strategy explorer.
</p>

## Overview

007 Solver builds and solves a heads-up postflop decision tree, then lets you inspect individual nodes without loading the whole tree into the UI.

```text
resources/default.json -> C++ solver service <-> Tauri/Rust <-> React UI
```

The application currently:

- runs DCFR or external-sampling CFR (ESCFR) over weighted hero and villain ranges;
- navigates decision, chance, and terminal nodes along the selected action line;
- displays a 13 x 13 hand matrix with mixed-strategy colors;
- reports aggregate action frequencies, combo strategies, and reach weights;
- shows the board, pot, remaining stacks, street, and available runouts.

The C++ service starts with the desktop application, builds the tree, trains a strategy and exports one snapshot. It releases training state before evaluating exploitability. The service keeps the result in memory and answers node queries through newline-delimited JSON. Each node ID identifies one complete action and card history within that tree.

The C++ module boundaries and dependency direction are documented in [`solver/ARCHITECTURE.md`](solver/ARCHITECTURE.md).

## Current Scope

The application automatically loads the single bundled scenario in [`resources/default.json`](resources/default.json). There is not yet an in-app scenario editor or file picker.

The default scenario uses an `Ac Kh Qs` flop, BTN and LJ weighted ranges, a pot of 5, stacks of 20, and 200 DCFR full-player updates. Edit `resources/default.json` before launching to use a different flop scenario. Range entries use pairs and suited/offsuit hand classes such as `"AA"`, `"AKs"`, and `"AKo"`. Each numeric weight is applied to every exact combo in that class. Amounts are in chips with 0.1-chip precision; the scenario does not define a big-blind conversion. The modeled game has no rake, uses a 0.1-chip minimum bet, and starts a fresh betting round on the flop.

The optional `algorithm` field accepts `dcfr` or `escfr` (the default when omitted). Each iteration updates one player: DCFR traverses the full tree; ESCFR samples a traversal. Their iteration counts are not comparable.

Training stops at the configured iteration count. The bundled scenario demonstrates the explorer; its default budget does not guarantee convergence. Exploitability is reported in the service diagnostics, and "Solution ready" means the solve has finished without enforcing an accuracy threshold.

## Requirements

- Windows with Visual Studio 2022 or newer and the **Desktop development with C++** workload
- Microsoft Visual C++ x64 runtime, including `VCOMP140.DLL` (OpenMP)
- CMake 3.20 or newer
- Ninja
- Node.js 22 (22.13+) or 24+ and npm
- Rust stable with the `x86_64-pc-windows-msvc` toolchain
- Microsoft Edge WebView2 Runtime for the desktop application
- Python 3, ClangFormat, and pre-commit for repository checks

Run the native commands below from **Developer PowerShell for Visual Studio**, with `cl`, Ninja, and Cargo on `PATH`.

## Set Up the Repository

Clone the repository and initialize its submodules:

```powershell
git clone --recurse-submodules https://github.com/LJFYC007/007Solver.git
Set-Location 007Solver
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

Install the desktop dependencies:

```powershell
npm --prefix desktop ci
```

## Build the Solver Service

Configure and compile the C++ service:

```powershell
chcp 65001
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
```

The build stages the executable in `desktop/src-tauri/binaries/` under the filename expected by Tauri. To build and stage only the service, run `npm --prefix desktop run build:solver`.

`chcp 65001` lets Ninja parse localized MSVC header dependencies. DCFR uses OpenMP; set `OMP_NUM_THREADS` to limit its worker count.

## Run the Solver Tests

Build and run the GoogleTest executable:

```powershell
cmake --build --preset windows-msvc-release --target 007SolverTests
ctest --test-dir build/windows-msvc-release --output-on-failure
```

The suite checks betting rules and settlement, an independent best-response baseline, two small reference solves, and fixed-policy node EVs and reach weights. Reference answers are checked into the repository; routine tests run offline. See [`tests/README.md`](tests/README.md) for coverage and reference generation.

CTest enforces a 60-second timeout for the suite.

## Repository Checks

After installing dependencies and building the solver service, run the checks relevant to your changes:

```powershell
npm --prefix desktop run check
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
pre-commit run --all-files
```

Pre-commit may fix formatting. Use `pre-commit run --files` with explicit paths to include new files that have not yet been staged.

## Run in Development

Build the solver service first, then start the desktop application:

```powershell
npm --prefix desktop run tauri -- dev
```

The application shows startup progress and opens the strategy explorer after training, exploitability evaluation and the initial node query finish.

## Build the Windows Installer

Build the NSIS installer. This command first configures, builds and stages the C++ service, then builds the frontend and desktop application:

```powershell
npm --prefix desktop run tauri -- build
```

The installer is written to `desktop/src-tauri/target/release/bundle/nsis/`.

## Repository Layout

```text
solver/{core,game,engine}/  Solver semantics and solving; headers and sources are colocated
solver/{analysis,io}/       Result navigation and external-input adapters
solver/service/             JSON/NDJSON service boundary
solver/main.cpp             Command-line entry point
solver/ARCHITECTURE.md      Solver module responsibilities and dependency direction
tests/                      Solver correctness tests and independent reference fixtures
desktop/src/                React and TypeScript strategy explorer
desktop/src-tauri/          Rust bridge and Tauri packaging configuration
resources/                  Bundled scenario and ranges
third_party/                Git submodules
scripts/                    Frontend tooling used by pre-commit
```
