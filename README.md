<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

<p align="center">
  A Windows/macOS desktop poker solver with a C++17 engine and a React/Tauri strategy explorer.
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

- macOS: Xcode Command Line Tools (`xcode-select --install`) and a libomp (OpenMP) runtime compatible with your OS; see the package requirements below
- Windows: Visual Studio 2022 or newer with **Desktop development with C++**, plus Microsoft Edge WebView2 Runtime
- Windows: Microsoft Visual C++ x64 runtime, including `VCOMP140.DLL` (OpenMP)
- CMake 3.20 or newer
- Ninja
- Node.js 22 (22.13+) or 24+ and npm
- [Rust stable via rustup](https://rust-lang.org/tools/install/), including rustfmt
- Python 3, ClangFormat, and pre-commit for repository checks

Use Terminal on macOS or **Developer PowerShell for Visual Studio** on Windows. CMake, Ninja, the native C++ compiler and Cargo must be on `PATH`. Builds are native: Apple Silicon or Intel on macOS, x64 MSVC on Windows. Universal macOS bundles and cross-compilation are not configured. The solver uses the CPU; no NVIDIA GPU or CUDA installation is needed.

On macOS with Homebrew, install CMake, Ninja and OpenMP with `brew install cmake ninja libomp`. Rustup installs the native Rust toolchain; start a new terminal after installation. Tauri is a project dependency installed by npm below, so no global Tauri installation is needed.

## Set Up the Repository

Clone the repository and initialize its submodules:

```sh
git clone --recurse-submodules https://github.com/LJFYC007/007Solver.git
cd 007Solver
```

For an existing clone:

```sh
git submodule update --init --recursive
```

Install the desktop dependencies:

```sh
npm --prefix desktop ci
```

## Develop and Package

From the repository root, the same commands work on Windows and macOS:

| Command | Result |
|---|---|
| `npm --prefix desktop run dev` | Build/stage the solver, start Vite, and open the desktop app |
| `npm --prefix desktop run build` | Build/stage the solver, compile the UI and Rust app, and package it |
| `npm --prefix desktop run build:solver` | Build/stage only the C++ service |

There is one Release/Ninja preset and one C++ output directory, `build/`. CMake stages the native service in `desktop/src-tauri/binaries/` with Tauri's target suffix. Both development and packaging run this step automatically. After changing C++ code during development, restart `npm --prefix desktop run dev` to rebuild and re-solve. React edits reload through Vite.

The app shows startup progress and opens the explorer after training, exploitability evaluation and the initial node query finish. It solves `resources/default.json` once per launch.

Packages appear under `desktop/src-tauri/target/release/bundle/`: `macos/007 Solver.app` and `dmg/` on Mac, or `nsis/` on Windows. A local Mac build does not require a Developer ID; public distribution should use [Apple signing and notarization](https://v2.tauri.app/distribute/sign/macos/).

Mac packages currently link to the build machine's Homebrew `libomp` path and do not bundle that runtime. Running a package on another Mac requires a matching `libomp` installation at that path. The deployment target is macOS 11, but the linked runtime can require a newer OS; a current Homebrew bottle does not establish macOS 11 compatibility.

On Windows, run `chcp 65001` before building to let Ninja parse localized MSVC header dependencies. DCFR uses OpenMP; set `OMP_NUM_THREADS` to limit its worker count.

## Run the Solver Tests

Build and run the existing offline correctness suite:

```sh
cmake --preset release
cmake --build --preset release --target 007SolverTests
ctest --test-dir build --output-on-failure
```

The suite checks betting rules and settlement, an independent best-response baseline, two small reference solves, and fixed-policy node EVs and reach weights. Reference answers are checked into the repository; routine tests run offline. See [`tests/README.md`](tests/README.md) for coverage and reference generation.

CTest enforces a 60-second timeout for the suite. CI runs the same commands on Windows and macOS.

## Repository Checks

After installing dependencies and building the solver service, run the checks relevant to your changes:

```sh
npm --prefix desktop run check
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
pre-commit run --all-files
```

Pre-commit may fix formatting. Use `pre-commit run --files` with explicit paths to include new files that have not yet been staged.

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
