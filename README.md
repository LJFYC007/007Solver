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
GTO Wizard preflop catalog -> action line + flop -> C++ service <-> Tauri/Rust <-> React
```

The application currently:

- browses captured GTO Wizard chip-EV ranges for 6-max and 8-max at 100bb;
- selects preflop actions, inspects every hand class and exact combo, and chooses a flop;
- runs CPU DCFR over the resulting heads-up ranges;
- navigates decision, chance, and terminal nodes along the selected action line;
- displays a 13 x 13 hand matrix with mixed-strategy colors;
- reports aggregate action frequencies, combo strategies, and reach weights;
- shows the board, pot, remaining stacks, street, and available runouts.

The desktop starts in the preflop browser. Selecting a heads-up line and flop starts a C++ service that builds the tree, trains a strategy and exports one snapshot. It releases training state before evaluating exploitability. The service keeps the result in memory and answers node queries through newline-delimited JSON. Each node ID identifies one complete action and card history within that tree.

The C++ module boundaries and dependency direction are documented in [`solver/ARCHITECTURE.md`](solver/ARCHITECTURE.md).

## Current Scope

The desktop uses [`resources/gtowizard-preflop/`](resources/gtowizard-preflop/), captured from GTO Wizard's displayed matrices from 2026-09-09 through 2026-09-11. Both solutions use chip EV, 100bb, no rake or ante, and cold calls. Opens are 2.5bb except SB (3.5bb in 6-max, 3bb in 8-max).

The source has one level of grouping: `catalog.json` holds shared metadata and solution settings, while `6max/<actor>.json` and `8max/<actor>.json` hold arrays of captured nodes for each acting position (14 position files in total). For example, `6max/bb.json` contains every saved 6-max BB decision, distinguished by its full action history. Add future captures to the corresponding position file and update capture metadata; no combined JSON or file list needs regeneration. Both the desktop and fixture script discover these files directly and reject duplicate histories or mismatched positions/solutions.

The catalog contains **208 decision spots: 87 in 6-max and 121 in 8-max**. It covers all unopened positions and responses to a single open; SB limp/BB raise/reraise main lines including direct all-ins; every opener versus a standard BB or SB 3-bet followed by the standard 4-bet/5-bet/all-in line with other players folding; direct 4-bet/5-bet shoves against BB and all 6-max SB 3-bets; and BB after BTN opens and SB calls. The 8-max direct-shove responses against SB cover BTN and CO's direct 4-bet shove so far. The third capture session added 68 matrices before reaching the source account's daily browsing limit. The catalog is not a complete preflop tree: non-blind 3-bets, additional cold-call/squeeze lines, early all-in branches, and remaining 8-max direct shoves against SB are still unavailable. Missing branches have no estimated or legacy range fallback.

Matrix collection stopped on 2026-09-12 by project decision. The saved 208 spots are the supported catalog; unavailable branches remain unavailable.

Each captured hand row stores action frequencies at the displayed 0.01 percentage-point precision, together with its source URL, history and action labels. Rows are normalized for display rounding. A player's range is the product of that player's selected action frequencies; a hand omitted from a later source matrix has no captured continuation. The full source ranges feed desktop solves. Folded contributions stay in the pot, and stacks reflect amount-to commitments. Only completed heads-up lines can start a postflop solve. This is a heads-up continuation using imported ranges; folded players' card-removal effects are not modeled.

The Study workspace uses one continuous preflop-to-river history, matrix and inspector. Completing a heads-up preflop line automatically opens the four-color flop picker. Select three distinct cards, then choose **Solve postflop**. **Solver settings** expands the iteration budget; the solve panel shows preparation, training, cancellation and retry states. In the card picker, Enter confirms a complete board and Backspace removes the last unlocked card. The **Change** control selects 6-max or 8-max; clicking the flop card in the action history reopens its picker. Clicking an earlier history card reviews it without discarding the solved continuation; choosing a different preflop action clears the board and solved continuation, while confirming a different flop invalidates the solution. Turn and river pickers lock existing board cards and allow only legal outcomes from the current node. Once an all-in is called and no betting remains, the UI ends the action line without prompting for more public cards; the solver still accounts for every legal runout. At the input boundary one source bb maps to one solver chip. Core amounts remain chips with 0.1-chip precision, no rake and a 0.1-chip minimum bet.

The inspector switches between **Overview**, **Table** and **Equity chart** while keeping the action mix and exact-suit hand details visible. Panel sizes depend on the selected tab and window, not the selected hand. Hand labels retain a fixed readable size; strategy text has a minimum size and the combo area scrolls when space is limited. Offsuit hands retain all 12 combinations. Preflop and postflop share action colors, four-suit colors, typography and neutral surface tokens; source catalog colors remain unchanged. Table and Equity chart share both players' EV, equity, EQR and combo cards. EQR uses current-node EV divided by equity times current pot, as identified in its tooltip; unavailable values remain blank. The equity chart computes exact heads-up showdown equity from current ranges over all legal remaining runouts, independently of future betting strategy. It is queried on demand for solved postflop nodes and cached for that solution; it is distinct from node strategy EV. Unsolved preflop spots have no equity report.

[`resources/default.json`](resources/default.json) remains a small GTO Wizard-derived CLI example, with reduced pot/stacks and hand classes for quick runs. It is not loaded by the desktop. Run `./build/solver/solver_service resources/default.json`, or pass `--stdin` and send one scenario JSON line followed by node-query lines.

The default postflop tree uses one 50% pot bet and one 50% pot-after-call raise size. Each street allows one ordinary raise; further aggression uses the effective-stack maximum. Sizes clamp to legal minimums and effective stacks, with duplicates removed. A called all-in is evaluated over every legal remaining runout without storing all its traversal nodes. Betting topology is shared across cards while every concrete history retains its own logical node ID and strategy.

Before allocating training state, the service estimates solve/export/evaluation peak memory and checks an available-memory budget. The solve panel shows the estimate during training. CLI scenarios may set a positive integer `memoryBudgetMiB`; otherwise the service uses 75% of available physical memory (also limited by available commit on Windows). Estimates include headroom but exclude later navigation caches. Oversized games fail before the large allocations; ranges are never reduced automatically.

DCFR is the sole solver. The optional `algorithm` field accepts only `dcfr`, which is also the default. Each iteration updates one player across the full tree, using bounded depth-first workspaces. Ordinary chance subtrees run in a bounded OpenMP pool; regrets and average strategies persist across iterations.

Training stops at the configured iteration count. The selected iteration budget does not guarantee convergence. Exploitability is reported in the service diagnostics, and "Solution ready" means the solve has finished without enforcing an accuracy threshold.

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

There is one Release/Ninja preset and one C++ output directory, `build/`. CMake stages the native service in `desktop/src-tauri/binaries/` with Tauri's target suffix. Both development and packaging run this step automatically. After changing C++ code during development, restart `npm --prefix desktop run dev` to rebuild the service. React edits reload through Vite.

The app shows solve progress and extends the shared study timeline after training, exploitability evaluation and the initial node query finish. Each solve starts a fresh process. Viewing preflop history preserves the result; changing the preflop line or flop releases it.

In the frontend, `App.tsx` owns the service session; `StudyWorkspace.tsx` owns the selected history, board and view. `catalog.ts` loads source data and `preflop.ts` replays it into ranges/scenarios. The two street-specific matrix adapters share cell rendering, while timeline, solve-panel and equity-chart components handle their own presentation. Navigation and equity hooks retain reports for the current solution and discard them when its generation changes. History/board edits pass through one guarded invalidation path before committing new input.

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

The local Release target is under 20 seconds for the whole correctness suite. CTest retains a 60-second timeout to allow for slower CI machines. CI runs the same commands on Windows and macOS.

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
resources/                  GTO Wizard catalog and sourced CLI example
third_party/                Git submodules
scripts/                    Frontend tooling and sourced fixture generation
```
