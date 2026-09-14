<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

007 Solver is a Windows/macOS desktop app with a C++17 CPU DCFR engine and a React/Tauri interface. Select a captured GTO Wizard preflop line and a flop, solve the heads-up continuation, then inspect strategies, node EVs and exact showdown equity.

The bundled [catalog](resources/gtowizard-preflop/) contains chip-EV, 100bb, 6-max and 8-max ranges. Only captured branches are available. Postflop solving is heads-up and rake-free; folded players' card-removal effects are not modeled. Finishing the iteration budget does not guarantee convergence. The displayed memory estimate is informational and does not limit the solve.

## Requirements

- CMake 3.20+, Ninja and a native C++ compiler.
- Node.js 22.13+ on the 22.x line, or 24+, with npm.
- Rust stable with Cargo and rustfmt.
- Python 3, ClangFormat and pre-commit for repository checks.

On macOS, install Xcode Command Line Tools (`xcode-select --install`) and `brew install cmake ninja libomp`. Build for the Mac's native Apple Silicon or Intel architecture.

On Windows, use **Developer PowerShell for Visual Studio** with Visual Studio 2022+ and **Desktop development with C++** installed. Run `chcp 65001` before building so Ninja can parse localized MSVC dependencies. Running the app also requires WebView2 and the Microsoft Visual C++ x64 runtime, including OpenMP (`VCOMP140.DLL`).

## Setup and development

From the repository root:

```sh
git submodule update --init --recursive
npm --prefix desktop ci
npm --prefix desktop run dev
```

Development builds and stages the C++ service automatically. Restart the development command after C++ changes; React edits reload through Vite.

| Command | Purpose |
|---|---|
| `npm --prefix desktop run build` | Build and package the desktop app |
| `npm --prefix desktop run build:solver` | Build and stage only the C++ service |

Packages appear under `desktop/src-tauri/target/release/bundle/`: `macos/` and `dmg/` on macOS, or `nsis/` on Windows.

Mac packages link to the build machine's Homebrew `libomp` path without bundling it. Another Mac needs a compatible runtime at that path; the linked runtime can require a newer OS than the configured deployment target.

For a small CLI solve after building the service:

```sh
./build/solver/solver_service resources/default.json
```

On Windows, append `.exe`. Use `--stdin` instead of the file path to send a scenario as the first JSON line, followed by query lines. The example and parser are [resources/default.json](resources/default.json) and [ScenarioLoader.cpp](solver/io/ScenarioLoader.cpp).

## Checks

Build and run the offline C++ correctness suite:

```sh
cmake --preset release
cmake --build --preset release --target 007SolverTests
ctest --test-dir build --output-on-failure
```

For desktop changes, after installing dependencies and building/staging the service:

```sh
npm --prefix desktop run check
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
```

Run formatting and repository hooks for the changed paths, including untracked files:

```sh
pre-commit run --files <changed-paths>
git diff --check
```

Hooks may fix formatting. Use `pre-commit run --all-files` when a whole-repository check is needed; it only includes tracked files. Benchmark commands and independent reference generation are in [tests/README.md](tests/README.md).

## Code and documentation

- [solver/](solver/): C++ solver and service; read [ARCHITECTURE.md](solver/ARCHITECTURE.md) for shared semantics and ownership.
- [desktop/src/](desktop/src/) and [desktop/src-tauri/](desktop/src-tauri/): React UI and Rust bridge.
- [tests/](tests/): correctness tests, benchmark and independent oracle.
- [resources/](resources/) and [scripts/](scripts/): captured ranges, CLI example and tooling.
- [AGENTS.md](AGENTS.md): contribution constraints and verification rules.

Build targets, scripts and hooks are defined in [CMakeLists.txt](CMakeLists.txt), [CMakePresets.json](CMakePresets.json), [desktop/package.json](desktop/package.json) and [.pre-commit-config.yaml](.pre-commit-config.yaml).
