<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

007 Solver is a Windows/macOS desktop app for heads-up postflop solving and strategy analysis. Its C++17 DCFR engine supports CPU, CUDA on Windows and Metal on Apple Silicon.

The bundled [GTO Wizard catalog](resources/gtowizard-preflop/) contains chip-EV, 100bb, 6-max and 8-max ranges. Missing branches are not inferred. Solving is rake-free and does not model folded players' card-removal effects.

## Requirements

- CMake 3.20+, Ninja and a native C++ compiler.
- Node.js 22.13+ on the 22.x line, or 24+, with npm.
- Rust stable with Cargo and rustfmt.
- Python 3, ClangFormat and pre-commit for repository checks.

On macOS, install Xcode Command Line Tools (`xcode-select --install`) and `brew install cmake ninja libomp`. Build for the Mac's native Apple Silicon or Intel architecture.

On Windows, use **Developer PowerShell for Visual Studio** with Visual Studio 2022+ and **Desktop development with C++** installed. Run `chcp 65001` before building so Ninja can parse localized MSVC dependencies. Running the app requires an AVX2-capable CPU, WebView2 and the Microsoft Visual C++ x64 runtime, including OpenMP (`VCOMP140.DLL`).

GPU support is enabled by default; Intel Macs use CPU. Windows builds use CUDA when CMake finds a Toolkit compatible with MSVC; otherwise they use CPU. CUDA requires compute capability 8.9+ and a compatible NVIDIA driver. Use `-DSOLVER_ENABLE_GPU=OFF` for CPU-only builds or `-DCMAKE_CUDA_COMPILER=<path-to-nvcc>` for a toolkit outside the compiler search path.

## Setup and development

From the repository root:

```sh
git submodule update --init --recursive
npm --prefix desktop ci
npm --prefix desktop run dev
```

Restart `dev` after C++ changes; React edits reload automatically. Use `npm --prefix desktop run build` to package the app. Both commands build and stage the C++ service; `npm --prefix desktop run build:solver` does only that step.

Packages appear under `desktop/src-tauri/target/release/bundle/`: `macos/` and `dmg/` on macOS, or `nsis/` on Windows.

Mac packages link to the build machine's Homebrew `libomp` path without bundling it. Another Mac needs a compatible runtime at that path; the linked runtime can require a newer OS than the configured deployment target.

For a small CLI solve after building the service:

```sh
./build/release/solver/solver_service tests/fixtures/weighted-flop.json
```

On Windows, append `.exe`. `--stdin` accepts a scenario as the first JSON line, followed by queries. `iterations` limits player updates; `accuracyPercent` is exploitability as a percentage of the initial pot (`0.01` means `0.01%`). See the [fixture](tests/fixtures/weighted-flop.json) and [parser](solver/io/ScenarioLoader.cpp) for input fields and defaults.

The service and desktop select an available GPU automatically. Append `--device=cpu`, `--device=gpu` or `--device=auto` to override; stderr reports the backend. An unavailable requested GPU or insufficient device memory is an error. The displayed memory estimate covers combined host/device allocations and does not impose a limit. GPU stopping and final metrics are [CPU-certified](solver/ARCHITECTURE.md#training-and-memory).

## Checks

```sh
cmake --preset Release
cmake --build --preset Release --target 007SolverTests
ctest --test-dir build/release --output-on-failure
```

Release tests target roughly 10 seconds, excluding builds, and include real CLI end-to-end checks.

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

Hooks may fix formatting. `pre-commit run --all-files` checks only tracked files. See [tests/README.md](tests/README.md) for benchmarks and independent reference generation.

Shared semantics and ownership are in [solver/ARCHITECTURE.md](solver/ARCHITECTURE.md); contribution rules are in [AGENTS.md](AGENTS.md).
