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

GPU support is enabled by default. Apple Silicon uses Metal; Intel Macs use CPU. Windows builds use CUDA when CMake finds a CUDA Toolkit compatible with the native MSVC compiler. CUDA requires compute capability 8.9 or newer and a compatible NVIDIA driver; without a CUDA compiler, the build uses CPU. Pass `-DSOLVER_ENABLE_GPU=OFF` for a CPU-only build, or `-DCMAKE_CUDA_COMPILER=<path-to-nvcc>` if the toolkit is outside the compiler search path.

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
./build/release/solver/solver_service tests/fixtures/weighted-flop.json
```

On Windows, append `.exe`. Use `--stdin` instead of the file path to send a scenario as the first JSON line, followed by query lines. `iterations` limits player updates; `accuracyPercent` is a percentage of the initial pot (`0.01` means `0.01%`). See [the input fixture](tests/fixtures/weighted-flop.json) and [parser](solver/io/ScenarioLoader.cpp) for the schema and defaults.

The service and desktop select an available GPU automatically. Append `--device=cpu`, `--device=gpu` or `--device=auto` to select explicitly; stderr reports the backend. An unavailable requested GPU or insufficient device memory is an error. The displayed memory estimate covers combined host/device allocations and does not impose a limit. CPU and GPU use float arithmetic; the CPU evaluator independently certifies GPU stopping and final metrics with the exported strategy's normalization.

## Checks

Build and run the offline C++ correctness suite:

```sh
cmake --preset Release
cmake --build --preset Release --target 007SolverTests
ctest --test-dir build/release --output-on-failure
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

Shared semantics and ownership are in [solver/ARCHITECTURE.md](solver/ARCHITECTURE.md); contribution rules are in [AGENTS.md](AGENTS.md).
