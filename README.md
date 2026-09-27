<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

007 Solver is a Windows desktop app for heads-up postflop solving and strategy analysis that can also [serve its interface to browsers](#web-server). Its C++17 DCFR engine runs on the CPU and on NVIDIA GPUs through CUDA.

The bundled [GTO Wizard catalog](resources/gtowizard-preflop/) contains chip-EV, 100bb, 6-max and 8-max ranges. Missing branches are not inferred. Solving is rake-free and does not model folded players' card-removal effects.

## Requirements

- CMake 3.20+, Ninja and MSVC.
- Node.js 22.13+ on the 22.x line, or 24+, with npm.
- Rust stable with Cargo and rustfmt.
- Python 3, ClangFormat and pre-commit for repository checks.

Use **Developer PowerShell for Visual Studio** with Visual Studio 2022+ and **Desktop development with C++** installed. Run `chcp 65001` before building so Ninja can parse localized MSVC dependencies. The installer downloads WebView2 when missing and bundles the Microsoft Visual C++ x64 runtime, including OpenMP (`VCOMP140.DLL`); CLI services copied elsewhere need that runtime installed. `solver_service` requires AVX2; the app runs `solver_service_sse2` on CPUs without AVX2, FMA or BMI.

Builds use CUDA by default when CMake finds a Toolkit compatible with MSVC; otherwise they use CPU. The CUDA build requires Toolkit 12.8+. CUDA requires compute capability 8.6+ (RTX 30 series or newer) and a compatible NVIDIA driver. Use `-DSOLVER_ENABLE_GPU=OFF` for CPU-only builds or `-DCMAKE_CUDA_COMPILER=<path-to-nvcc>` for a toolkit outside the compiler search path.

## Setup and development

From the repository root:

```sh
git submodule update --init --recursive
npm --prefix desktop ci
npm --prefix desktop run dev
```

Restart `dev` after C++ changes; React edits reload automatically. Use `npm --prefix desktop run build` to package the app. Both commands build and stage the C++ service; `npm --prefix desktop run build:solver` does only that step.

The installer appears under `desktop/src-tauri/target/release/bundle/nsis/`. Every push also builds a CUDA-enabled installer as a [workflow](.github/workflows/package.yml) artifact.

The installer is not signed with a developer certificate, so users must allow it past Windows SmartScreen.

For a small CLI solve after building the service:

```sh
./build/release/solver/solver_service.exe tests/fixtures/weighted-flop.json
```

`--stdin` accepts a scenario as the first JSON line, followed by queries. `iterations` limits player updates; `accuracyPercent` is exploitability as a percentage of the initial pot (`0.01` means `0.01%`). See the [fixture](tests/fixtures/weighted-flop.json) and [parser](solver/io/ScenarioLoader.cpp) for input fields and defaults.

The service and desktop select an available GPU automatically. Append `--device=cpu`, `--device=gpu` or `--device=auto` to override; stderr reports the backend. An unavailable requested GPU or insufficient device memory is an error. The displayed memory estimate covers combined host/device allocations and does not impose a limit. GPU stopping and final metrics are [CPU-certified](solver/ARCHITECTURE.md#training-and-memory).

## Web server

`007solver.exe --serve` serves the interface to browsers at `http://127.0.0.1:8007` instead of opening a window; `--serve=<address>` listens elsewhere. Only `npm --prefix desktop run build` embeds the page. The server has no authentication, so expose it only through an authenticating proxy, such as Cloudflare Tunnel with Access. Against DNS rebinding it answers only requests addressed to an IP address or `localhost`, so the proxy must send `Host: 127.0.0.1:8007` (the tunnel's HTTP Host Header setting).

Each page has its own session with one solve, like the desktop app. Solves take the GPU one at a time in request order and show as waiting until then; desktop, CLI and benchmark solves do not wait for it, so stop the server before benchmarking. Closing a page releases its solution. The server also releases a session after 10 minutes without requests and keeps at most four ready solutions, dropping the least recently used. A ready solution holds about 4 bytes of host memory per strategy entry, and its service keeps a CUDA context of a few hundred MB.

[deploy-server.ps1](scripts/deploy-server.ps1) deploys the committed tree from any shell: it builds, copies the release build to `%LOCALAPPDATA%\007 Solver Server\007solver-server.exe`, points the `007 Solver server` start-up entry (Task Manager → Startup apps) at it and restarts it, which drops all sessions. It appends results to `build/deploy-server.log`, keeps the previous server when the build fails (output in `build/deploy-build.log`) and deploys uncommitted changes only with `-Force`. Copy [post-commit](scripts/post-commit) to `.git/hooks/` to deploy every commit on `main` in the background. `Stop-Process -Name 007solver-server` stops the server until the next logon or deploy.

To develop the browser build, stop the server, stage the service with `npm --prefix desktop run build:solver`, run `cargo run --manifest-path desktop/src-tauri/Cargo.toml -- --serve` and `npm --prefix desktop run dev:ui`, then open `http://127.0.0.1:1420`, which forwards `/api` to the server.

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
