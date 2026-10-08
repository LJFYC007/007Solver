<p align="center">
  <img src="desktop/src-tauri/app-icon.svg" width="128" height="128" alt="007 Solver logo">
</p>

<h1 align="center">007 Solver</h1>

007 Solver is a Windows desktop app for heads-up postflop solving and strategy analysis that can also [serve its interface to browsers](#web-server). Its C++17 DCFR engine runs on the CPU and on NVIDIA GPUs through CUDA. Preflop lines, ranges and rake come from four [captured GTO Wizard solutions](resources/gtowizard-preflop/README.md); folded players' card removal is not modeled.

## Requirements

- Visual Studio 2022+ with **Desktop development with C++**, CMake 3.20+ and Ninja. Build from **Developer PowerShell for Visual Studio** after `chcp 65001`, so Ninja can parse localized MSVC dependencies.
- Node.js 22.13+ on the 22.x line, or 24+.
- Rust stable with rustfmt.
- Python 3, ClangFormat and pre-commit for repository checks.
- Optionally CUDA Toolkit 12.8+. GPU solving needs compute capability 8.6+ (RTX 30 series or newer).

CMake enables CUDA when it finds a Toolkit compatible with MSVC. Pass `-DSOLVER_ENABLE_GPU=OFF` to build CPU only, or `-DCMAKE_CUDA_COMPILER=<path-to-nvcc>` for a toolkit outside the compiler search path.

## Setup and development

```sh
git submodule update --init --recursive
npm --prefix desktop ci
npm --prefix desktop run dev
```

Restart `dev` after C++ changes. `npm --prefix desktop run build` packages the installer under `desktop/src-tauri/target/release/bundle/nsis/`. Both commands build and stage the C++ service; `npm --prefix desktop run build:solver` does only that step. Every push also builds a CUDA installer as a [workflow](.github/workflows/package.yml) artifact. The installer is unsigned, so Windows SmartScreen must be told to allow it.

### CLI

```sh
./build/release/solver/solver_service.exe tests/fixtures/weighted-flop.json
```

- The [fixture](tests/fixtures/weighted-flop.json) and [parser](solver/io/ScenarioLoader.cpp) show the scenario fields and defaults. `--stdin` reads the scenario as the first JSON line, followed by queries.
- `accuracyPercent` is the exploitability target as a percentage of the initial pot (`0.01` means 0.01%); `iterations` limits player updates.
- `--device=cpu|gpu|auto` overrides automatic GPU selection.
- `solver_service` requires AVX2; the app falls back to `solver_service_sse2` on CPUs without AVX2, FMA or BMI. Outside the installer, the service needs the Visual C++ x64 runtime, including OpenMP (`VCOMP140.DLL`).

## Web server

`007solver.exe --serve` serves the interface at `http://127.0.0.1:8007` instead of opening a window; `--serve=<address>` listens elsewhere. Only `npm --prefix desktop run build` embeds the page.

- **Exposure:** the server has no authentication, so expose it only through an authenticating proxy, such as Cloudflare Tunnel with Access. Against DNS rebinding it answers only requests addressed to an IP address or `localhost`, so the proxy must send `Host: 127.0.0.1:8007` (the tunnel's HTTP Host Header setting).
- **Benchmarking:** server solves queue for the GPU, but desktop, CLI and benchmark solves do not, so stop the server before benchmarking.
- **Deployment:** [deploy-server.ps1](scripts/deploy-server.ps1) builds the committed tree (uncommitted changes only with `-Force`), installs it as the `007 Solver server` startup app and restarts it, dropping all sessions; a failed build keeps the previous server. It logs to `build/deploy-server.log` and `build/deploy-build.log`. Copy [post-commit](scripts/post-commit) to `.git/hooks/` to deploy every commit on `main`. `Stop-Process -Name 007solver-server` stops the server until the next logon or deploy.
- **Browser development:** stop the server, run `npm --prefix desktop run build:solver`, `cargo run --manifest-path desktop/src-tauri/Cargo.toml -- --serve` and `npm --prefix desktop run dev:ui`, then open `http://127.0.0.1:1420`, which forwards `/api` to the server.

## Checks

```sh
cmake --preset Release
cmake --build --preset Release --target 007SolverTests
ctest --test-dir build/release --output-on-failure
```

CI runs these tests, which must finish within 15 seconds excluding builds. For desktop changes, after staging the service:

```sh
npm --prefix desktop run check
cargo check --manifest-path desktop/src-tauri/Cargo.toml --locked
```

`pre-commit install` installs the commit hooks, which format and lint, once per clone. Copy [pre-push](scripts/pre-push) to `.git/hooks/` (it replaces and runs Git LFS's hook): before pushes that change `solver/`, `tests/` or CMake files, it builds and runs these tests and, on machines with a supported NVIDIA GPU, `benchmark_gpu`, failing beyond its [time budget](tests/README.md#benchmark). It checks the working tree, including uncommitted changes, and waits for a running server deploy. Run the commit hooks on the changed paths; `pre-commit run --all-files` skips untracked files.

```sh
pre-commit run --files <changed-paths>
git diff --check
```

See [tests/README.md](tests/README.md) for benchmarks and reference generation, [solver/ARCHITECTURE.md](solver/ARCHITECTURE.md) for solver contracts and [AGENTS.md](AGENTS.md) for contribution rules.
