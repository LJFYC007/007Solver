# Daily checks

An independent, local verification of `origin/main`, run once a day by a scheduled Claude Code session. It doesn't use GitHub Actions. The harness lives on branch `claude/peaceful-edison-vk5axu`, pushed there only because cloud containers are ephemeral. It tests a detached worktree of `origin/main` at `build/daily/src` and never edits repository files outside `tests/daily/`.

Read this file first, then [BACKLOG.md](BACKLOG.md) and the latest [JOURNAL.md](JOURNAL.md) entry.

## Daily routine

1. **Skip unless `main` moved.** Run `python3 tests/daily/run.py --check-new`. On `SKIP`, stop immediately: no builds, tests or new tests, and reply in one line. This saves the owner's API credit, at the owner's request.
2. **Run everything.** Run `python3 tests/daily/run.py` in the background, about 70 minutes on 4 CPUs. It bootstraps the container ([setup.sh](setup.sh)), builds four configurations and writes `build/daily/runs/<UTC stamp>/{summary.md,summary.json,logs/}`. It then appends to [history.jsonl](history.jsonl) and sets `lastTestedMainSha` in [state.json](state.json).
3. **Triage every FAIL and WARN.** Read the stage log and decide between a product regression, a known issue or a harness defect.
   - Reproduce product regressions minimally and report them. Don't change product code unless the owner asks.
   - Fix harness defects in `tests/daily/`. Upstream changes to interfaces, CMake or fixtures are expected to break the harness; keeping it working is part of the job.
4. **Grow the suite.** Add at least one test from [BACKLOG.md](BACKLOG.md), or a better idea found during triage.
   - Show that it passes, or that it fails for a real reason, before wiring it into `run.py`.
   - Test ranges must come from the captured catalog (use [scenarios.py](scenarios.py)); never introduce custom ranges.
   - Update the backlog.
5. **Record.** Add a dated [JOURNAL.md](JOURNAL.md) entry: tested commit, counts, failures and their triage, benchmark rate, tests added. Then run `pre-commit run --files <changed tests/daily paths>`, commit, and push to `claude/peaceful-edison-vk5axu`.
6. **Report** a short summary to the owner: new failures first, then what was added.

For harness development, use `run.py --quick` (skips the slowest stages) or `--only '<regex>'`. Neither records state. `--no-record` suppresses recording for a complete run.

## Stages

| Stage | Verifies |
|---|---|
| `build-{release,emulation,sanitize,cuda}` | The tested revision builds with GCC 13 (warnings are WARN). The CUDA build uses real `nvcc` and reports ptxas registers and spills per kernel and architecture. |
| `cpu-ctest` | Repository gtests and CLI e2e; only GPU cases may skip. |
| `cpu-determinism` | Training state is bitwise identical for 1–4 CPU workers ([cpu_determinism.cpp](tools/cpu_determinism.cpp)). |
| `protocol-{release,sanitize}` | JSON-lines robustness: one reply per line, sessions survive errors, report invariants on sampled nodes, EOF drains EV work ([protocol_test.py](protocol_test.py)). |
| `sanitize-cpu` | Gtests and CLI e2e clean under ASan, UBSan and LSan. |
| `cuda-no-driver-fallback` | A CUDA build on a host without an NVIDIA driver: auto uses CPU, `--device=gpu` fails as JSON with exit 1. |
| `emu-gtest-{cuda,metal}` | The repository's GPU gtests (both reference solves plus `BackendParity`) on the host-emulated kernels. |
| `emu-cli-{cuda,metal}` | CLI e2e with the emulated GPU selected by `--device=auto`. |
| `emu-race-verdict:<workload>` | Every emulator configuration gives one bitwise fingerprint. Configurations are dialect × thread order × lane schedule; see below. |
| `emu-parity:<scenario>:<dialect>` | Lockstep CPU/GPU parity on catalog scenarios with more than 256 hands per player, which reach CUDA's multi-tile Backup grid. |
| `gpu-plan-coverage` | `backend-parity` still splits river regions into two lanes (a [tests/README.md](../README.md) contract); records lane and sync coverage per input. |
| `sanitize-gpu-{cuda,metal}` | `BackendParity` on emulated kernels under ASan and UBSan, catching out-of-bounds device-buffer and shared-memory access. |
| `desktop-check`, `desktop-cargo-check` | `npm run check`, the UI build and `cargo check --locked` of the Tauri crate. |
| `repo-pre-commit` | All repository hooks pass without modifying the tested revision. |
| `perf-benchmark-cpu` | The opt-in wide benchmark passes its independent uniform reference. WARN when updates/s drops more than 15% below the median of up to five earlier runs on the same host type. |

## GPU emulation

The host has no GPU. [gpu/](gpu/) replaces `GpuUnavailable.cpp` with an `Executor` that compiles the real [GpuKernels.inc](../../solver/engine/gpu/GpuKernels.inc) as host C++, once through its CUDA branch and once through its Metal branch. It launches passes with [CudaExecutor.cu](../../solver/engine/gpu/CudaExecutor.cu) or [MetalExecutor.mm](../../solver/engine/gpu/MetalExecutor.mm) grid geometry. Each GPU thread of a Terminal block is a fiber, so block barriers and warp shuffles are real rendezvous. Divergent barriers, collectives with exited lanes and deadlocks abort.

- **Environment:**
  - `SOLVER_GPU_EMULATION=cuda|metal|off`
  - `SOLVER_GPU_EMULATION_ORDER=forward|reverse` (thread and block order within a pass)
  - `SOLVER_GPU_EMULATION_SCHEDULE=inorder|lane1-late|lane0-late` (legal CUDA-graph orders of the two river lanes under `Pass::sync`)
- **Why fingerprints must match:** each thread's arithmetic is deterministic and independent of the others, and both dialects compute the same per-thread operations. So an intra-pass race, or a missing lane-sync edge between lanes that share data, changes bits under some configuration. If a kernel change ever makes CUDA and Metal differ legitimately (for example, summation order tied to group size), compare within each dialect rather than loosening the check.
- **Stricter than hardware:** device-only buffers and shared memory are poisoned with NaN, so reads before writes show up. Host buffers are sized exactly as the executors allocate them, so ASan sees overruns.
- **Not covered:** Metal shader compilation (MSL), real concurrency and memory ordering, driver behaviour and GPU performance. The real CUDA code is only compiled.

## Contracts that are easy to break

- **Linux is not a supported repository host.** [cmake/daily.cmake](cmake/daily.cmake) swallows the desktop-sidecar `FATAL_ERROR`, adds the Mach stand-in [shims/mach/mach.h](shims/mach/mach.h) for the benchmark, and swaps GPU backends after the repository's CMake runs. It targets `solver_lib`. If upstream renames targets, fix the hook, not the repository.
- **CUDA settings are mirrored, not copied.** `CUDA_ARCHITECTURES` and the CUDA compile options are read from the tested `solver/CMakeLists.txt` (`c0ae04a` moved from `89` to `86;89-real;120`). A missing pattern fails the configure stage by design.
- **[scenarios.py](scenarios.py) executes the tested `scripts/sync-preflop-fixtures.py` up to `fixtures = ROOT`.** If that line moves, update the split.
- **Toolchains:** NVIDIA's download host is blocked by the network policy, so CUDA 13.4 comes from PyPI wheels into `build/daily/toolchains/`. WebKitGTK comes from apt; an unreachable PPA during `apt-get update` is harmless.
- **Benchmark memory fields on Linux come from the shim.** `physical_footprint_bytes` is the current RSS, and `peak_resident_set_bytes` is `ru_maxrss` in KiB, not bytes. Don't compare them with macOS or Windows reports.
- **Recording:** only a complete run (no `--only`, `--quick` or `--no-record`) writes `state.json` and `history.jsonl`, whether it passed or failed.

## Known issues

Known product defects stay visible as `unittest.expectedFailure` tests, which turn into failures ("unexpected success") once fixed. When that happens, remove the decorator and move the entry to the journal.

| ID | Found | Issue | Test |
|---|---|---|---|
| KI-1 | 2026-09-24 | A request line with invalid UTF-8 ends the solve session. nlohmann's parse error quotes the bytes, `WriteMessage` can't serialize the reply, and the exception escapes the per-request handler in [SolverService.cpp](../../solver/service/SolverService.cpp), so the service emits `failed`, exits 1 and drops later queries. The desktop sends serde JSON, so this mainly affects CLI clients. | `protocol_test.py` `test_invalid_utf8_*` |

## Layout

| Path | Contents |
|---|---|
| [run.py](run.py) | Orchestrator: preparation, builds, CPU-budgeted parallel stages, race verdicts, final sequential stages, recording |
| [setup.sh](setup.sh) | Idempotent container bootstrap with pinned CUDA wheel and pre-commit versions |
| [cmake/daily.cmake](cmake/daily.cmake) | CMake hook injected with `CMAKE_PROJECT_007Solver_INCLUDE` |
| [gpu/](gpu/) | Emulated executor, fiber runtime and CUDA/Metal dialects |
| [tools/](tools/) | Drivers built as `daily_*` targets: fingerprint, lockstep parity, plan statistics, CPU determinism |
| [scenarios.py](scenarios.py), [protocol_test.py](protocol_test.py) | Catalog-derived scenarios; protocol robustness suite |
| [state.json](state.json), [history.jsonl](history.jsonl) | Last tested `main` commit; one line of stage results and benchmark metrics per recorded run |
