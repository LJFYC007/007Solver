# Daily checks

An independent, local verification of `origin/main`, run once a day by a scheduled Claude Code session. It doesn't use GitHub Actions.

- **Where it lives:** branch `claude/peaceful-edison-vk5axu` ([LJFYC007/007Solver#1](https://github.com/LJFYC007/007Solver/pull/1)). It is pushed there only because cloud containers are ephemeral.
- **What it tests:** a detached worktree of `origin/main` at `build/daily/src`.
- **What it may edit:** only files under `tests/daily/`.

Read this file first, then [BACKLOG.md](BACKLOG.md) and the latest [JOURNAL.md](JOURNAL.md) entry.

## Daily routine

1. **Skip unless `main` moved.** Run `python3 tests/daily/run.py --check-new`. On `SKIP`, stop immediately: no builds, tests or new tests, and reply in one line. This saves the owner's API credit, at the owner's request.
2. **Run everything.** Run `python3 tests/daily/run.py` in the background, about 70 minutes on 4 CPUs. It bootstraps the container ([setup.sh](setup.sh)), builds four configurations and writes `build/daily/runs/<UTC stamp>/{summary.md,summary.json,logs/}`. It then appends to [history.jsonl](history.jsonl) and sets `lastTestedMainSha` in [state.json](state.json).
3. **Triage every FAIL and WARN.** Read the stage log and decide between a product regression, a known issue or a harness defect.
   - Reproduce product regressions minimally and report them. Don't change product code unless the owner asks.
   - Fix harness defects in `tests/daily/`. Upstream changes to interfaces, CMake or fixtures are expected to break the harness; keeping it working is part of the job.
4. **Grow the suite.** Add at least one test from [BACKLOG.md](BACKLOG.md), or a better idea found during triage.
   - Show that it passes, or that it fails for a real reason, before wiring it into `run.py`.
   - A check that can't fail is worse than none: prefer mutation self-tests, as in [tools/planstats.cpp](tools/planstats.cpp).
   - Test ranges must come from the captured catalog (use [scenarios.py](scenarios.py)); never introduce custom ranges.
   - Update the backlog.
5. **Record.** Add a dated [JOURNAL.md](JOURNAL.md) entry: tested commit, counts, failures and their triage, benchmark rate, tests added. Then run `pre-commit run --files <changed tests/daily paths>`, commit, and push to `claude/peaceful-edison-vk5axu`.
   - Pushes update PR #1 and trigger the repository's CI. Never merge it, and never push to `main`.
   - If PR #1 has been merged, recreate the branch from `origin/main` (which then contains `tests/daily/`) and keep pushing daily records there.
6. **Report** a short summary to the owner: new failures first, then what was added.

For harness development, use `run.py --quick` (skips the slowest stages) or `--only '<regex>'`. Neither records state. `--no-record` suppresses recording for a complete run.

## Stages

| Stage | Verifies |
|---|---|
| `build-{release,emulation,sanitize,cuda}` | The tested revision builds. Compiler warnings from files recompiled in this run are WARN; a fresh container rebuilds everything. The CUDA build uses real `nvcc` and reports ptxas registers and spills per kernel and architecture. |
| `cpu-ctest` | Repository gtests and CLI e2e; only GPU cases may skip. |
| `cpu-determinism` | Training state is bitwise identical for 1–4 CPU workers ([cpu_determinism.cpp](tools/cpu_determinism.cpp)). |
| `protocol-{release,sanitize}` | JSON-lines robustness: one reply per line, sessions survive errors, report invariants on sampled nodes, EOF drains EV work. KI-1 is pinned ([protocol_test.py](protocol_test.py)). |
| `sanitize-cpu` | Gtests and CLI e2e clean under ASan, UBSan and LSan. Reports from every child process are collected through `log_path` (exit code 86). |
| `cuda-no-driver-fallback` | A CUDA build on a host without an NVIDIA driver: auto uses CPU, `--device=gpu` fails as JSON with exit 1. |
| `emu-gtest-{cuda,metal}` | The repository's GPU gtests (both reference solves plus `BackendParity`) on the host-emulated kernels. The report must name the emulated device. |
| `emu-cli-{cuda,metal}` | CLI e2e against the emulation build. Its one `--device=auto` case runs on the emulated GPU; the others request CPU. |
| `emu-race-verdict:<workload>` | Every emulator configuration gives one bitwise fingerprint. Configurations are dialect × thread order, times lane schedule for two-lane plans. |
| `emu-parity:<scenario>:<dialect>` | Lockstep CPU/GPU parity on catalog scenarios with more than 256 hands per player (enforced by `gpu-plan-coverage`), which reach CUDA's multi-tile Backup grid. |
| `gpu-plan-coverage` | For every input, the two-lane CUDA schedule is legal and conflict-free, and each mutation of the check is detected ([LaneCheck.h](gpu/LaneCheck.h)). `backend-parity` must still use two-lane batches, which implies the several batches [tests/README.md](../README.md) requires. |
| `sanitize-gpu-{cuda,metal}` | `BackendParity` on emulated kernels under ASan and UBSan, catching out-of-bounds device-buffer and shared-memory access. |
| `desktop-check`, `desktop-cargo-check` | `npm run check`, the UI build and `cargo check --locked` of the Tauri crate. |
| `repo-pre-commit` | All repository hooks pass without modifying the tested revision. |
| `perf-benchmark-cpu` | The opt-in wide benchmark passes its independent uniform reference. WARN when updates/s drops more than 15% below the median of up to five earlier runs on the same host type (at least two earlier runs are needed). |

## GPU emulation

The host has no GPU. [gpu/](gpu/) replaces `GpuUnavailable.cpp` with an `Executor` that compiles the real [GpuKernels.inc](../../solver/engine/gpu/GpuKernels.inc) as host C++, once through its CUDA branch and once through its Metal branch. It launches passes with [CudaExecutor.cu](../../solver/engine/gpu/CudaExecutor.cu) or [MetalExecutor.mm](../../solver/engine/gpu/MetalExecutor.mm) grid geometry and 32-bit launch arithmetic. Each GPU thread of a Terminal block is a fiber, so block barriers and warp shuffles are real rendezvous.

- **Environment:**
  - `SOLVER_GPU_EMULATION=cuda|metal|off`
  - `SOLVER_GPU_EMULATION_ORDER=forward|reverse` (thread and block order within a pass)
  - `SOLVER_GPU_EMULATION_SCHEDULE=inorder|lane1-late|lane0-late` (legal CUDA-graph orders of the two river lanes)
- **What aborts:**
  - Threads meeting at barriers or collectives from different source lines.
  - A collective with exited lanes.
  - A Metal thread exiting before a barrier.
  - Deadlocks.
  - Plans whose lane structure fails [LaneCheck.h](gpu/LaneCheck.h).
- **Why fingerprints must match:** each thread's arithmetic is deterministic, both dialects compute the same per-thread operations, and emulated sources don't contract FMAs (mirroring `--fmad=false`). A read that depends on thread order therefore changes bits under forward versus reverse order. If a kernel change ever makes CUDA and Metal differ legitimately (for example, summation order tied to group size), compare within each dialect rather than loosening the check.
- **What the schedules miss, and what covers it:** the schedules run each pass to completion, so they can't expose two lanes interleaving on overlapping scratch. [LaneCheck.h](gpu/LaneCheck.h) covers that statically. It rebuilds the captured stream/event graph from `Pass::sync` and requires that passes left unordered touch disjoint data.
- **Stricter than hardware:**
  - Scratch and values are re-poisoned with NaN before every update, and flags with all ones, so a read before this update's write shows up.
  - Host buffers are sized exactly as the executors allocate them. Under ASan, CUDA shared memory beyond the launch's dynamic size is unaddressable.
- **Not covered:**
  - Metal shader compilation (MSL).
  - Real concurrency and memory ordering, including lost updates between threads of one pass, since a thread's segment between barriers runs without interruption.
  - Driver behaviour and GPU performance.
  - The real CUDA code is only compiled.

## Contracts that are easy to break

- **Linux is not a supported repository host.** [cmake/daily.cmake](cmake/daily.cmake) swallows the desktop-sidecar `FATAL_ERROR`, adds the Mach stand-in [shims/mach/mach.h](shims/mach/mach.h) for the benchmark, and swaps GPU backends after the repository's CMake runs. It targets `solver_lib`. If upstream renames targets, fix the hook, not the repository.
- **CUDA settings are mirrored, not copied.** `CUDA_ARCHITECTURES` and the CUDA compile options are read from the tested `solver/CMakeLists.txt` (`c0ae04a` moved from `89` to `86;89-real;120`). A missing pattern fails the configure stage by design.
- **[scenarios.py](scenarios.py) executes the tested `scripts/sync-preflop-fixtures.py` up to `fixtures = ROOT`.** If that line moves, update the split.
- **Toolchains:** NVIDIA's download host is blocked by the network policy, so CUDA 13.4 comes from PyPI wheels into `build/daily/toolchains/`. WebKitGTK comes from apt; an unreachable PPA during `apt-get update` is harmless.
- **Benchmark memory fields on Linux come from the shim.** `physical_footprint_bytes` is the current RSS, and `peak_resident_set_bytes` is `ru_maxrss` in KiB, not bytes. Don't compare them with macOS or Windows reports.
- **Recording:** only a complete run of `origin/main` (no `--only`, `--quick`, `--no-record` or other `--ref`) writes `state.json` and `history.jsonl`, whether it passed or failed.

## Known issues

Known product defects are pinned by tests that assert today's broken behaviour and fail with "appears fixed" once the product changes. When that happens, replace the test with the correct expectation and move the entry to the journal.

| ID | Found | Issue | Test |
|---|---|---|---|
| KI-1 | 2026-09-24 | A request line with invalid UTF-8 ends the solve session. nlohmann's parse error quotes the bytes, `WriteMessage` can't serialize the reply, and the exception escapes the per-request handler in [SolverService.cpp](../../solver/service/SolverService.cpp), so the service emits `failed`, exits 1 and drops later queries. The desktop sends serde JSON, so this mainly affects CLI clients. | `protocol_test.py` `test_ki1_*` |

## Layout

| Path | Contents |
|---|---|
| [run.py](run.py) | Orchestrator: preparation, builds, CPU-budgeted parallel stages, race verdicts, final sequential stages, recording |
| [setup.sh](setup.sh) | Idempotent container bootstrap with pinned CUDA wheel and pre-commit versions |
| [cmake/daily.cmake](cmake/daily.cmake) | CMake hook injected with `CMAKE_PROJECT_007Solver_INCLUDE` |
| [gpu/](gpu/) | Emulated executor, fiber runtime, CUDA/Metal dialects and the static lane check |
| [tools/](tools/) | Drivers built as `daily_*` targets: fingerprint, lockstep parity, plan statistics with lane check, CPU determinism |
| [scenarios.py](scenarios.py), [protocol_test.py](protocol_test.py) | Catalog-derived scenarios; protocol robustness suite |
| [BACKLOG.md](BACKLOG.md), [JOURNAL.md](JOURNAL.md) | Test ideas queue; dated run log |
| [state.json](state.json), [history.jsonl](history.jsonl) | Last tested `main` commit; one line of stage results and benchmark metrics per recorded run |
