# Daily check journal

Newest first. One entry per recorded run: tested commit, results, triage, tests added. Detailed stage results are in [history.jsonl](history.jsonl) and in each run's `build/daily/runs/<stamp>/summary.md` (container-local).

## 2026-09-24 — harness created; baseline `c0ae04a`

**Before the harness existed** (manual session on `3617da4`, then `c51f6f2`):

- CPU backend: 9/9 gtests, CLI e2e 3/3, and the benchmark (2000 updates, 3.0 updates/s on 4 × Xeon @ 2.80GHz, 0.0107% accuracy) all passed. Also clean under ASan, UBSan and LSan.
- GPU kernels, emulated on the host, both CUDA and Metal dialects:
  - Repository GPU gtests: 3/3 each.
  - CLI e2e: 3/3 each.
  - `BackendParity` under ASan/UBSan: clean.
  - Bitwise race and lane-sync fingerprints: identical across up to 12 configurations on `backend-parity`, `weighted-flop`, `raise-flop`, `utg-bb-wide`, and a catalog scenario with more than 256 hands.
- Real `nvcc` 13.4 compile of `CudaExecutor.cu` for sm_89: no spills, 40–43 registers.
- The no-driver CUDA build falls back to CPU correctly.
- Desktop lint, typecheck, UI build, `cargo check` and pre-commit passed on both commits.

**Findings:**

- KI-1: an invalid UTF-8 request line kills the service session. See [README.md](README.md#known-issues).
- Coverage gaps in repository tests:
  - The GPU reference fixtures use at most 9 hands and a single lane.
  - No fixture exceeds 256 hands per player, so CUDA's multi-tile Backup path is never exercised.
  - CI runs `ctest` without `-V` and uploads no JSON, so it can't show whether GPU cases ran.
- `main` moved during the day to `c0ae04a`: CUDA architectures `86;89-real;120`, `DeviceAvailable` threshold 8.6, SSE2 service variant on MSVC. The harness now reads CUDA settings from the tested CMake instead of copying them.

**Tests added:**

- `cpu-determinism`: workers 1–4 are bitwise identical on all three fixtures.
- `protocol-{release,sanitize}`: KI-1 is an expected failure.
- `emu-parity` on three catalog-derived boards (dry, paired, monotone) with 547/654 hands.
- `gpu-plan-coverage`.
- Emulated GPU, race and sanitizer stages turned into the repeatable harness.

**Baseline run `20260924T044018Z`** (harness `bd10afd` with uncommitted edits): 82 PASS, 0 WARN, 0 FAIL, 0 SKIP in 61.5 min.

- All 51 race configurations are bitwise identical per workload.
- The emulated repository GPU gtests take 725 s (CUDA) and 1804 s (Metal). The GPU sanitizer runs take 1158 s and 2401 s and are the long poles.
- Catalog parity is within 1.1e-5 of node scale.
- The benchmark measured 2.61 updates/s at 0.0107% accuracy, but it overlapped harness compiles. The clean measurement from earlier in the day is 3.0 updates/s.

**Routine:** `trig_01JHuqTskJ3FtSCP2HLfWHQK` was created: daily at 18:00 UTC, a fresh session per firing (see [README.md](README.md#schedule)). It was fired manually once to verify the skip path; the result is recorded below.

**Independent review of the harness:** it found that replayed lane schedules can't expose overlapping lane scratch. [LaneCheck.h](gpu/LaneCheck.h) now proves the two-lane schedule conflict-free:

| Input | Unordered pass pairs per player | Conflicts |
|---|---|---|
| `backend-parity` | 450 | 0 |
| `btn-bb-srp-*` | 3,234 | 0 |
| `utg-bb-wide` | 25,250 | 0 |

The no-join, no-fork-wait and overlapping-scratch mutations are all detected. The review also led to stricter emulation, sanitizer logs collected from child processes, and hardened run.py stage and timeout handling (commit `46f2597`).
