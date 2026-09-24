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

**Baseline run:** pending.
