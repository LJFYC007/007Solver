# Test backlog

Each full daily run adds at least one test from here (see [README.md](README.md#daily-routine)). Keep items concrete: what to check, why it can fail, and how to run it cheaply. Move finished items to **Done** with the date.

## Next

1. **Node EV identity.** [ARCHITECTURE.md](../../solver/ARCHITECTURE.md#values-and-reach) says the two node strategy EVs sum to that node's pot. Derive the exact reach-weighted aggregation from `analysis/` before asserting it. Then query EVs on sampled decision nodes of a solved fixture and check the identity within float tolerance.
2. **Scenario loader fuzzing.** Malformed or boundary scenarios must fail as one JSON `failed` message, never crash. Cover missing streets, negative stacks, a board with duplicate cards, a range hand that conflicts with the board, empty ranges, huge bet percentages, `maxRaises` limits and `allInSpr` edges. Run under the sanitizer build.
3. **AVX2/FMA build variant.** `c0ae04a` ships `/arch:AVX2` and SSE2 services on Windows. Build with `-march=x86-64-v3` (and once with `-ffp-contract=fast`) and require the reference gtests to still pass. This guards the documented tolerances against contracted arithmetic.
4. **Clang build.** Production uses AppleClang and MSVC, but every daily build uses GCC. Add a clang++ build with `-Wall -Wextra` (informational WARN) that runs `cpu-ctest`.
5. **Memory estimate versus reality.** For each fixture, compare `DcfrSession::Memory()` with peak RSS measured from `/proc/self/status`. Flag an estimate below measured usage, or grossly above it.
6. **Checkpoint behaviour.** On emulated GPU, the solve stops only after CPU certification when a provisional GPU checkpoint reaches the target ([ARCHITECTURE.md](../../solver/ARCHITECTURE.md#training-and-memory)). Drive `DcfrSession::EvaluateCheckpoint` directly with a target between the GPU and CPU estimates.
7. **Longer GPU/CPU agreement.** Independent 200-update trajectories diverge bitwise, but final exploitability on `backend-parity` should agree within a small relative bound between the CPU and emulated GPU backends.

## Later

- **ThreadSanitizer for the service.** The input thread and the EV worker share the analysis session. GCC's libgomp isn't TSan-aware, so evaluate clang with an instrumented libomp, or run the service with one OpenMP thread.
- **Independent oracle on new scenarios.** Run [tests/oracle](../oracle/) on the catalog scenarios and compare converged exploitability and root values. Measure the runtime first; the oracle needs the pinned Rust lint allowances.
- **Emulator speed.** Terminal fibers dominate emulated runtime. Faster switching or warp-level batching would allow longer GPU workloads within the daily budget.
- **Desktop unit tests.** Test the preflop adapter (`desktop/src/solver/preflop.ts`) with `node --test` against catalog data: weight normalization, missing branches without fallback, IP/OOP mapping.
- **Rust lint.** Add `cargo clippy --locked` on `desktop/src-tauri` as informational WARN.
- **Wider GPU group sizes.** Run Terminal with 256-thread groups in the emulator to confirm the kernels don't depend on the executors' current group sizes. This only matters if an executor starts choosing sizes dynamically.

## Done

- 2026-09-24: CPU worker-count determinism (`cpu-determinism`).
- 2026-09-24: Protocol robustness, including KI-1 (`protocol-*`).
- 2026-09-24: Catalog scenarios with more than 256 hands per player and lockstep GPU parity (`emu-parity:*`).
- 2026-09-24: Emulated GPU gtests, race and lane-sync fingerprints, GPU sanitizers, CUDA ptxas statistics and no-driver fallback, and plan coverage.
