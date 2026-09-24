#!/usr/bin/env python3
"""Daily 007 Solver check runner; see tests/daily/README.md for the stage catalogue and workflow.

  python3 tests/daily/run.py --check-new     prints RUN or SKIP (compares origin/main with state.json)
  python3 tests/daily/run.py                 full run of origin/main, records state and history
  python3 tests/daily/run.py --only 'cpu-.*' --no-record   iterate on a subset
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
BUILD = ROOT / "build/daily"
SRC = BUILD / "src"
STATE = HARNESS / "state.json"
HISTORY = HARNESS / "history.jsonl"
NVCC = BUILD / "toolchains/cuda/nvidia/cu13/bin/nvcc"
CPUS = os.cpu_count() or 4
SANITIZER_PATTERN = re.compile(r"ERROR: AddressSanitizer|ERROR: LeakSanitizer|runtime error:")


def sanitizer_env(stream, extra=None):
    """Sanitizer reports from every child process land in <log>.asan.<pid>/.ubsan.<pid>, exit 86."""
    prefix = stream.name[: -len(".log")]
    for stale in Path(prefix).parent.glob(Path(prefix).name + ".*san.*"):
        stale.unlink()
    return {
        "ASAN_OPTIONS": f"detect_leaks=1:exitcode=86:log_path={prefix}.asan",
        "UBSAN_OPTIONS": f"print_stacktrace=1:exitcode=86:log_path={prefix}.ubsan",
        **(extra or {}),
    }


def sanitizer_reports(stream, output=""):
    prefix = Path(stream.name[: -len(".log")])
    files = sorted(prefix.parent.glob(prefix.name + ".*san.*"))
    return len(files) + (1 if SANITIZER_PATTERN.search(output) else 0)
EMULATION_CONFIGS = [(d, o, s) for d in ("cuda", "metal") for o in ("forward", "reverse") for s in ("inorder", "lane1-late", "lane0-late")]


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


class Result:
    def __init__(self, status, summary, **metrics):
        self.status, self.summary, self.metrics = status, summary, metrics


class Stage:
    """A named check. `run(log)` returns a Result; cpus is its share of the parallel budget."""

    def __init__(self, name, run, cpus=1, estimate=60, needs=()):
        self.name, self.run, self.cpus, self.estimate, self.needs = name, run, min(cpus, CPUS), estimate, set(needs)


class Runner:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.lock = threading.Condition()
        self.free = CPUS

    def log_path(self, name):
        return self.run_dir / "logs" / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".log")

    def execute(self, stage):
        blocked = [need for need in stage.needs if self.results.get(need, Result("SKIP", "")).status not in ("PASS", "WARN")]
        if blocked:
            return Result("SKIP", "blocked by " + ", ".join(sorted(blocked)))
        start = time.time()
        try:  # a harness defect must fail its stage, not hide it or stall the scheduler
            with open(self.log_path(stage.name), "w") as stream:
                result = stage.run(stream)
            if not isinstance(result, Result):
                raise TypeError(f"stage returned {result!r}")
        except Exception as error:
            result = Result("FAIL", f"harness error: {error!r}")
        result.metrics["seconds"] = round(time.time() - start, 1)
        try:
            print(f"[{time.strftime('%H:%M:%S')}] {result.status:5} {stage.name} ({result.metrics['seconds']}s) {result.summary}", flush=True)
        except OSError:
            pass
        return result

    def run_sequential(self, stages):
        for stage in stages:
            self.results[stage.name] = self.execute(stage)

    def run_parallel(self, stages):
        """Longest-first list scheduling under the CPU budget."""
        pending = sorted(stages, key=lambda stage: -stage.estimate)
        threads = []

        def worker(stage):
            result = Result("FAIL", "harness error: stage thread died")
            try:
                result = self.execute(stage)
            finally:
                with self.lock:
                    self.results[stage.name] = result
                    self.free += stage.cpus
                    self.lock.notify_all()

        with self.lock:
            while pending:
                stage = next((s for s in pending if s.cpus <= self.free), None)
                if stage is None:
                    self.lock.wait()
                    continue
                pending.remove(stage)
                self.free -= stage.cpus
                thread = threading.Thread(target=worker, args=(stage,))
                thread.start()
                threads.append(thread)
        for thread in threads:
            thread.join()
        for stage in stages:
            self.results.setdefault(stage.name, Result("FAIL", "harness error: no result recorded"))


def sh(stream, command, cwd=ROOT, env=None, timeout=7200, check=False):
    """Runs a command, appending its combined output to the stage log; returns (code, output)."""
    stream.write(f"$ {' '.join(map(str, command))}\n")
    stream.flush()
    # A new session lets a timeout kill grandchildren too (ctest's test binaries, services, npm).
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        code = -1
        output = (output or "") + f"\nTIMEOUT after {timeout}s\n"
    stream.write(output + f"\n[exit {code}]\n")
    stream.flush()
    if check and code != 0:
        raise RuntimeError(f"{command[0]} exited {code}")
    return code, output


def gtest_report(path):
    """Returns {test name: (result, device)} from a gtest JSON report."""
    report = json.loads(Path(path).read_text())
    tests = {}
    for suite in report["testsuites"]:
        for test in suite["testsuite"]:
            status = "FAILED" if test.get("failures") else test.get("result", "COMPLETED")
            tests[f"{suite['name']}.{test['name']}"] = (status, test.get("device", ""))
    return tests


# ---------------------------------------------------------------------------------------------
# Preparation and builds


def prepare(ref):
    def run(stream):
        sh(stream, ["git", "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main"], check=True)
        sha = git("rev-parse", ref)
        if not (SRC / ".git").exists():
            sh(stream, ["git", "worktree", "prune"])
            sh(stream, ["git", "worktree", "add", "--detach", SRC, sha], check=True)
        sh(stream, ["git", "checkout", "-q", "--force", "--detach", sha], cwd=SRC, check=True)
        sh(stream, ["git", "clean", "-fdxq", "-e", "desktop/node_modules"], cwd=SRC, check=True)
        sh(stream, ["git", "submodule", "update", "--init", "--recursive", "-q"], cwd=SRC, check=True)
        sh(stream, ["bash", HARNESS / "setup.sh"], check=True, timeout=3600)
        sh(stream, [sys.executable, HARNESS / "scenarios.py", SRC, BUILD / "scenarios"], check=True)
        return Result("PASS", f"{git('log', '--oneline', '-1', sha)}", sha=sha)

    return Stage("prepare", run, cpus=CPUS)


SANITIZER_FLAGS = "-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"
BUILDS = {
    "release": (["-DCMAKE_BUILD_TYPE=Release", "-DDAILY_BACKEND=cpu"], ["007SolverTests", "solver_service", "007SolverBenchmark", "daily_cpu_determinism"]),
    "emulation": (["-DCMAKE_BUILD_TYPE=Release", "-DDAILY_BACKEND=emulation"], ["007SolverTests", "solver_service", "daily_fingerprint", "daily_parity", "daily_planstats"]),
    "sanitize": (
        [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DDAILY_BACKEND=emulation",
            f"-DCMAKE_CXX_FLAGS={SANITIZER_FLAGS}",
            "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
        ],
        ["007SolverTests", "solver_service"],
    ),
    "cuda": (["-DCMAKE_BUILD_TYPE=Release", "-DDAILY_BACKEND=cuda", f"-DCMAKE_CUDA_COMPILER={NVCC}"], ["007SolverTests", "solver_service"]),
}


def build(name):
    options, targets = BUILDS[name]

    def run(stream):
        directory = BUILD / name
        code, _ = sh(
            stream,
            ["cmake", "-S", SRC, "-B", directory, "-G", "Ninja", f"-DCMAKE_PROJECT_007Solver_INCLUDE={HARNESS / 'cmake/daily.cmake'}", *options],
        )
        if code:
            return Result("FAIL", "configure failed")
        # Force the CUDA objects to rebuild so ptxas statistics are always in this log.
        if name == "cuda":
            for stale in directory.glob("**/CudaExecutor.cu.o"):
                stale.unlink()
        code, output = sh(stream, ["cmake", "--build", directory, "--target", *targets])
        if code:
            return Result("FAIL", "build failed")
        warnings = len(re.findall(r"\bwarning\b", output))
        metrics = {"warnings": warnings}
        status = "WARN" if warnings else "PASS"
        summary = f"{len(targets)} targets" + (f", {warnings} compiler warnings" if warnings else "")
        if name == "cuda":
            kernels = parse_ptxas(output)
            metrics["kernels"] = kernels
            spills = {key: value for key, value in kernels.items() if value["spill_bytes"]}
            if not kernels:
                return Result("FAIL", "no ptxas statistics found", **metrics)
            if spills:
                status, summary = "WARN", summary + f"; register spills in {sorted(spills)}"
            summary += "; " + ", ".join(sorted({key.split('@')[1] for key in kernels})) + f"; max {max(v['registers'] for v in kernels.values())} registers"
        return Result(status, summary, **metrics)

    return Stage(f"build-{name}", run, cpus=CPUS, needs={"prepare"})


def parse_ptxas(output):
    kernels, current = {}, None
    for line in output.splitlines():
        entry = re.search(r"Compiling entry function '_Z(\d+)(\w+)' for '(sm_\d+)'", line)
        if entry:  # Itanium mangling: the name length precedes the name
            current = f"{entry.group(2)[: int(entry.group(1))]}@{entry.group(3)}"
            kernels[current] = {"registers": 0, "spill_bytes": 0}
            continue
        if current and (spill := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)):
            kernels[current]["spill_bytes"] = int(spill.group(1)) + int(spill.group(2))
        if current and (used := re.search(r"Used (\d+) registers", line)):
            kernels[current]["registers"] = int(used.group(1))
    return kernels


# ---------------------------------------------------------------------------------------------
# Test stages


def cpu_ctest():
    def run(stream):
        (BUILD / "release/solver-test-results.json").unlink(missing_ok=True)
        code, _ = sh(stream, ["ctest", "--test-dir", BUILD / "release", "--output-on-failure"], timeout=1800)
        tests = gtest_report(BUILD / "release/solver-test-results.json")
        passed = sum(status == "COMPLETED" for status, _ in tests.values())
        skipped = [name for name, (status, _) in tests.items() if status == "SKIPPED"]
        unexpected = [name for name in skipped if "Gpu" not in name]
        if code or unexpected:
            return Result("FAIL", f"ctest exit {code}; unexpected skips {unexpected}")
        return Result("PASS", f"{passed} gtest cases + CLI e2e passed; {len(skipped)} GPU cases skipped as expected", gtest=len(tests))

    return Stage("cpu-ctest", run, cpus=4, estimate=40, needs={"build-release"})


def cpu_determinism():
    def run(stream):
        failures = []
        for fixture in ("weighted-flop", "raise-flop", "backend-parity"):
            code, _ = sh(stream, [BUILD / "release/daily_cpu_determinism", SRC / f"tests/fixtures/{fixture}.json", 40, 1, 2, 3, 4])
            if code:
                failures.append(fixture)
        if failures:
            return Result("FAIL", f"worker count changes training state: {failures}")
        return Result("PASS", "workers 1-4 bitwise identical on 3 fixtures (40 updates)")

    return Stage("cpu-determinism", run, cpus=4, estimate=90, needs={"build-release"})


def protocol(build_name, env=None, cpus=2, estimate=20):
    def run(stream):
        environment = sanitizer_env(stream, env) if build_name == "sanitize" else env
        code, output = sh(
            stream,
            [sys.executable, HARNESS / "protocol_test.py", BUILD / build_name / "solver/solver_service", SRC / "tests/fixtures", "-v"],
            env=environment,
        )
        if build_name == "sanitize" and sanitizer_reports(stream, output):
            return Result("FAIL", f"{sanitizer_reports(stream, output)} sanitizer reports; see {Path(stream.name).name[:-4]}.*san.*")
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        if code:
            return Result("FAIL", tail)
        return Result("PASS", tail)

    return Stage(f"protocol-{build_name}", run, cpus=cpus, estimate=estimate, needs={f"build-{build_name}"})


def sanitize_cpu():
    def run(stream):
        env = sanitizer_env(stream, {"SOLVER_GPU_EMULATION": "off"})
        report = stream.name + ".gtest.json"
        code1, out1 = sh(stream, [BUILD / "sanitize/007SolverTests", f"--gtest_output=json:{report}"], env=env)
        code2, out2 = sh(stream, [sys.executable, SRC / "tests/cli_e2e.py", BUILD / "sanitize/solver/solver_service"], env=env)
        if sanitizer_reports(stream, out1 + out2):
            return Result("FAIL", f"{sanitizer_reports(stream, out1 + out2)} sanitizer reports")
        if code1 or code2:
            return Result("FAIL", f"gtest exit {code1}, cli exit {code2}")
        return Result("PASS", f"{len(gtest_report(report))} gtest cases + CLI e2e clean under ASan/UBSan/LSan")

    return Stage("sanitize-cpu", run, cpus=4, estimate=240, needs={"build-sanitize"})


def cuda_fallback():
    def run(stream):
        directory = BUILD / "cuda"
        code, _ = sh(stream, ["ctest", "--test-dir", directory, "--output-on-failure"], timeout=1800)
        forced_code, forced = sh(stream, [directory / "solver/solver_service", SRC / "tests/fixtures/weighted-flop.json", "--device=gpu"])
        auto_code, auto = sh(stream, [directory / "solver/solver_service", SRC / "tests/fixtures/weighted-flop.json", "--device=auto"])
        problems = []
        if code:
            problems.append(f"ctest exit {code}")
        if forced_code != 1 or '"event":"failed"' not in forced:
            problems.append("forced GPU did not fail cleanly")
        if auto_code != 0 or "Device: CPU" not in auto:
            problems.append("auto did not fall back to CPU")
        if problems:
            return Result("FAIL", "; ".join(problems))
        return Result("PASS", "no-driver CUDA build: tests pass, auto uses CPU, forced GPU fails with JSON + exit 1")

    return Stage("cuda-no-driver-fallback", run, cpus=4, estimate=40, needs={"build-cuda"})


def emulated_gtest(dialect, estimate):
    def run(stream):
        report = stream.name + ".gtest.json"
        code, _ = sh(
            stream,
            [BUILD / "emulation/007SolverTests", "--gtest_filter=*Gpu*:BackendParity*", f"--gtest_output=json:{report}"],
            env={"SOLVER_GPU_EMULATION": dialect},
        )
        tests = gtest_report(report)
        failed = [name for name, (status, _) in tests.items() if status != "COMPLETED"]
        not_emulated = [name for name, (_, device) in tests.items() if "emulation" not in device]
        if code or failed or not_emulated or len(tests) < 3:
            return Result("FAIL", f"exit {code}; failed {failed}; not emulated {not_emulated}")
        return Result("PASS", f"{len(tests)} GPU gtest cases passed on {dialect} kernels")

    return Stage(f"emu-gtest-{dialect}", run, cpus=1, estimate=estimate, needs={"build-emulation"})


def emulated_cli(dialect, estimate):
    def run(stream):
        code, output = sh(
            stream, [sys.executable, SRC / "tests/cli_e2e.py", BUILD / "emulation/solver/solver_service"], env={"SOLVER_GPU_EMULATION": dialect}
        )
        return Result("FAIL" if code else "PASS", output.strip().splitlines()[-1] if output.strip() else "")

    return Stage(f"emu-cli-{dialect}", run, cpus=2, estimate=estimate, needs={"build-emulation"})


RACE_WORKLOADS = {
    # name: (scenario path, updates, configurations)
    "backend-parity": (SRC / "tests/fixtures/backend-parity.json", 12, EMULATION_CONFIGS),
    # Single-lane plans: lane schedules would repeat inorder.
    "weighted-flop": (SRC / "tests/fixtures/weighted-flop.json", 8, [c for c in EMULATION_CONFIGS if c[2] == "inorder"]),
    "raise-flop": (SRC / "tests/fixtures/raise-flop.json", 8, [c for c in EMULATION_CONFIGS if c[2] == "inorder"]),
    "btn-bb-srp-dry": (BUILD / "scenarios/btn-bb-srp-dry.json", 4, EMULATION_CONFIGS[:6] + [EMULATION_CONFIGS[6], EMULATION_CONFIGS[10]]),
    "utg-bb-wide": (SRC / "tests/fixtures/utg-bb-wide.json", 2, EMULATION_CONFIGS[:6] + [EMULATION_CONFIGS[6]]),
}
RACE_ESTIMATES = {"backend-parity": 60, "weighted-flop": 40, "raise-flop": 40, "btn-bb-srp-dry": 150, "utg-bb-wide": 400}


def race(workload, config):
    path, updates, _ = RACE_WORKLOADS[workload]
    dialect, order, schedule = config

    def run(stream):
        env = {"SOLVER_GPU_EMULATION": dialect, "SOLVER_GPU_EMULATION_ORDER": order, "SOLVER_GPU_EMULATION_SCHEDULE": schedule, "OMP_NUM_THREADS": "1"}
        code, output = sh(stream, [BUILD / "emulation/daily_fingerprint", path, updates], env=env)
        match = re.search(r"regrets (\w+) sums (\w+) stamps (\w+) gpuExpl (\w+) \(([^)]*)\) nans (\d+)", output)
        if code or not match:
            return Result("FAIL", f"exit {code}")
        fingerprint = "/".join(match.group(1, 2, 3, 4))
        if int(match.group(6)):
            return Result("FAIL", f"{match.group(6)} NaNs", fingerprint=fingerprint)
        return Result("PASS", fingerprint, fingerprint=fingerprint, exploitability=float(match.group(5)))

    return Stage(f"emu-race:{workload}:{dialect}-{order}-{schedule}", run, cpus=1, estimate=RACE_ESTIMATES[workload], needs={"build-emulation"})


def race_verdicts(results):
    """Race-free kernels and complete lane sync give one fingerprint per workload."""
    verdicts = {}
    for workload in RACE_WORKLOADS:
        runs = {name: result for name, result in results.items() if name.startswith(f"emu-race:{workload}:")}
        prints = {result.metrics.get("fingerprint") for result in runs.values()}
        if not runs:
            continue
        if any(result.status != "PASS" for result in runs.values()):
            verdicts[f"emu-race-verdict:{workload}"] = Result("FAIL", "a configuration failed")
        elif len(prints) != 1:
            verdicts[f"emu-race-verdict:{workload}"] = Result("FAIL", f"{len(prints)} distinct fingerprints across {len(runs)} configurations")
        else:
            verdicts[f"emu-race-verdict:{workload}"] = Result("PASS", f"{len(runs)} configurations bitwise identical")
    return verdicts


def parity(scenario, dialect, estimate):
    def run(stream):
        code, output = sh(
            stream,
            [BUILD / "emulation/daily_parity", BUILD / f"scenarios/{scenario}.json", 4, 1],
            env={"SOLVER_GPU_EMULATION": dialect, "OMP_NUM_THREADS": "1"},
        )
        worst = max((float(value) for value in re.findall(r"worst ([0-9.e+-]+)", output)), default=None)
        if code or worst is None:
            return Result("FAIL", f"exit {code}; " + (output.strip().splitlines() or [""])[-1][:200])
        return Result("PASS", f"4 lockstep updates within tolerance; worst {worst:.1e} of node scale", worst=worst)

    return Stage(f"emu-parity:{scenario}:{dialect}", run, cpus=1, estimate=estimate, needs={"build-emulation"})


def plan_coverage():
    def run(stream):
        paths = [SRC / f"tests/fixtures/{name}.json" for name in ("backend-parity", "weighted-flop", "raise-flop", "utg-bb-wide")]
        paths += sorted((BUILD / "scenarios").glob("*.json"))
        coverage, failures = {}, []
        for path in paths:
            code, output = sh(stream, [BUILD / "emulation/daily_planstats", path])
            match = re.search(r"hands (\d+)/(\d+) .* lane-1 passes (\d+), fork (\d+) wait (\d+) join (\d+)", output)
            if not match:
                return Result("FAIL", f"planstats failed for {path.name}")
            coverage[path.stem] = dict(zip(("hands0", "hands1", "lane1", "fork", "wait", "join"), map(int, match.groups())))
            coverage[path.stem]["unorderedPairs"] = sum(map(int, re.findall(r"(\d+) unordered pass pairs", output)))
            if code:  # lane conflicts, capture errors or an undetected mutation (see LaneCheck.h)
                failures.append(path.stem)
        if failures:
            return Result("FAIL", f"two-lane schedule check failed for {failures}", coverage=coverage)
        # tests/README.md requires several GPU batches; two-lane batching implies it.
        if coverage["backend-parity"]["lane1"] == 0:
            return Result("FAIL", "backend-parity no longer produces two-lane batches", coverage=coverage)
        narrow = [name for name in coverage if name.startswith("btn-bb-") and max(coverage[name]["hands0"], coverage[name]["hands1"]) <= 256]
        if narrow:
            return Result("FAIL", f"parity scenarios no longer exceed 256 hands: {narrow}", coverage=coverage)
        pairs = sum(value["unorderedPairs"] for value in coverage.values())
        return Result("PASS", f"{len(coverage)} plans: no lane conflicts in {pairs} unordered pass pairs; mutations detected", coverage=coverage)

    return Stage("gpu-plan-coverage", run, cpus=1, estimate=20, needs={"build-emulation"})


def sanitize_gpu(dialect, estimate):
    def run(stream):
        report = stream.name + ".gtest.json"
        code, output = sh(
            stream,
            [BUILD / "sanitize/007SolverTests", "--gtest_filter=BackendParity*", f"--gtest_output=json:{report}"],
            env=sanitizer_env(stream, {"SOLVER_GPU_EMULATION": dialect}),
            timeout=10800,
        )
        if sanitizer_reports(stream, output):
            return Result("FAIL", f"{sanitizer_reports(stream, output)} sanitizer reports")
        tests = gtest_report(report) if Path(report).exists() else {}
        failed = [name for name, (status, _) in tests.items() if status != "COMPLETED"]
        if code or failed or not tests:
            return Result("FAIL", f"exit {code}; failed {failed}")
        return Result("PASS", f"{len(tests)} BackendParity case(s) clean under ASan/UBSan on {dialect} kernels")

    return Stage(f"sanitize-gpu-{dialect}", run, cpus=1, estimate=estimate, needs={"build-sanitize"})


def desktop_check():
    def run(stream):
        desktop = SRC / "desktop"
        for command in (["npm", "ci", "--no-audit", "--no-fund"], ["npm", "run", "check"], ["npm", "run", "build:ui"]):
            code, output = sh(stream, command, cwd=desktop, timeout=1800)
            if code:
                return Result("FAIL", f"{' '.join(command)} exit {code}")
        return Result("PASS", "lint, typecheck, prettier and UI build")

    return Stage("desktop-check", run, cpus=2, estimate=90, needs={"prepare"})


def desktop_cargo():
    def run(stream):
        binaries = SRC / "desktop/src-tauri/binaries"
        binaries.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUILD / "release/solver/solver_service", binaries / "solver-service-x86_64-unknown-linux-gnu")
        code, output = sh(
            stream,
            ["cargo", "check", "--manifest-path", SRC / "desktop/src-tauri/Cargo.toml", "--locked"],
            env={"CARGO_TARGET_DIR": str(BUILD / "cargo-target")},
            timeout=3600,
        )
        warnings = len(re.findall(r"^warning", output, re.MULTILINE))
        if code:
            return Result("FAIL", f"cargo check exit {code}")
        return Result("WARN" if warnings else "PASS", f"cargo check --locked, {warnings} warnings")

    return Stage("desktop-cargo-check", run, cpus=4, estimate=150, needs={"build-release"})


def pre_commit():
    def run(stream):
        code, output = sh(stream, ["pre-commit", "run", "--all-files"], cwd=SRC, timeout=3600)
        changed = git("status", "--porcelain", cwd=SRC)
        if changed:
            sh(stream, ["git", "diff", "--stat"], cwd=SRC)
            sh(stream, ["git", "checkout", "-q", "--", "."], cwd=SRC)
        if code or changed:
            failing = re.findall(r"^(.+?)\.+Failed$", output, re.MULTILINE)
            return Result("FAIL", f"hooks failed {failing}; files changed: {len(changed.splitlines())}")
        return Result("PASS", "all hooks pass without changes")

    return Stage("repo-pre-commit", run, cpus=CPUS, estimate=60, needs={"desktop-check"})


def benchmark():
    def run(stream):
        report = stream.name.replace(".log", ".json")
        code, output = sh(stream, [BUILD / "release/007SolverBenchmark", "--device=cpu", f"--report={report}"], cwd=SRC, timeout=7200)
        if code or not Path(report).exists():
            return Result("FAIL", f"exit {code}")
        data = json.loads(Path(report).read_text())
        rate = data["completed_iterations"] / data["training_loop_seconds"]
        metrics = {
            "updates_per_second": round(rate, 3),
            "accuracy_percent": data["trained"]["accuracyPercent"],
            "target_reached": data["target_reached"],
            "estimate_peak_bytes": data["tree_estimate"]["peak_bytes"],
        }
        if data.get("status") != "passed":
            return Result("FAIL", f"benchmark status {data.get('status')}", **metrics)
        host = host_info()
        previous = [
            entry["benchmark"]["updates_per_second"]
            for entry in read_history()
            if entry.get("benchmark", {}).get("updates_per_second") and entry.get("host") == host
        ][-5:]
        summary = f"{rate:.2f} updates/s, accuracy {metrics['accuracy_percent']:.4f}%"
        if len(previous) >= 2 and rate < 0.85 * statistics.median(previous):
            return Result("WARN", summary + f"; >15% slower than recent median {statistics.median(previous):.2f}", **metrics)
        return Result("PASS", summary + (f"; recent median {statistics.median(previous):.2f}" if previous else "; first measurement"), **metrics)

    return Stage("perf-benchmark-cpu", run, cpus=CPUS, estimate=700, needs={"build-release"})


# ---------------------------------------------------------------------------------------------


def host_info():
    model = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "unknown")
    return {"cpu": model, "cpus": CPUS}


def read_history():
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def all_stages(quick, ref):
    tests = [
        cpu_ctest(),
        cpu_determinism(),
        protocol("release"),
        sanitize_cpu(),
        protocol("sanitize", env={"ASAN_OPTIONS": "detect_leaks=1", "SOLVER_GPU_EMULATION": "off"}, cpus=2, estimate=60),
        cuda_fallback(),
        emulated_cli("cuda", 120),
        emulated_cli("metal", 200),
        plan_coverage(),
        desktop_check(),
        desktop_cargo(),
    ]
    tests += [race(workload, config) for workload, (_, _, configs) in RACE_WORKLOADS.items() for config in configs if not quick or workload != "utg-bb-wide"]
    tests += [parity(scenario, "cuda", 200) for scenario in ("btn-bb-srp-dry", "btn-bb-srp-paired", "btn-bb-srp-monotone")]
    tests += [parity("btn-bb-srp-dry", "metal", 300)]
    if not quick:
        tests += [emulated_gtest("cuda", 900), emulated_gtest("metal", 1800), sanitize_gpu("cuda", 1500), sanitize_gpu("metal", 2500)]
    final = [pre_commit()] + ([] if quick else [benchmark()])
    return [prepare(ref)] + [build(name) for name in BUILDS], tests, final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="origin/main", help="revision to test (default origin/main)")
    parser.add_argument("--check-new", action="store_true", help="print RUN or SKIP and exit")
    parser.add_argument("--only", help="regex of stage names to run (preparation and builds always run)")
    parser.add_argument("--quick", action="store_true", help="skip the slowest stages; never recorded (for harness development)")
    parser.add_argument("--no-record", action="store_true", help="do not update state.json or history.jsonl")
    args = parser.parse_args()

    if args.check_new:
        subprocess.run(["git", "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main"], cwd=ROOT, check=True)
        current, last = git("rev-parse", args.ref), load_state().get("lastTestedMainSha")
        print(f"SKIP {current} already tested" if current == last else f"RUN {last or 'none'} -> {current}")
        return 0

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner = Runner(BUILD / "runs" / stamp)
    setup, tests, final = all_stages(args.quick, args.ref)
    if args.only:
        pattern = re.compile(args.only)
        tests = [stage for stage in tests if pattern.search(stage.name)]
        final = [stage for stage in final if pattern.search(stage.name)]
    started = time.time()
    # The harness revision that actually runs; later commits during the run must not relabel it.
    harness_sha = git("rev-parse", "HEAD")
    harness_dirty = bool(git("status", "--porcelain", "--", "tests/daily", ":!tests/daily/state.json", ":!tests/daily/history.jsonl"))
    runner.run_sequential(setup)
    runner.run_parallel(tests)
    runner.results.update(race_verdicts(runner.results))
    runner.run_sequential(final)

    results = runner.results
    sha = results["prepare"].metrics.get("sha", "unknown")
    counts = {status: sum(r.status == status for r in results.values()) for status in ("PASS", "WARN", "FAIL", "SKIP")}
    visible = {name: r for name, r in results.items() if not name.startswith("emu-race:") or r.status != "PASS"}
    lines = [
        f"# Daily run {stamp}",
        "",
        f"- Tested: `{git('log', '--oneline', '-1', sha) if sha != 'unknown' else sha}`",
        f"- Harness: `{harness_sha[:7]}`{' with uncommitted changes' if harness_dirty else ''}",
        f"- Host: {host_info()['cpu']}, {CPUS} CPUs; {subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout.splitlines()[0]}; wall time {(time.time() - started) / 60:.1f} min",
        f"- Stages: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail, {counts['SKIP']} skip"
        f" ({sum(name.startswith('emu-race:') for name in results)} race configurations folded into verdicts)",
        "",
        "| Stage | Status | Seconds | Summary |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {r.status} | {r.metrics.get('seconds', '')} | {r.summary} |" for name, r in visible.items()]
    summary = "\n".join(lines) + "\n"
    (runner.run_dir / "summary.md").write_text(summary)
    (runner.run_dir / "summary.json").write_text(
        json.dumps({name: {"status": r.status, "summary": r.summary, **r.metrics} for name, r in results.items()}, indent=1)
    )
    print("\n" + summary)
    print(f"Logs: {runner.run_dir / 'logs'}")

    # Only complete runs mark a commit as tested; --check-new skips tested commits.
    if not (args.no_record or args.only or args.quick or args.ref != "origin/main") and sha != "unknown":
        entry = {
            "run": stamp,
            "mainSha": sha,
            "harnessSha": harness_sha,
            "harnessDirty": harness_dirty,
            "host": host_info(),
            "counts": counts,
            "failures": sorted(name for name, r in results.items() if r.status == "FAIL"),
            "warnings": sorted(name for name, r in results.items() if r.status == "WARN"),
            "stageSeconds": {name: r.metrics.get("seconds") for name, r in results.items() if not name.startswith("emu-race:")},
        }
        if "perf-benchmark-cpu" in results:
            entry["benchmark"] = {k: v for k, v in results["perf-benchmark-cpu"].metrics.items() if k != "seconds"}
        with open(HISTORY, "a") as stream:
            stream.write(json.dumps(entry) + "\n")
        state = load_state()
        state.update({"lastTestedMainSha": sha, "lastRun": stamp, "lastRunCounts": counts})
        STATE.write_text(json.dumps(state, indent=2) + "\n")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
