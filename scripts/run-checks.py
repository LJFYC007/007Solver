"""Builds and runs the solver checks of the push hook (scripts/pre-push): CTest, then the GPU
benchmark within its time budget."""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

repository = Path(__file__).resolve().parent.parent
BENCHMARK_BUDGET_SECONDS = 120  # tests/README.md "Benchmark"


def run(command):
    # The build needs the MSVC environment and UTF-8 output for Ninja, as in scripts/deploy-server.ps1.
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vcvars = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-find", r"VC\Auxiliary\Build\vcvars64.bat"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() if vswhere.exists() else ""
    if not vcvars:
        sys.exit("The solver checks need Visual Studio's C++ tools (vcvars64.bat)")
    if not (repository / "build" / "release" / "CMakeCache.txt").exists():
        command = f"cmake --preset Release && {command}"
    subprocess.run(f'cmd /d /s /c ""{vcvars}" >nul && chcp 65001 >nul && {command}"', cwd=repository, check=True)


def main():
    # A server deploy (scripts/post-commit) builds the same tree and restarts the server: wait for it.
    kernel32 = ctypes.WinDLL("kernel32")
    mutex = kernel32.CreateMutexW(None, False, "Local\\007SolverDeployServer")
    kernel32.WaitForSingleObject(mutex, 0xFFFFFFFF)
    try:
        # Without a supported GPU, CTest's GPU cases and the GPU benchmark skip themselves.
        run("cmake --build --preset Release --target 007SolverTests 007SolverBenchmark"
            " && ctest --test-dir build/release --output-on-failure")
        start = time.monotonic()
        run("cmake --build --preset Release --target benchmark_gpu")
        seconds = time.monotonic() - start
        if seconds > BENCHMARK_BUDGET_SECONDS:
            sys.exit(f"benchmark_gpu took {seconds:.0f} s, beyond its {BENCHMARK_BUDGET_SECONDS} s budget "
                     "(other GPU work, such as web server solves, also slows it)")
    finally:
        kernel32.ReleaseMutex(mutex)
        kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
