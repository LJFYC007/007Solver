#!/usr/bin/env bash
# Idempotent bootstrap for a fresh Ubuntu 24.04 cloud container (see tests/daily/README.md).
# Installs only what is missing; safe to rerun. Toolchains go under build/daily/toolchains.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOOLCHAINS="$ROOT/build/daily/toolchains"
mkdir -p "$TOOLCHAINS"

need() { command -v "$1" > /dev/null || { echo "setup: missing $1 (expected in the base image)" >&2; exit 1; }; }
for tool in git cmake ninja g++ python3 node npm cargo rustfmt clang-format pip; do need "$tool"; done

# Tauri's Linux build dependencies for `cargo check` of desktop/src-tauri.
if ! pkg-config --exists webkit2gtk-4.1 2> /dev/null; then
    echo "setup: installing WebKitGTK development packages"
    apt-get update -qq > /dev/null 2>&1 || true # an unreachable PPA must not stop the Ubuntu archive
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libwebkit2gtk-4.1-dev librsvg2-dev libxdo-dev > /dev/null
fi

# NVIDIA's CUDA redistributables are blocked by the network policy; PyPI wheels carry nvcc.
# The repository requires Toolkit 12.8+; 13.4 still targets sm_86/89/120.
CUDA="$TOOLCHAINS/cuda/nvidia/cu13"
if [ ! -x "$CUDA/bin/nvcc" ]; then
    echo "setup: installing CUDA 13.4 toolchain wheels"
    pip install --quiet --disable-pip-version-check --target "$TOOLCHAINS/cuda" \
        "nvidia-cuda-nvcc==13.4.92" "nvidia-cuda-runtime==13.4.92" "nvidia-cuda-cccl==13.3.4.3.1" 2>&1 | grep -v "Running pip as the 'root' user" || true
fi
# FindCUDAToolkit looks for the unversioned runtime name, which the wheel omits.
ln -sf libcudart.so.13 "$CUDA/lib/libcudart.so"
"$CUDA/bin/nvcc" --version | tail -1

if ! command -v pre-commit > /dev/null; then
    pip install --quiet --disable-pip-version-check "pre-commit==4.6.2" 2>&1 | grep -v "Running pip as the 'root' user" || true
fi
echo "setup: ok"
