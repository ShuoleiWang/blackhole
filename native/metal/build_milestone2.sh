#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"
TOOLCHAIN_ID="com.apple.dt.toolchain.Metal.32023.883"

mkdir -p \
  "${BUILD_DIR}/milestone2-module-cache" \
  "${BUILD_DIR}/milestone2-metal-cache"

CLANG_MODULE_CACHE_PATH="${BUILD_DIR}/milestone2-metal-cache" \
xcrun --toolchain "${TOOLCHAIN_ID}" -sdk macosx metal \
  -std=metal3.2 \
  -fmodules-cache-path="${BUILD_DIR}/milestone2-metal-cache" \
  -fmetal-math-mode=safe \
  -fmetal-math-fp32-functions=precise \
  -ffp-contract=on \
  -c "${SCRIPT_DIR}/Milestone2.metal" \
  -o "${BUILD_DIR}/Milestone2.air"

xcrun --toolchain "${TOOLCHAIN_ID}" -sdk macosx metallib \
  "${BUILD_DIR}/Milestone2.air" \
  -o "${BUILD_DIR}/Milestone2.metallib"

xcrun --sdk macosx swiftc \
  -O \
  -whole-module-optimization \
  -module-cache-path "${BUILD_DIR}/milestone2-module-cache" \
  "${SCRIPT_DIR}/Milestone2.swift" \
  -framework Metal \
  -o "${BUILD_DIR}/metal-dd-milestone2"

echo "built ${BUILD_DIR}/metal-dd-milestone2"
echo "metallib sha256: $(shasum -a 256 "${BUILD_DIR}/Milestone2.metallib" | awk '{print $1}')"
