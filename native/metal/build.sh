#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"
TOOLCHAIN_ID="com.apple.dt.toolchain.Metal.32023.883"

mkdir -p "${BUILD_DIR}/module-cache" "${BUILD_DIR}/metal-module-cache"

CLANG_MODULE_CACHE_PATH="${BUILD_DIR}/metal-module-cache" \
xcrun --toolchain "${TOOLCHAIN_ID}" -sdk macosx metal \
  -std=metal3.2 \
  -fmodules-cache-path="${BUILD_DIR}/metal-module-cache" \
  -fmetal-math-mode=safe \
  -fmetal-math-fp32-functions=precise \
  -ffp-contract=on \
  -c "${SCRIPT_DIR}/DoubleDouble.metal" \
  -o "${BUILD_DIR}/DoubleDouble.air"

xcrun --toolchain "${TOOLCHAIN_ID}" -sdk macosx metallib \
  "${BUILD_DIR}/DoubleDouble.air" \
  -o "${BUILD_DIR}/DoubleDouble.metallib"

xcrun --sdk macosx swiftc \
  -O \
  -whole-module-optimization \
  -module-cache-path "${BUILD_DIR}/module-cache" \
  "${SCRIPT_DIR}/Milestone1.swift" \
  -framework Metal \
  -o "${BUILD_DIR}/metal-dd-milestone1"

echo "built ${BUILD_DIR}/metal-dd-milestone1"
echo "metallib sha256: $(shasum -a 256 "${BUILD_DIR}/DoubleDouble.metallib" | awk '{print $1}')"
