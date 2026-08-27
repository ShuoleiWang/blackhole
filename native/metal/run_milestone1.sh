#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"

"${SCRIPT_DIR}/build.sh"
"${BUILD_DIR}/metal-dd-milestone1" \
  --metallib "${BUILD_DIR}/DoubleDouble.metallib" \
  --report "${BUILD_DIR}/milestone1-report.json" \
  --benchmark-count "${BENCHMARK_COUNT:-1048576}" \
  --benchmark-iterations "${BENCHMARK_ITERATIONS:-9}"
