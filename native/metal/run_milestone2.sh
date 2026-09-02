#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"
CORPUS_PATH="${BUILD_DIR}/milestone2-corpus.json"
CPU_BENCHMARK_PATH="${BUILD_DIR}/milestone2-cpu-benchmark.json"

"${SCRIPT_DIR}/build_milestone2.sh"

cd "${REPOSITORY_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python3 \
  "${SCRIPT_DIR}/generate_milestone2_corpus.py" \
  "${CORPUS_PATH}" \
  --cpu-benchmark-output "${CPU_BENCHMARK_PATH}" \
  --real-count "${REAL_KERR_COUNT:-2048}" \
  --adversarial-count "${ADVERSARIAL_COUNT:-2048}"

"${BUILD_DIR}/metal-dd-milestone2" \
  --metallib "${BUILD_DIR}/Milestone2.metallib" \
  --corpus "${CORPUS_PATH}" \
  --cpu-benchmark "${CPU_BENCHMARK_PATH}" \
  --report "${BUILD_DIR}/milestone2-report.json" \
  --benchmark-count "${BENCHMARK_COUNT:-32768}" \
  --benchmark-iterations "${BENCHMARK_ITERATIONS:-7}"
