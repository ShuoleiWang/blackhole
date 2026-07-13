#!/usr/bin/env bash

set -euo pipefail

readonly SOURCE_URL='https://cdn.sci.esa.int/documents/33580/35361/Gaia_EDR3_flux_cartesian_16k.png/f116e989-fc70-0dac-e453-f1f2141420be?t=1606986368242&version=1.0'
readonly EXPECTED_SHA256='10a372d392e9493f6333b7f782e6a973742b71a8da8adc926e0129807462b7e9'
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DESTINATION="${PROJECT_DIR}/assets/gaia-edr3-16k.png"
readonly PARTIAL="${DESTINATION}.part"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    printf 'Neither shasum nor sha256sum is available.\n' >&2
    return 1
  fi
}

verify_file() {
  local file="$1"
  local actual
  actual="$(sha256_file "${file}")"
  if [[ "${actual}" != "${EXPECTED_SHA256}" ]]; then
    printf 'SHA-256 mismatch for %s\nexpected: %s\nactual:   %s\n' \
      "${file}" "${EXPECTED_SHA256}" "${actual}" >&2
    return 1
  fi
}

if [[ -f "${DESTINATION}" ]]; then
  if verify_file "${DESTINATION}"; then
    printf 'Verified existing 16000x8000 Gaia sky: %s\n' "${DESTINATION}"
    exit 0
  fi
  printf 'Removing invalid local asset before downloading again.\n' >&2
  rm -f "${DESTINATION}"
fi

mkdir -p "$(dirname "${DESTINATION}")"
printf 'Downloading the official ESA/Gaia 16000x8000 sky (about 236 MiB)...\n'
curl \
  --fail \
  --location \
  --continue-at - \
  --retry 3 \
  --retry-delay 2 \
  --output "${PARTIAL}" \
  "${SOURCE_URL}"

verify_file "${PARTIAL}"
mv -f "${PARTIAL}" "${DESTINATION}"
printf 'Verified and installed: %s\n' "${DESTINATION}"
