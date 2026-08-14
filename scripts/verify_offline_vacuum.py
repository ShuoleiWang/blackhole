#!/usr/bin/env python3
"""Independently verify an offline vacuum spectral product.

The output manifest is validated against its own versioned schema and bound to
an independently validated ``blackhole.nr-transfer-map/v1`` input.  The
verifier authenticates every file, checks tile/state topology, enforces the
outcome encoding policy, and recomputes every Planck/Liouville float32 value.

Passing establishes conformance of this narrow vacuum endpoint compositor.  It
does not establish an NR spacetime solution, GRRT plasma transport, or an
OpenEXR scientific master.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
from typing import Any, NoReturn, Sequence

try:
    from scripts.verify_nr_contract import (
        ContractError,
        DEFAULT_SCHEMA as DEFAULT_TRANSFER_SCHEMA,
        audit_schema_dialect,
        load_json_strict,
        validate_contract as validate_transfer_contract,
        validate_json_schema,
    )
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from verify_nr_contract import (
        ContractError,
        DEFAULT_SCHEMA as DEFAULT_TRANSFER_SCHEMA,
        audit_schema_dialect,
        load_json_strict,
        validate_contract as validate_transfer_contract,
        validate_json_schema,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "offline-vacuum-spectral-v1.schema.json"

OUTPUT_SCHEMA = "blackhole.offline-vacuum-spectral/v1"
TRANSFER_SCHEMA = "blackhole.nr-transfer-map/v1"
COMPOSITOR_ALGORITHM = "vacuum-Liouville-specific-intensity/v1"
PLANCK_ENVIRONMENT_KIND = "isotropic-planck-blackbody"

TRANSFER_RECORD = struct.Struct("<7fBBH")
FLOAT32 = struct.Struct("<f")
UINT32 = struct.Struct("<I")
RECORD_BYTES = TRANSFER_RECORD.size
CANONICAL_NAN_FLOAT32 = bytes.fromhex("0000c07f")
POSITIVE_ZERO_FLOAT32 = b"\x00\x00\x00\x00"

PLANCK_CONSTANT_J_S = 6.62607015e-34
LIGHT_SPEED_M_S = 299_792_458.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
MAX_FLOAT64_LOG = math.log(float.fromhex("0x1.fffffffffffffp+1023"))


class OfflineVacuumContractError(ContractError):
    """A deterministic offline vacuum product validation failure."""


def fail(path: str, message: str) -> NoReturn:
    raise OfflineVacuumContractError(f"{path}: {message}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable_file(path: Path, label: str) -> bytes:
    """Read a regular, non-symlink file and reject a changed read snapshot."""

    try:
        if path.is_symlink():
            fail(label, "symlinked files are forbidden")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            fail(label, "expected a regular file")
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        fail(label, f"unable to read file: {error}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        fail(label, "file changed while it was being read")
    if len(payload) != before.st_size:
        fail(label, "file length changed while it was being read")
    return payload


def _normalized_uri(value: Any, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        fail(path, "URI must be a non-empty normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        fail(path, "URI must be normalized, relative, and traversal-free")
    return pure


def _read_relative_file(root: Path, uri: Any, path: str) -> bytes:
    """Read one artifact while rejecting symlinked path components."""

    pure = _normalized_uri(uri, path)
    if root.is_symlink():
        fail(path, "a symlinked artifact root is forbidden")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            fail(path, "symlinked artifacts or path components are forbidden")
    try:
        root_resolved = root.resolve(strict=True)
        candidate = cursor.resolve(strict=True)
        candidate.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        fail(path, f"artifact does not resolve inside its dataset root: {error}")
    return _read_stable_file(candidate, path)


def _validate_hashed_artifact(
    root: Path,
    artifact: dict[str, Any],
    path: str,
) -> bytes:
    payload = _read_relative_file(root, artifact["uri"], f"{path}.uri")
    if len(payload) != artifact["byteLength"]:
        fail(
            f"{path}.byteLength",
            f"declares {artifact['byteLength']} bytes, stored file has {len(payload)}",
        )
    actual_hash = _sha256_bytes(payload)
    if actual_hash != artifact["sha256"]:
        fail(
            f"{path}.sha256",
            f"hash mismatch: expected {artifact['sha256']}, got {actual_hash}",
        )
    return payload


def _load_manifest_snapshot(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.name != "manifest.json":
        fail(label, "v1 manifest file must be named 'manifest.json'")
    payload = _read_stable_file(path, label)
    parsed = load_json_strict(path)
    if _read_stable_file(path, label) != payload:
        fail(label, "manifest changed while it was being parsed")
    if not isinstance(parsed, dict):
        fail(label, "manifest root must be an object")
    return parsed, payload


def _validate_manifest_sidecar(
    root: Path,
    manifest: dict[str, Any],
    manifest_payload: bytes,
) -> None:
    uri = manifest["integrity"]["manifestSidecar"]
    actual = _read_relative_file(root, uri, "$.integrity.manifestSidecar")
    expected = (
        f"{_sha256_bytes(manifest_payload)}  manifest.json\n".encode("ascii")
    )
    if actual != expected:
        fail(
            "$.integrity.manifestSidecar",
            "sidecar must exactly contain the lowercase manifest SHA-256, two "
            "spaces, 'manifest.json', and one newline",
        )


def _positive_finite(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        fail(path, "expected a finite positive number")
    return float(value)


def _planck_b_nu(frequency_hz: float, temperature_k: float, normalization: float) -> float:
    """Independent analytic Planck B_nu evaluation using exact SI constants."""

    exponent = (
        PLANCK_CONSTANT_J_S
        * frequency_hz
        / (BOLTZMANN_CONSTANT_J_K * temperature_k)
    )
    log_denominator = exponent if exponent > 50.0 else math.log(math.expm1(exponent))
    log_intensity = (
        math.log(2.0 * PLANCK_CONSTANT_J_S)
        - 2.0 * math.log(LIGHT_SPEED_M_S)
        + 3.0 * math.log(frequency_hz)
        - log_denominator
        + math.log(normalization)
    )
    if log_intensity < -745.0:
        return 0.0
    if log_intensity > MAX_FLOAT64_LOG:
        fail("$.environment", "declared Planck spectrum overflows float64")
    intensity = math.exp(log_intensity)
    if not math.isfinite(intensity) or intensity < 0.0:
        fail("$.environment", "declared Planck spectrum is not finite and non-negative")
    return intensity


def _quantize_expected_float32(value: float, path: str) -> bytes:
    if not math.isfinite(value) or value < 0.0:
        fail(path, "recomputed radiance is not finite and non-negative")
    try:
        payload = FLOAT32.pack(value)
    except (OverflowError, struct.error):
        fail(path, "recomputed radiance exceeds float32 range")
    if not math.isfinite(FLOAT32.unpack(payload)[0]):
        fail(path, "recomputed radiance quantizes to non-finite float32")
    return payload


def _float32_ulp_distance(first: bytes, second: bytes) -> int:
    return abs(UINT32.unpack(first)[0] - UINT32.unpack(second)[0])


def _validate_planck_environment(environment: dict[str, Any]) -> tuple[float, float]:
    if environment.get("kind") != PLANCK_ENVIRONMENT_KIND:
        fail(
            "$.environment.kind",
            "v1 independent numerical verification supports only "
            "'isotropic-planck-blackbody'",
        )
    temperature = _positive_finite(environment["temperatureK"], "$.environment.temperatureK")
    normalization = _positive_finite(
        environment["normalization"],
        "$.environment.normalization",
    )
    return temperature, normalization


def _validate_declared_input_binding(
    output: dict[str, Any],
    source: dict[str, Any],
    source_payload: bytes,
    validation_report: dict[str, Any],
) -> None:
    expected_input = {
        "datasetKind": source["datasetKind"],
        "id": source["id"],
        "manifestSha256": _sha256_bytes(source_payload),
        "schema": source["schema"],
        "scientificStatus": source["scientificStatus"],
        "validationStatus": validation_report["status"],
    }
    if output["inputTransferMap"] != expected_input:
        fail(
            "$.inputTransferMap",
            "declaration does not exactly match the independently validated input manifest",
        )
    if source.get("renderable") is not True:
        fail("$input.renderable", "input transfer map is not renderable")
    if source.get("physicalSystem", {}).get("vacuum") is not True:
        fail("$input.physicalSystem.vacuum", "input transfer map is not vacuum")

    expected_projection = {
        "heightPixels": source["projection"]["heightPixels"],
        "imageOrigin": source["projection"]["imageOrigin"],
        "pixelSampleLocation": source["projection"]["pixelSampleLocation"],
        "widthPixels": source["projection"]["widthPixels"],
    }
    if output["projection"] != expected_projection:
        fail("$.projection", "projection is not bound to the input transfer map")
    expected_sampling = {
        "dimensionOrder": source["sampling"]["dimensionOrder"],
        "observationTimesM": source["sampling"]["observationTimesM"],
        "pixelOrder": source["sampling"]["pixelOrder"],
    }
    if output["sampling"] != expected_sampling:
        fail("$.sampling", "sampling is not bound to the input transfer map")
    expected_transport = {
        "emittedFrequency": "nu_emit=nu_obs/g",
        "escapeBoundaryReferenceObserver": source["escapeBoundary"][
            "referenceObserver"
        ],
        "frequencyShiftConvention": source["escapeBoundary"][
            "frequencyShiftConvention"
        ],
        "mode": COMPOSITOR_ALGORITHM,
        "observedSpecificIntensity": "I_nu_obs=g^3*I_nu_emit",
        "samplingPolicy": (
            "sample environment only for escaped records with valid ICRS "
            "direction and positive frequency shift"
        ),
        "storedEscapeDirection": source["escapeBoundary"]["storedEscapeDirection"],
    }
    if output["transport"] != expected_transport:
        fail("$.transport", "transport declaration is not bound to the input boundary")


def _validate_layout_and_identity(
    manifest: dict[str, Any],
    source_manifest_sha256: str,
) -> None:
    frequencies = manifest["observerFrequencyBinsHz"]
    for index, value in enumerate(frequencies):
        _positive_finite(value, f"$.observerFrequencyBinsHz[{index}]")
    if any(current <= previous for previous, current in zip(frequencies, frequencies[1:])):
        fail("$.observerFrequencyBinsHz", "bins must be strictly increasing")

    expected_layout = {
        "spectral": {
            "byteOrder": "little-endian",
            "componentType": "float32",
            "componentsPerRecord": len(frequencies),
            "invalidEncoding": "canonical-quiet-NaN-0x7fc00000",
            "order": "record-major-then-observer-frequency-bin",
            "units": "W m^-2 sr^-1 Hz^-1",
        },
        "state": {
            "byteOrder": "little-endian",
            "policy": "byte-exact-authenticated-input-record-copy",
            "recordBytes": RECORD_BYTES,
            "schema": TRANSFER_SCHEMA,
            "structFormat": "<7fBBH",
        },
    }
    if manifest["recordLayout"] != expected_layout:
        fail("$.recordLayout", "record layout does not match the immutable v1 ABI")

    configuration = {
        "compositor": manifest["compositor"],
        "environment": manifest["environment"],
        "inputManifestSha256": source_manifest_sha256,
        "observerFrequencyBinsHz": frequencies,
        "schema": OUTPUT_SCHEMA,
    }
    configuration_digest = _sha256_bytes(_canonical_json_bytes(configuration))
    if manifest["integrity"]["configurationSha256"] != configuration_digest:
        fail("$.integrity.configurationSha256", "configuration digest mismatch")

    product_identity = {
        "configurationSha256": configuration_digest,
        "outputChunks": manifest["chunks"],
        "recordLayout": manifest["recordLayout"],
    }
    product_digest = _sha256_bytes(_canonical_json_bytes(product_identity))
    if manifest["integrity"]["productSha256"] != product_digest:
        fail("$.integrity.productSha256", "product identity digest mismatch")
    expected_id = f"offline-vacuum-{product_digest[:20]}"
    if manifest["id"] != expected_id:
        fail("$.id", f"expected content-bound identifier {expected_id!r}")

    source_uri = manifest["compositor"]["sourceUri"]
    source_payload = _read_relative_file(ROOT, source_uri, "$.compositor.sourceUri")
    source_digest = _sha256_bytes(source_payload)
    if manifest["compositor"]["sourceSha256"] != source_digest:
        fail(
            "$.compositor.sourceSha256",
            "declared compositor source does not match this verifier checkout",
        )


def _expected_chunk_stem(chunk: dict[str, Any]) -> str:
    tile = chunk["tile"]
    return (
        f"t{chunk['sampleIndex']:04d}"
        f"-y{tile['y']:06d}"
        f"-x{tile['x']:06d}"
    )


def _validate_files_and_spectra(
    manifest: dict[str, Any],
    output_root: Path,
    source: dict[str, Any],
    input_root: Path,
) -> tuple[int, int, int]:
    output_chunks = manifest["chunks"]
    input_chunks = source["chunks"]
    if len(output_chunks) != len(input_chunks):
        fail("$.chunks", "output chunk count does not match the input topology")

    frequencies = tuple(float(value) for value in manifest["observerFrequencyBinsHz"])
    temperature, normalization = _validate_planck_environment(manifest["environment"])
    outcome_codes = source["recordLayout"]["rayOutcomes"]
    code_to_name = {int(code): name for name, code in outcome_codes.items()}
    counts = {name: 0 for name in sorted(outcome_codes)}
    unusable = 0
    records = 0
    max_ulp = 0
    seen_uris: set[str] = {"manifest.json", manifest["integrity"]["manifestSidecar"]}

    for chunk_index, (chunk, input_chunk) in enumerate(
        zip(output_chunks, input_chunks)
    ):
        path = f"$.chunks[{chunk_index}]"
        for field in ("sampleIndex", "recordCount", "tile"):
            if chunk[field] != input_chunk[field]:
                fail(f"{path}.{field}", "does not match the validated input chunk")
        if chunk["inputChunkSha256"] != input_chunk["sha256"]:
            fail(f"{path}.inputChunkSha256", "does not match the input chunk hash")

        stem = _expected_chunk_stem(chunk)
        expected_spectral_uri = f"spectral/{stem}.f32"
        expected_state_uri = f"states/{stem}.bin"
        if chunk["spectral"]["uri"] != expected_spectral_uri:
            fail(f"{path}.spectral.uri", "URI does not encode sample/tile coordinates")
        if chunk["state"]["uri"] != expected_state_uri:
            fail(f"{path}.state.uri", "URI does not encode sample/tile coordinates")
        for artifact_name in ("spectral", "state"):
            uri = chunk[artifact_name]["uri"]
            if uri in seen_uris:
                fail(f"{path}.{artifact_name}.uri", f"duplicate output URI {uri!r}")
            seen_uris.add(uri)

        expected_state_bytes = chunk["recordCount"] * RECORD_BYTES
        expected_spectral_bytes = chunk["recordCount"] * len(frequencies) * FLOAT32.size
        if chunk["state"]["byteLength"] != expected_state_bytes:
            fail(f"{path}.state.byteLength", "does not equal recordCount * 32")
        if chunk["spectral"]["byteLength"] != expected_spectral_bytes:
            fail(
                f"{path}.spectral.byteLength",
                "does not equal recordCount * frequencyBinCount * 4",
            )

        state_payload = _validate_hashed_artifact(
            output_root,
            chunk["state"],
            f"{path}.state",
        )
        spectral_payload = _validate_hashed_artifact(
            output_root,
            chunk["spectral"],
            f"{path}.spectral",
        )
        input_payload = _read_relative_file(
            input_root,
            input_chunk["uri"],
            f"$input.chunks[{chunk_index}].uri",
        )
        if len(input_payload) != input_chunk["byteLength"]:
            fail(f"$input.chunks[{chunk_index}].byteLength", "input changed after validation")
        if _sha256_bytes(input_payload) != input_chunk["sha256"]:
            fail(f"$input.chunks[{chunk_index}].sha256", "input changed after validation")
        if state_payload != input_payload:
            fail(f"{path}.state", "state sidecar is not a byte-exact input record copy")
        if chunk["state"]["sha256"] != input_chunk["sha256"]:
            fail(f"{path}.state.sha256", "state hash is not the authenticated input hash")

        for record_index in range(chunk["recordCount"]):
            record_offset = record_index * RECORD_BYTES
            values = TRANSFER_RECORD.unpack_from(state_payload, record_offset)
            frequency_shift = float(values[3])
            outcome = int(values[7])
            outcome_name = code_to_name.get(outcome)
            if outcome_name is None:
                fail(f"{path}.state[{record_index}]", f"unknown ray outcome {outcome}")
            counts[outcome_name] += 1
            records += 1

            spectral_offset = record_index * len(frequencies) * FLOAT32.size
            bin_payloads = [
                spectral_payload[
                    spectral_offset + bin_index * FLOAT32.size :
                    spectral_offset + (bin_index + 1) * FLOAT32.size
                ]
                for bin_index in range(len(frequencies))
            ]
            if outcome_name == "captured":
                if any(payload != POSITIVE_ZERO_FLOAT32 for payload in bin_payloads):
                    fail(
                        f"{path}.spectral[{record_index}]",
                        "captured records must use positive float32 zero in every bin",
                    )
                continue
            if outcome_name != "escaped":
                unusable += 1
                if any(payload != CANONICAL_NAN_FLOAT32 for payload in bin_payloads):
                    fail(
                        f"{path}.spectral[{record_index}]",
                        "unusable records must use canonical float32 quiet NaN in every bin",
                    )
                continue

            if not math.isfinite(frequency_shift) or frequency_shift <= 0.0:
                fail(f"{path}.state[{record_index}]", "escaped record has invalid g")
            shift_cubed = frequency_shift * frequency_shift * frequency_shift
            if not math.isfinite(shift_cubed) or shift_cubed <= 0.0:
                fail(f"{path}.state[{record_index}]", "escaped record has invalid g^3")
            for bin_index, (observer_frequency, actual_payload) in enumerate(
                zip(frequencies, bin_payloads)
            ):
                emitted_frequency = observer_frequency / frequency_shift
                if not math.isfinite(emitted_frequency) or emitted_frequency <= 0.0:
                    fail(
                        f"{path}.spectral[{record_index}][{bin_index}]",
                        "emitted frequency is not finite and positive",
                    )
                emitted_intensity = _planck_b_nu(
                    emitted_frequency,
                    temperature,
                    normalization,
                )
                expected_payload = _quantize_expected_float32(
                    shift_cubed * emitted_intensity,
                    f"{path}.spectral[{record_index}][{bin_index}]",
                )
                actual_value = FLOAT32.unpack(actual_payload)[0]
                if not math.isfinite(actual_value) or actual_value < 0.0:
                    fail(
                        f"{path}.spectral[{record_index}][{bin_index}]",
                        "escaped radiance must be finite and non-negative",
                    )
                ulp_distance = _float32_ulp_distance(actual_payload, expected_payload)
                max_ulp = max(max_ulp, ulp_distance)
                if ulp_distance > 1:
                    fail(
                        f"{path}.spectral[{record_index}][{bin_index}]",
                        f"Planck/Liouville mismatch by {ulp_distance} float32 ULP",
                    )

    if manifest["outcomes"]["counts"] != counts:
        fail("$.outcomes.counts", "declared counts do not match decoded state records")
    if manifest["outcomes"]["recordCount"] != records:
        fail("$.outcomes.recordCount", "declared total does not match decoded records")
    if manifest["outcomes"]["unusableRecordCount"] != unusable:
        fail(
            "$.outcomes.unusableRecordCount",
            "declared unusable total does not match decoded state records",
        )
    return records, unusable, max_ulp


def _validate_no_extra_files(root: Path, allowed_files: set[str]) -> None:
    allowed_directories = {
        PurePosixPath(uri).parent.as_posix()
        for uri in allowed_files
        if PurePosixPath(uri).parent.as_posix() != "."
    }
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directories):
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                fail("$files", f"symlinked directory {relative!r} is forbidden")
            if relative not in allowed_directories:
                fail("$files", f"undeclared output directory {relative!r}")
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                fail("$files", f"symlinked file {relative!r} is forbidden")
            if relative not in allowed_files:
                fail("$files", f"undeclared output file {relative!r}")


def validate_offline_vacuum_product(
    manifest_path: Path | str,
    input_manifest_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA,
    transfer_schema_path: Path | str = DEFAULT_TRANSFER_SCHEMA,
) -> dict[str, Any]:
    """Validate a product and return a deterministic conformance report."""

    output_manifest_path = Path(manifest_path).absolute()
    source_manifest_path = Path(input_manifest_path).absolute()
    output_schema_path = Path(schema_path).absolute()
    transfer_schema = Path(transfer_schema_path).absolute()

    schema = load_json_strict(output_schema_path)
    if not isinstance(schema, dict):
        fail("$schema", "schema root must be an object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only JSON Schema Draft 2020-12 is supported")
    audit_schema_dialect(schema)

    manifest, manifest_payload = _load_manifest_snapshot(
        output_manifest_path,
        "$",
    )
    validate_json_schema(manifest, schema, schema)
    output_root = output_manifest_path.parent
    _validate_manifest_sidecar(output_root, manifest, manifest_payload)

    source, source_payload = _load_manifest_snapshot(
        source_manifest_path,
        "$input",
    )
    validation_report = validate_transfer_contract(
        source_manifest_path,
        transfer_schema,
    )
    if _read_stable_file(source_manifest_path, "$input") != source_payload:
        fail("$input", "input manifest changed during contract validation")
    if source.get("schema") != TRANSFER_SCHEMA:
        fail("$input.schema", "only blackhole.nr-transfer-map/v1 is supported")

    _validate_declared_input_binding(
        manifest,
        source,
        source_payload,
        validation_report,
    )
    _validate_layout_and_identity(
        manifest,
        _sha256_bytes(source_payload),
    )
    records, unusable, max_ulp = _validate_files_and_spectra(
        manifest,
        output_root,
        source,
        source_manifest_path.parent,
    )

    allowed_files = {"manifest.json", manifest["integrity"]["manifestSidecar"]}
    for chunk in manifest["chunks"]:
        allowed_files.add(chunk["spectral"]["uri"])
        allowed_files.add(chunk["state"]["uri"])
    _validate_no_extra_files(output_root, allowed_files)

    return {
        "chunks": len(manifest["chunks"]),
        "id": manifest["id"],
        "input": source["id"],
        "maxPlanckFloat32Ulp": max_ulp,
        "records": records,
        "schema": manifest["schema"],
        "status": "offline-vacuum-contract-conformant",
        "unusableRecords": unusable,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="offline product manifest.json")
    parser.add_argument(
        "--input-manifest",
        required=True,
        type=Path,
        help="source transfer-map manifest.json used by the product",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="offline product v1 JSON Schema",
    )
    parser.add_argument(
        "--transfer-schema",
        type=Path,
        default=DEFAULT_TRANSFER_SCHEMA,
        help="input transfer-map v1 JSON Schema",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_offline_vacuum_product(
            arguments.manifest,
            arguments.input_manifest,
            arguments.schema,
            arguments.transfer_schema,
        )
    except ContractError as error:
        print(f"Offline vacuum product validation failed: {error}", file=sys.stderr)
        return 1
    print("Offline vacuum product checks passed")
    for key in sorted(report):
        print(f"  {key} = {report[key]}")
    print(
        "  scope = authenticated vacuum endpoint spectral composition only; "
        "no NR/GRRT/OpenEXR claim was validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SCHEMA",
    "OfflineVacuumContractError",
    "validate_offline_vacuum_product",
]
