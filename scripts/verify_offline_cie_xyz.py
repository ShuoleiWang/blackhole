#!/usr/bin/env python3
"""Independently verify a CIE XYZ artifact against its spectral input.

The verifier independently authenticates the contract, inputs, topology, and
records, but replays the repository's canonical CIE integrator.  It is not an
independent colour-algorithm oracle; separate Decimal Planck-spectrum goldens
provide the external numerical check for that integrator.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.cie_color import (
    CIE_ROW_COUNT,
    CieColorError,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
    cie_1931_frequency_grid_hz,
    load_authenticated_cie_1931_2deg,
    spectral_i_nu_to_cie_xyz,
)
from offline.job import canonical_json_bytes
from offline.cie_product import CONVERTER_SOURCE_FILES
from offline.spectral_frame import (
    REQUIRED_CONVERGENCE_MASK,
    SpectralFrameError,
    SpectralPixelLayout,
    unpack_spectral_pixel,
)
from scripts.verify_nr_contract import (
    ContractError,
    audit_schema_dialect,
    validate_json_schema,
)
from scripts.verify_offline_spectral_frame import (
    validate_scientific_spectral_frame,
)


DEFAULT_SCHEMA = ROOT / "schemas" / "offline-cie-xyz-frame-v1.schema.json"
SCHEMA_ID = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-cie-xyz-frame-v1.schema.json"
)
PRODUCT_SCHEMA = "blackhole.scientific-cie-xyz-frame/v1"
CONVERTER_IMPLEMENTATION_ID = "blackhole.spectral-to-cie-xyz/v1"
PIXEL_LAYOUT_ID = "blackhole.scientific-cie-xyz-pixel/le-f64-v1"
RECORD_STRUCT = struct.Struct("<6d32sII")
RECORD_BYTES = RECORD_STRUCT.size

EXPECTED_SCIENTIFIC_STATUS = {
    "classification": (
        "authenticated CIE 1931 2-degree XYZ frame derived from exact-grid "
        "observer-frame spectral specific intensity"
    ),
    "primaryQuantity": "unnormalised-CIE-XYZ-from-mean-observer-frame-I_nu",
    "primaryUnits": "W m^-2 sr^-1 CIE tristimulus weighting",
    "errorSemantics": (
        "positive linear propagation of finite-stencil estimated absolute "
        "spectral errors; not a rigorous bound"
    ),
    "toneMappingApplied": False,
    "displayTransferApplied": False,
    "linearSrgbStored": False,
    "isCameraSensorModel": False,
    "isAbsoluteHumanAppearanceModel": False,
    "isNumericalRelativitySolver": False,
    "isGeneralRelativisticMagnetohydrodynamics": False,
    "isInputPhysicsRecomputed": False,
    "isIndependentColourAlgorithmOracle": False,
    "algorithmValidation": (
        "canonical CIE integrator with separate Decimal Planck-spectrum "
        "golden tests; artifact verifier replays that shared integrator"
    ),
    "prohibitedClaim": (
        "XYZ contract verification recomputes colour integration and record "
        "binding, not geodesics or emission physics; do not call it a camera, "
        "absolute appearance, NR, GRMHD, or display-rendering result."
    ),
}

EXPECTED_LAYOUT = {
    "convergenceMaskOffsetBytes": 84,
    "endianness": "little",
    "estimatedAbsoluteErrorXyzOffsetBytes": 24,
    "errorSemantics": (
        "linear-propagated-finite-stencil-estimate-not-rigorous-bound"
    ),
    "floatEncoding": "IEEE-754-binary64",
    "id": PIXEL_LAYOUT_ID,
    "inputRecordSha256OffsetBytes": 48,
    "meanCieXyzOffsetBytes": 0,
    "recordBytes": RECORD_BYTES,
    "sourceMaskOffsetBytes": 80,
}


class OfflineCieXyzContractError(ContractError):
    """A CIE XYZ product or its derivation evidence is invalid."""


def fail(path: str, message: str) -> NoReturn:
    raise OfflineCieXyzContractError(f"{path}: {message}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    try:
        return _sha256(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        fail("$", f"value is not finite canonical JSON: {error}")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                fail(label, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(label, f"invalid UTF-8 JSON: {error}")
    if not isinstance(result, dict):
        fail(label, "JSON root must be an object")
    return result


def _read_stable_file(path: Path, label: str, maximum_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        if path.is_symlink():
            fail(label, "symlinked files are forbidden")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(label, "expected a regular file")
        if before.st_size > maximum_bytes:
            fail(label, f"file exceeds the {maximum_bytes}-byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    except OfflineCieXyzContractError:
        raise
    except OSError as error:
        fail(label, f"unable to read file: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        fail(label, "file changed while it was being read")
    return payload


def _normalized_uri(value: Any, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        fail(path, "URI must be a normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        fail(path, "URI must be normalized, relative, and traversal-free")
    return pure


def _read_relative_file(
    root: Path,
    uri: Any,
    label: str,
    expected_bytes: int,
) -> bytes:
    pure = _normalized_uri(uri, label)
    if root.is_symlink():
        fail(label, "symlinked artifact root is forbidden")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directories: list[int] = []
    descriptor: int | None = None
    try:
        directories.append(os.open(root, directory_flags))
        for part in pure.parts[:-1]:
            directories.append(
                os.open(part, directory_flags, dir_fd=directories[-1])
            )
        descriptor = os.open(
            pure.parts[-1],
            file_flags,
            dir_fd=directories[-1],
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(label, "artifact must be a regular file")
        if before.st_size != expected_bytes:
            fail(label, "artifact byte length disagrees with its record topology")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(expected_bytes + 1)
        after = os.fstat(descriptor)
    except OfflineCieXyzContractError:
        raise
    except OSError as error:
        fail(label, f"unable to read traversal-safe artifact: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != before.st_size:
        fail(label, "artifact changed while it was being read")
    return payload


def _validate_no_extra_files(root: Path, allowed_files: set[str]) -> None:
    allowed_directories = {
        PurePosixPath(uri).parent.as_posix()
        for uri in allowed_files
        if PurePosixPath(uri).parent.as_posix() != "."
    }
    if root.is_symlink():
        fail("$files", "symlinked product root is forbidden")
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


def _source_descriptor() -> list[dict[str, Any]]:
    result = []
    for module_uri in CONVERTER_SOURCE_FILES:
        payload = _read_stable_file(
            ROOT / module_uri,
            f"converter source {module_uri}",
            4 << 20,
        )
        result.append(
            {
                "byteLength": len(payload),
                "moduleUri": module_uri,
                "sha256": _sha256(payload),
            }
        )
    return result


def _expected_output_uri(input_uri: str) -> str:
    pure = _normalized_uri(input_uri, "$.tiles[].inputPayload.uri")
    if pure.parent.as_posix() != "tiles" or pure.suffix != ".spx":
        fail("$.tiles[].inputPayload.uri", "unsupported spectral tile URI")
    return f"tiles/{pure.stem}.cxyz"


def _ulp_distance_non_negative(first: float, second: float) -> int:
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or first < 0.0
        or second < 0.0
    ):
        fail("$tiles", "XYZ values must be finite and non-negative")
    first_bits = struct.unpack("<Q", struct.pack("<d", first))[0]
    second_bits = struct.unpack("<Q", struct.pack("<d", second))[0]
    return abs(first_bits - second_bits)


class _CompensatedSum:
    __slots__ = ("total", "correction")

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def value(self) -> float:
        return self.total + self.correction


class _Summary:
    def __init__(self) -> None:
        self.record_count = 0
        self.integrated_xyz = tuple(_CompensatedSum() for _ in range(3))
        self.integrated_errors = tuple(_CompensatedSum() for _ in range(3))
        self.maximum_xyz = [0.0, 0.0, 0.0]
        self.maximum_errors = [0.0, 0.0, 0.0]
        self.source_union = 0
        self.convergence_intersection: int | None = None
        self.hash_chain = hashlib.sha256()

    def add(
        self,
        xyz: tuple[float, float, float],
        errors: tuple[float, float, float],
        solid_angle: float,
        source_mask: int,
        convergence_mask: int,
        input_digest: bytes,
    ) -> None:
        self.record_count += 1
        for index in range(3):
            self.integrated_xyz[index].add(xyz[index] * solid_angle)
            self.integrated_errors[index].add(errors[index] * solid_angle)
            self.maximum_xyz[index] = max(self.maximum_xyz[index], xyz[index])
            self.maximum_errors[index] = max(
                self.maximum_errors[index], errors[index]
            )
        self.source_union |= source_mask
        self.convergence_intersection = (
            convergence_mask
            if self.convergence_intersection is None
            else self.convergence_intersection & convergence_mask
        )
        self.hash_chain.update(input_digest)

    def descriptor(self) -> dict[str, Any]:
        return {
            "convergenceMaskIntersection": self.convergence_intersection,
            "estimatedAbsoluteErrorCieXyzOverFrame": [
                value.value() for value in self.integrated_errors
            ],
            "inputRecordSha256Chain": self.hash_chain.hexdigest(),
            "integratedCieXyzOverFrame": [
                value.value() for value in self.integrated_xyz
            ],
            "maximumMeanCieXyz": self.maximum_xyz,
            "maximumMeanEstimatedAbsoluteErrorXyz": self.maximum_errors,
            "recordCount": self.record_count,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_union,
        }


def _input_identity(
    manifest: dict[str, Any],
    payload: bytes,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": manifest["id"],
        "manifestSha256": _sha256(payload),
        "physicsVerified": report["physicsVerified"],
        "productSha256": manifest["integrity"]["productSha256"],
        "provenanceScope": report["provenanceScope"],
        "recordCount": report["recordCount"],
        "schema": manifest["schema"],
        "structuralStatus": report["status"],
        "tileCount": report["tileCount"],
    }


def _validate_sidecar(root: Path, manifest: dict[str, Any], payload: bytes) -> None:
    sidecar = _read_relative_file(
        root,
        manifest["integrity"]["manifestSidecar"],
        "$.integrity.manifestSidecar",
        80,
    )
    expected = f"{_sha256(payload)}  manifest.json\n".encode("ascii")
    if sidecar != expected:
        fail("$.integrity.manifestSidecar", "sidecar does not bind manifest")


def validate_scientific_cie_xyz_frame(
    manifest_path: Path | str,
    input_spectral_manifest_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_SCHEMA,
    cie_csv_path: Path | str = DEFAULT_CIE_CSV,
    cie_metadata_path: Path | str = DEFAULT_CIE_METADATA,
) -> dict[str, Any]:
    """Reauthenticate inputs and recompute every output XYZ record."""

    path = Path(manifest_path).absolute()
    input_path = Path(input_spectral_manifest_path).absolute()
    if path.name != "manifest.json" or input_path.name != "manifest.json":
        fail("$", "both v1 manifests must be named manifest.json")
    schema_payload = _read_stable_file(
        Path(schema_path).absolute(),
        "$schema",
        4 << 20,
    )
    default_schema_payload = _read_stable_file(
        DEFAULT_SCHEMA,
        "$defaultSchema",
        4 << 20,
    )
    if schema_payload != default_schema_payload:
        fail("$schema", "schema must byte-match the repository default")
    schema = _strict_json(schema_payload, "$schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only JSON Schema Draft 2020-12 is supported")
    if schema.get("$id") != SCHEMA_ID:
        fail("$schema.$id", "unexpected CIE XYZ schema id")
    audit_schema_dialect(schema)

    manifest_payload = _read_stable_file(path, "$", 16 << 20)
    manifest = _strict_json(manifest_payload, "$")
    validate_json_schema(manifest, schema, schema)
    if canonical_json_bytes(manifest) != manifest_payload:
        fail("$", "manifest is not canonical JSON")
    root = path.parent
    _validate_sidecar(root, manifest, manifest_payload)
    if manifest["schema"] != PRODUCT_SCHEMA:
        fail("$.schema", "unsupported product schema")
    if manifest["pixelLayout"] != EXPECTED_LAYOUT:
        fail("$.pixelLayout", "unsupported CIE XYZ record layout")
    if manifest["scientificStatus"] != EXPECTED_SCIENTIFIC_STATUS:
        fail("$.scientificStatus", "scientific capability boundary drifted")

    try:
        input_report = validate_scientific_spectral_frame(input_path)
    except ContractError as error:
        fail("$.inputSpectralProduct", f"input verification failed: {error}")
    input_payload = _read_stable_file(input_path, "$input", 64 << 20)
    input_manifest = _strict_json(input_payload, "$input")
    expected_input_identity = _input_identity(
        input_manifest,
        input_payload,
        input_report,
    )
    if manifest["inputSpectralProduct"] != expected_input_identity:
        fail("$.inputSpectralProduct", "input spectral identity mismatch")
    if manifest["frame"] != input_manifest["frame"]:
        fail("$.frame", "frame geometry differs from the spectral input")

    try:
        table = load_authenticated_cie_1931_2deg(
            cie_csv_path,
            cie_metadata_path,
        )
    except (CieColorError, OSError, TypeError, ValueError) as error:
        fail("$.cieDataset", f"CIE authentication failed: {error}")
    if manifest["cieDataset"] != table.descriptor():
        fail("$.cieDataset", "manifest does not bind the authenticated CIE table")
    frequencies = tuple(input_manifest["observerFrequencyBinsHz"])
    expected_frequencies = cie_1931_frequency_grid_hz(table)
    if len(frequencies) != CIE_ROW_COUNT or frequencies != expected_frequencies:
        fail(
            "$.inputSpectralProduct",
            "input does not use the exact authenticated 471-bin CIE grid",
        )
    try:
        spectral_layout = SpectralPixelLayout(frequencies)
    except (TypeError, ValueError) as error:
        fail("$.inputSpectralProduct", f"invalid spectral layout: {error}")
    if dict(spectral_layout.descriptor()) != input_manifest["pixelLayout"]:
        fail("$.inputSpectralProduct", "spectral layout descriptor mismatch")

    converter = manifest["converter"]
    descriptor = converter["descriptor"]
    if descriptor["implementationId"] != CONVERTER_IMPLEMENTATION_ID:
        fail("$.converter.descriptor.implementationId", "unsupported converter")
    expected_algorithm = (
        "I_nu-to-I_lambda Jacobian and 1-nm trapezoidal integration; "
        "same positive weights applied to estimated absolute errors"
    )
    if descriptor["algorithm"] != expected_algorithm:
        fail("$.converter.descriptor.algorithm", "converter algorithm drifted")
    if descriptor["sourceFiles"] != _source_descriptor():
        fail("$.converter.descriptor.sourceFiles", "converter source hash mismatch")
    if converter["descriptorSha256"] != _canonical_hash(descriptor):
        fail("$.converter.descriptorSha256", "converter descriptor hash mismatch")

    input_entries = input_manifest["tiles"]
    output_entries = manifest["tiles"]
    if len(output_entries) != len(input_entries):
        fail("$.tiles", "tile count differs from the spectral input")
    allowed = {"manifest.json", manifest["integrity"]["manifestSidecar"]}
    summary = _Summary()
    maximum_ulp_difference = 0
    for tile_index, (input_entry, output_entry) in enumerate(
        zip(input_entries, output_entries)
    ):
        tile_path = f"$.tiles[{tile_index}]"
        expected_shared = {
            "recordCount": input_entry["recordCount"],
            "recordOrder": input_entry["recordOrder"],
            "sampleIndex": input_entry["sampleIndex"],
            "tile": input_entry["tile"],
        }
        if any(output_entry[key] != value for key, value in expected_shared.items()):
            fail(tile_path, "tile topology differs from the spectral input")
        if output_entry["inputPayload"] != input_entry["payload"]:
            fail(f"{tile_path}.inputPayload", "input tile identity mismatch")
        expected_output_uri = _expected_output_uri(input_entry["payload"]["uri"])
        output_artifact = output_entry["outputPayload"]
        if output_artifact["uri"] != expected_output_uri:
            fail(f"{tile_path}.outputPayload.uri", "non-canonical output URI")
        record_count = input_entry["recordCount"]
        spectral_bytes = record_count * spectral_layout.record_bytes
        output_bytes = record_count * RECORD_BYTES
        source_payload = _read_relative_file(
            input_path.parent,
            input_entry["payload"]["uri"],
            f"{tile_path}.inputPayload",
            spectral_bytes,
        )
        if (
            input_entry["payload"]["byteLength"] != spectral_bytes
            or input_entry["payload"]["sha256"] != _sha256(source_payload)
        ):
            fail(f"{tile_path}.inputPayload", "input tile hash mismatch")
        xyz_payload = _read_relative_file(
            root,
            output_artifact["uri"],
            f"{tile_path}.outputPayload",
            output_bytes,
        )
        if (
            output_artifact["byteLength"] != output_bytes
            or output_artifact["sha256"] != _sha256(xyz_payload)
        ):
            fail(f"{tile_path}.outputPayload", "output tile hash mismatch")
        allowed.add(output_artifact["uri"])
        for record_index in range(record_count):
            record_path = f"{tile_path}.records[{record_index}]"
            spectral_offset = record_index * spectral_layout.record_bytes
            source_record_payload = source_payload[
                spectral_offset : spectral_offset + spectral_layout.record_bytes
            ]
            try:
                source_record = unpack_spectral_pixel(
                    spectral_layout,
                    source_record_payload,
                )
                expected_xyz_value = spectral_i_nu_to_cie_xyz(
                    frequencies,
                    source_record.mean_specific_intensities_nu,
                    table=table,
                )
                expected_error_value = spectral_i_nu_to_cie_xyz(
                    frequencies,
                    source_record.mean_estimated_absolute_errors_nu,
                    table=table,
                )
            except (CieColorError, SpectralFrameError, TypeError, ValueError) as error:
                fail(record_path, f"unable to recompute XYZ: {error}")
            output_offset = record_index * RECORD_BYTES
            values = RECORD_STRUCT.unpack(
                xyz_payload[output_offset : output_offset + RECORD_BYTES]
            )
            output_xyz = tuple(values[:3])
            output_errors = tuple(values[3:6])
            input_digest = values[6]
            source_mask = values[7]
            convergence_mask = values[8]
            expected_xyz = (
                expected_xyz_value.x,
                expected_xyz_value.y,
                expected_xyz_value.z,
            )
            expected_errors = (
                expected_error_value.x,
                expected_error_value.y,
                expected_error_value.z,
            )
            for channel, (actual, expected) in enumerate(
                zip((*output_xyz, *output_errors), (*expected_xyz, *expected_errors))
            ):
                difference = _ulp_distance_non_negative(actual, expected)
                maximum_ulp_difference = max(maximum_ulp_difference, difference)
                if difference > 1:
                    fail(
                        f"{record_path}.binary64[{channel}]",
                        f"recomputed value differs by {difference} ULP",
                    )
            expected_digest = hashlib.sha256(source_record_payload).digest()
            if input_digest != expected_digest:
                fail(record_path, "input record SHA-256 binding mismatch")
            if (
                source_mask != source_record.source_mask
                or convergence_mask != source_record.convergence_mask
            ):
                fail(record_path, "source or convergence mask binding mismatch")
            if convergence_mask & REQUIRED_CONVERGENCE_MASK != (
                REQUIRED_CONVERGENCE_MASK
            ):
                fail(record_path, "input convergence gates are incomplete")
            summary.add(
                expected_xyz,
                expected_errors,
                source_record.pixel_solid_angle_sr,
                source_mask,
                convergence_mask,
                input_digest,
            )

    expected_summary = summary.descriptor()
    if canonical_json_bytes(manifest["summary"]) != canonical_json_bytes(
        expected_summary
    ):
        fail("$.summary", "summary differs from recomputed input-derived XYZ")
    configuration = {
        "cieDataset": manifest["cieDataset"],
        "converter": descriptor,
        "frame": manifest["frame"],
        "inputSpectralProduct": manifest["inputSpectralProduct"],
        "pixelLayout": manifest["pixelLayout"],
        "schema": PRODUCT_SCHEMA,
    }
    configuration_hash = _canonical_hash(configuration)
    if manifest["integrity"]["configurationSha256"] != configuration_hash:
        fail("$.integrity.configurationSha256", "configuration hash mismatch")
    product_identity = {
        "configurationSha256": configuration_hash,
        "schema": PRODUCT_SCHEMA,
        "summary": manifest["summary"],
        "tiles": manifest["tiles"],
    }
    product_hash = _canonical_hash(product_identity)
    if manifest["integrity"]["productSha256"] != product_hash:
        fail("$.integrity.productSha256", "product hash mismatch")
    if manifest["id"] != f"scientific-cie-xyz-frame-{product_hash[:24]}":
        fail("$.id", "product id does not match product hash")
    _validate_no_extra_files(root, allowed)

    try:
        final_input_report = validate_scientific_spectral_frame(input_path)
    except ContractError as error:
        fail("$.inputSpectralProduct", f"final input verification failed: {error}")
    final_input_payload = _read_stable_file(input_path, "$input", 64 << 20)
    if final_input_report != input_report or final_input_payload != input_payload:
        fail("$.inputSpectralProduct", "input changed during XYZ verification")
    if descriptor["sourceFiles"] != _source_descriptor():
        fail("$.converter.descriptor.sourceFiles", "source changed during verification")
    return {
        "cieIntegrationVerified": True,
        "colourAlgorithmOracleIndependent": False,
        "colourAlgorithmValidation": (
            "shared-canonical-integrator-plus-separate-Decimal-Planck-goldens"
        ),
        "id": manifest["id"],
        "inputPhysicsVerified": False,
        "maximumUlpDifference": maximum_ulp_difference,
        "recordCount": summary.record_count,
        "status": "scientific-cie-xyz-frame-contract-conformant",
        "tileCount": len(output_entries),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input_spectral_manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--cie-csv", type=Path, default=DEFAULT_CIE_CSV)
    parser.add_argument(
        "--cie-metadata",
        type=Path,
        default=DEFAULT_CIE_METADATA,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_scientific_cie_xyz_frame(
            arguments.manifest,
            arguments.input_spectral_manifest,
            schema_path=arguments.schema,
            cie_csv_path=arguments.cie_csv,
            cie_metadata_path=arguments.cie_metadata,
        )
    except ContractError as error:
        print(f"offline CIE XYZ validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
