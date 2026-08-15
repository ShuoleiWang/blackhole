#!/usr/bin/env python3
"""Verify an unclamped linear-sRGB v1 derivative against XYZ and spectra.

Artifact decoding and identity checks are independent of the producer.  The
matrix replay intentionally uses the same canonical D65 transform as the
producer, so this is not an independent colour-algorithm oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.job import canonical_json_bytes
from offline.linear_rgb_product import CONVERTER_SOURCE_FILES
from offline.spectral_frame import REQUIRED_CONVERGENCE_MASK
from scripts.verify_nr_contract import (
    ContractError,
    audit_schema_dialect,
    validate_json_schema,
)
from scripts.verify_offline_cie_xyz import (
    _read_relative_file,
    _read_stable_file,
    _strict_json,
    _validate_no_extra_files,
    _validate_sidecar,
    validate_scientific_cie_xyz_frame,
)


DEFAULT_SCHEMA = ROOT / "schemas" / "offline-linear-srgb-frame-v1.schema.json"
SCHEMA_ID = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-linear-srgb-frame-v1.schema.json"
)
PRODUCT_SCHEMA = "blackhole.scientific-linear-srgb-frame/v1"
CONVERTER_IMPLEMENTATION_ID = "blackhole.cie-xyz-to-linear-srgb/v1"
RECORD_STRUCT = struct.Struct("<6d32s32sII")
RECORD_BYTES = RECORD_STRUCT.size
CIE_XYZ_RECORD_BYTES = 88

MATRIX = (
    (3.2409699419045226, -1.537383177570094, -0.4986107602930034),
    (-0.9692436362808796, 1.8759675015077202, 0.04155505740717559),
    (0.05563007969699366, -0.20397695888897652, 1.0569715142428786),
)

EXPECTED_LAYOUT = {
    "convergenceMaskOffsetBytes": 116,
    "endianness": "little",
    "estimatedAbsoluteErrorLinearSrgbOffsetBytes": 24,
    "errorSemantics": "abs-D65-matrix-propagated-non-rigorous-estimate",
    "floatEncoding": "IEEE-754-binary64",
    "id": "blackhole.scientific-linear-srgb-pixel/le-f64-v1",
    "inputCieXyzRecordSha256OffsetBytes": 48,
    "inputSpectralRecordSha256OffsetBytes": 80,
    "linearSrgbOffsetBytes": 0,
    "recordBytes": 120,
    "sourceMaskOffsetBytes": 112,
}

EXPECTED_STATUS = {
    "classification": (
        "authenticated unclamped scene-linear sRGB D65 derivative of "
        "scientific CIE XYZ"
    ),
    "primaryQuantity": "unclamped-scene-linear-sRGB-D65",
    "primaryUnits": "same linear radiance scale as input CIE XYZ",
    "estimatedError": "abs(XYZ-to-linear-sRGB-matrix) times XYZ estimate",
    "errorIsRigorousBound": False,
    "negativeChannelPolicy": "preserve-finite-out-of-gamut-coordinates",
    "clamped": False,
    "exposureApplied": False,
    "gamutMappingApplied": False,
    "toneMappingApplied": False,
    "hdrTransferApplied": False,
    "srgbTransferCurveApplied": False,
    "integerQuantizationApplied": False,
    "isDisplayImage": False,
    "isCameraSensorModel": False,
    "isAbsoluteHumanAppearanceModel": False,
    "isNumericalRelativitySolver": False,
    "isGeneralRelativisticMagnetohydrodynamics": False,
    "isInputPhysicsRecomputed": False,
    "isIndependentColourAlgorithmOracle": False,
    "algorithmValidation": (
        "artifact verifier replays the shared canonical D65 matrix; separate "
        "matrix white, negative-gamut, and error-propagation goldens apply"
    ),
    "prohibitedClaim": (
        "Do not describe unclamped linear-sRGB coordinates as display-ready "
        "pixels, a camera/appearance model, an independent colour oracle, "
        "NR, GRMHD, or a recomputation of the input physics."
    ),
}


class OfflineLinearSrgbContractError(ContractError):
    """A linear-sRGB derivative contract failed closed."""


def fail(path: str, message: str) -> NoReturn:
    raise OfflineLinearSrgbContractError(f"{path}: {message}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    try:
        return _sha256(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        fail("$", f"value is not finite canonical JSON: {error}")


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


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & (1 << 63):
        return (~bits) & 0xFFFFFFFFFFFFFFFF
    return bits | (1 << 63)


def _ulp_distance(first: float, second: float) -> int:
    if not math.isfinite(first) or not math.isfinite(second):
        fail("$tiles", "linear-sRGB channels must be finite")
    return abs(_ordered_float_bits(first) - _ordered_float_bits(second))


def _propagated_error(errors: tuple[float, float, float]) -> tuple[float, float, float]:
    if any(not math.isfinite(value) or value < 0.0 for value in errors):
        fail("$tiles", "linear-sRGB error inputs must be finite and non-negative")
    result = tuple(
        math.fsum(abs(MATRIX[row][column]) * errors[column] for column in range(3))
        for row in range(3)
    )
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        fail("$tiles", "linear-sRGB error propagation overflowed")
    return result  # type: ignore[return-value]


def _apply_manifest_matrix(
    xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Replay the declared D65 matrix without calling producer colour code."""

    if any(not math.isfinite(value) or value < 0.0 for value in xyz):
        fail("$tiles", "XYZ matrix inputs must be finite and non-negative")
    result = tuple(
        math.fsum(
            MATRIX[row][column] * xyz[column]
            for column in range(3)
        )
        for row in range(3)
    )
    if any(not math.isfinite(value) for value in result):
        fail("$tiles", "declared XYZ-to-linear-sRGB matrix overflowed")
    return result  # type: ignore[return-value]


class _Summary:
    def __init__(self) -> None:
        self.records = 0
        self.minimum = [math.inf, math.inf, math.inf]
        self.maximum = [-math.inf, -math.inf, -math.inf]
        self.maximum_error = [0.0, 0.0, 0.0]
        self.negative_union = 0
        self.records_negative = 0
        self.source_union = 0
        self.convergence_intersection: int | None = None
        self.xyz_chain = hashlib.sha256()
        self.spectral_chain = hashlib.sha256()

    def add(
        self,
        rgb: tuple[float, float, float],
        errors: tuple[float, float, float],
        xyz_digest: bytes,
        spectral_digest: bytes,
        source_mask: int,
        convergence_mask: int,
    ) -> None:
        self.records += 1
        negative_mask = 0
        for index in range(3):
            self.minimum[index] = min(self.minimum[index], rgb[index])
            self.maximum[index] = max(self.maximum[index], rgb[index])
            self.maximum_error[index] = max(self.maximum_error[index], errors[index])
            if rgb[index] < 0.0:
                negative_mask |= 1 << index
        self.negative_union |= negative_mask
        if negative_mask:
            self.records_negative += 1
        self.source_union |= source_mask
        self.convergence_intersection = (
            convergence_mask
            if self.convergence_intersection is None
            else self.convergence_intersection & convergence_mask
        )
        self.xyz_chain.update(xyz_digest)
        self.spectral_chain.update(spectral_digest)

    def descriptor(self) -> dict[str, Any]:
        return {
            "convergenceMaskIntersection": self.convergence_intersection,
            "inputCieXyzRecordSha256Chain": self.xyz_chain.hexdigest(),
            "inputSpectralRecordSha256Chain": self.spectral_chain.hexdigest(),
            "maximumEstimatedAbsoluteErrorLinearSrgb": self.maximum_error,
            "maximumLinearSrgb": self.maximum,
            "minimumLinearSrgb": self.minimum,
            "negativeChannelMaskUnion": self.negative_union,
            "recordCount": self.records,
            "recordsWithNegativeChannel": self.records_negative,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_union,
        }


def _input_xyz_identity(
    manifest: dict[str, Any],
    payload: bytes,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cieIntegrationVerified": report["cieIntegrationVerified"],
        "colourAlgorithmOracleIndependent": report[
            "colourAlgorithmOracleIndependent"
        ],
        "id": manifest["id"],
        "inputPhysicsVerified": report["inputPhysicsVerified"],
        "manifestSha256": _sha256(payload),
        "maximumUlpDifference": report["maximumUlpDifference"],
        "productSha256": manifest["integrity"]["productSha256"],
        "recordCount": report["recordCount"],
        "schema": manifest["schema"],
        "structuralStatus": report["status"],
        "tileCount": report["tileCount"],
    }


def _expected_output_uri(input_uri: str) -> str:
    if not input_uri.startswith("tiles/") or not input_uri.endswith(".cxyz"):
        fail("$.tiles[].inputCieXyzPayload.uri", "unsupported XYZ tile URI")
    stem = input_uri.removeprefix("tiles/").removesuffix(".cxyz")
    if not stem or "/" in stem or ".." in stem:
        fail("$.tiles[].inputCieXyzPayload.uri", "non-canonical XYZ tile URI")
    return f"tiles/{stem}.lsrgb"


def validate_scientific_linear_srgb_frame(
    manifest_path: Path | str,
    input_cie_xyz_manifest_path: Path | str,
    input_spectral_manifest_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    path = Path(manifest_path).absolute()
    xyz_path = Path(input_cie_xyz_manifest_path).absolute()
    spectral_path = Path(input_spectral_manifest_path).absolute()
    if any(item.name != "manifest.json" for item in (path, xyz_path, spectral_path)):
        fail("$", "all v1 manifests must be named manifest.json")
    schema_payload = _read_stable_file(
        Path(schema_path).absolute(), "$schema", 4 << 20
    )
    default_schema_payload = _read_stable_file(
        DEFAULT_SCHEMA, "$defaultSchema", 4 << 20
    )
    if schema_payload != default_schema_payload:
        fail("$schema", "schema must byte-match the repository default")
    schema = _strict_json(schema_payload, "$schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only Draft 2020-12 is supported")
    if schema.get("$id") != SCHEMA_ID:
        fail("$schema.$id", "unexpected linear-sRGB schema id")
    audit_schema_dialect(schema)
    manifest_payload = _read_stable_file(path, "$", 16 << 20)
    manifest = _strict_json(manifest_payload, "$")
    validate_json_schema(manifest, schema, schema)
    if canonical_json_bytes(manifest) != manifest_payload:
        fail("$", "manifest is not canonical JSON")
    _validate_sidecar(path.parent, manifest, manifest_payload)
    if manifest["schema"] != PRODUCT_SCHEMA:
        fail("$.schema", "unsupported product schema")
    if manifest["pixelLayout"] != EXPECTED_LAYOUT:
        fail("$.pixelLayout", "unsupported linear-sRGB record ABI")
    if manifest["scientificStatus"] != EXPECTED_STATUS:
        fail("$.scientificStatus", "scientific boundary drifted")

    try:
        xyz_report = validate_scientific_cie_xyz_frame(
            xyz_path,
            spectral_path,
        )
    except ContractError as error:
        fail("$.inputCieXyzProduct", f"XYZ verification failed: {error}")
    xyz_payload = _read_stable_file(xyz_path, "$xyz", 16 << 20)
    xyz_manifest = _strict_json(xyz_payload, "$xyz")
    if manifest["inputCieXyzProduct"] != _input_xyz_identity(
        xyz_manifest, xyz_payload, xyz_report
    ):
        fail("$.inputCieXyzProduct", "XYZ product identity mismatch")
    if manifest["inputSpectralProduct"] != xyz_manifest["inputSpectralProduct"]:
        fail("$.inputSpectralProduct", "original spectral identity mismatch")
    if manifest["cieDataset"] != xyz_manifest["cieDataset"]:
        fail("$.cieDataset", "CIE descriptor differs from the XYZ primary")
    if manifest["frame"] != xyz_manifest["frame"]:
        fail("$.frame", "frame differs from the XYZ primary")

    descriptor = manifest["converter"]["descriptor"]
    if descriptor["implementationId"] != CONVERTER_IMPLEMENTATION_ID:
        fail("$.converter.descriptor.implementationId", "unsupported converter")
    if descriptor["matrix"] != [list(row) for row in MATRIX]:
        fail("$.converter.descriptor.matrix", "D65 matrix drifted")
    if descriptor["sourceFiles"] != _source_descriptor():
        fail("$.converter.descriptor.sourceFiles", "producer source hash mismatch")
    if manifest["converter"]["descriptorSha256"] != _canonical_hash(descriptor):
        fail("$.converter.descriptorSha256", "converter hash mismatch")

    xyz_entries = xyz_manifest["tiles"]
    entries = manifest["tiles"]
    if len(entries) != len(xyz_entries):
        fail("$.tiles", "tile count differs from XYZ input")
    allowed = {"manifest.json", manifest["integrity"]["manifestSidecar"]}
    summary = _Summary()
    maximum_ulp = 0
    for tile_index, (xyz_entry, entry) in enumerate(zip(xyz_entries, entries)):
        tile_path = f"$.tiles[{tile_index}]"
        for key in ("recordCount", "recordOrder", "sampleIndex", "tile"):
            if entry[key] != xyz_entry[key]:
                fail(tile_path, "tile topology differs from XYZ input")
        if entry["inputCieXyzPayload"] != xyz_entry["outputPayload"]:
            fail(f"{tile_path}.inputCieXyzPayload", "XYZ tile identity mismatch")
        if entry["inputSpectralPayload"] != xyz_entry["inputPayload"]:
            fail(f"{tile_path}.inputSpectralPayload", "spectral tile mismatch")
        output_artifact = entry["outputPayload"]
        expected_uri = _expected_output_uri(xyz_entry["outputPayload"]["uri"])
        if output_artifact["uri"] != expected_uri:
            fail(f"{tile_path}.outputPayload.uri", "non-canonical output URI")
        count = entry["recordCount"]
        xyz_tile = _read_relative_file(
            xyz_path.parent,
            xyz_entry["outputPayload"]["uri"],
            f"{tile_path}.inputCieXyzPayload",
            count * CIE_XYZ_RECORD_BYTES,
        )
        if _sha256(xyz_tile) != xyz_entry["outputPayload"]["sha256"]:
            fail(f"{tile_path}.inputCieXyzPayload", "XYZ tile hash mismatch")
        output_tile = _read_relative_file(
            path.parent,
            output_artifact["uri"],
            f"{tile_path}.outputPayload",
            count * RECORD_BYTES,
        )
        if (
            output_artifact["byteLength"] != count * RECORD_BYTES
            or _sha256(output_tile) != output_artifact["sha256"]
        ):
            fail(f"{tile_path}.outputPayload", "output tile hash mismatch")
        allowed.add(output_artifact["uri"])
        for record_index in range(count):
            record_path = f"{tile_path}.records[{record_index}]"
            xyz_raw = xyz_tile[
                record_index * CIE_XYZ_RECORD_BYTES :
                (record_index + 1) * CIE_XYZ_RECORD_BYTES
            ]
            xyz_values = struct.unpack("<6d32sII", xyz_raw)
            xyz = tuple(xyz_values[:3])
            xyz_errors = tuple(xyz_values[3:6])
            spectral_digest = xyz_values[6]
            source_mask = xyz_values[7]
            convergence_mask = xyz_values[8]
            if any(
                not math.isfinite(value) or value < 0.0
                for value in (*xyz, *xyz_errors)
            ):
                fail(record_path, "input XYZ record is invalid")
            try:
                expected_rgb = _apply_manifest_matrix(
                    xyz  # type: ignore[arg-type]
                )
                expected_errors = _propagated_error(
                    xyz_errors  # type: ignore[arg-type]
                )
            except (TypeError, ValueError) as error:
                fail(record_path, f"matrix replay failed: {error}")
            raw = output_tile[
                record_index * RECORD_BYTES : (record_index + 1) * RECORD_BYTES
            ]
            values = RECORD_STRUCT.unpack(raw)
            actual_rgb = tuple(values[:3])
            actual_errors = tuple(values[3:6])
            for channel, (actual, expected) in enumerate(
                zip(
                    (*actual_rgb, *actual_errors),
                    (*expected_rgb, *expected_errors),
                )
            ):
                distance = _ulp_distance(actual, expected)
                maximum_ulp = max(maximum_ulp, distance)
                if distance > 1:
                    fail(
                        f"{record_path}.binary64[{channel}]",
                        f"matrix replay differs by {distance} ULP",
                    )
            xyz_digest = hashlib.sha256(xyz_raw).digest()
            if values[6] != xyz_digest or values[7] != spectral_digest:
                fail(record_path, "input record SHA-256 binding mismatch")
            if values[8] != source_mask or values[9] != convergence_mask:
                fail(record_path, "source or convergence binding mismatch")
            if convergence_mask & REQUIRED_CONVERGENCE_MASK != (
                REQUIRED_CONVERGENCE_MASK
            ):
                fail(record_path, "required convergence gates are incomplete")
            summary.add(
                expected_rgb,
                expected_errors,
                xyz_digest,
                spectral_digest,
                source_mask,
                convergence_mask,
            )
    if canonical_json_bytes(manifest["summary"]) != canonical_json_bytes(
        summary.descriptor()
    ):
        fail("$.summary", "summary differs from matrix replay")
    configuration = {
        "cieDataset": manifest["cieDataset"],
        "converter": descriptor,
        "frame": manifest["frame"],
        "inputCieXyzProduct": manifest["inputCieXyzProduct"],
        "inputSpectralProduct": manifest["inputSpectralProduct"],
        "pixelLayout": manifest["pixelLayout"],
        "schema": PRODUCT_SCHEMA,
    }
    configuration_hash = _canonical_hash(configuration)
    if manifest["integrity"]["configurationSha256"] != configuration_hash:
        fail("$.integrity.configurationSha256", "configuration hash mismatch")
    identity = {
        "configurationSha256": configuration_hash,
        "schema": PRODUCT_SCHEMA,
        "summary": manifest["summary"],
        "tiles": manifest["tiles"],
    }
    product_hash = _canonical_hash(identity)
    if manifest["integrity"]["productSha256"] != product_hash:
        fail("$.integrity.productSha256", "product hash mismatch")
    if manifest["id"] != f"scientific-linear-srgb-frame-{product_hash[:24]}":
        fail("$.id", "product id does not match product hash")
    _validate_no_extra_files(path.parent, allowed)
    try:
        final_xyz_report = validate_scientific_cie_xyz_frame(
            xyz_path, spectral_path
        )
    except ContractError as error:
        fail("$.inputCieXyzProduct", f"final XYZ verification failed: {error}")
    if (
        final_xyz_report != xyz_report
        or _read_stable_file(xyz_path, "$xyz", 16 << 20) != xyz_payload
    ):
        fail("$.inputCieXyzProduct", "XYZ product changed during verification")
    if descriptor["sourceFiles"] != _source_descriptor():
        fail("$.converter.descriptor.sourceFiles", "source changed during verification")
    return {
        "colourAlgorithmOracleIndependent": False,
        "id": manifest["id"],
        "inputPhysicsVerified": False,
        "matrixTransformVerified": True,
        "maximumUlpDifference": maximum_ulp,
        "recordCount": summary.records,
        "status": "scientific-linear-srgb-frame-contract-conformant",
        "tileCount": len(entries),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input_cie_xyz_manifest", type=Path)
    parser.add_argument("input_spectral_manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_scientific_linear_srgb_frame(
            arguments.manifest,
            arguments.input_cie_xyz_manifest,
            arguments.input_spectral_manifest,
            schema_path=arguments.schema,
        )
    except ContractError as error:
        print(f"offline linear-sRGB validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
