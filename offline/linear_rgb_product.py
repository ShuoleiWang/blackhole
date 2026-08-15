"""Versioned unclamped scene-linear sRGB derived from scientific CIE XYZ.

This layer consumes only a strictly verified CIE XYZ v1 product and its
original spectral manifest.  It applies the fixed D65 XYZ-to-linear-sRGB
matrix without exposure, clamping, gamut mapping, tone mapping, HDR transfer,
or an sRGB encoding curve.  Finite negative channels are valid out-of-gamut
linear coordinates and are preserved exactly.

The propagated error is ``abs(M) * estimated_abs_error_XYZ``.  Its inputs are
finite-stencil estimates, so the resulting non-negative triplet remains an
estimate rather than a rigorous bound.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import struct
from types import MappingProxyType
from typing import Any, Final, Mapping
import uuid

import offline.cie_product as cie_product_module
from offline.cie_product import (
    CieProductError,
    RECORD_BYTES as CIE_XYZ_RECORD_BYTES,
    unpack_cie_xyz_pixel,
)
from offline.job import canonical_json_bytes
from offline.spectral_frame import REQUIRED_CONVERGENCE_MASK
from offline.spectral_product import default_numeric_backend_descriptor


PRODUCT_SCHEMA: Final = "blackhole.scientific-linear-srgb-frame/v1"
PRODUCT_SCHEMA_ID: Final = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-linear-srgb-frame-v1.schema.json"
)
CONVERTER_IMPLEMENTATION_ID: Final = "blackhole.cie-xyz-to-linear-srgb/v1"
PIXEL_LAYOUT_ID: Final = "blackhole.scientific-linear-srgb-pixel/le-f64-v1"
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
RECORD_STRUCT: Final = struct.Struct("<6d32s32sII")
RECORD_BYTES: Final = RECORD_STRUCT.size
CONVERTER_SOURCE_FILES: Final = (
    "assets/science/cie/CIE_xyz_1931_2deg.csv",
    "assets/science/cie/CIE_xyz_1931_2deg.csv_metadata.json",
    "offline/__init__.py",
    "offline/adaptive_frame.py",
    "offline/cie_color.py",
    "offline/cie_product.py",
    "offline/geodesic.py",
    "offline/job.py",
    "offline/kerr.py",
    "offline/linear_rgb_product.py",
    "offline/radiative_transfer.py",
    "offline/spacetime.py",
    "offline/spectral_frame.py",
    "offline/spectral_product.py",
    "schemas/offline-cie-xyz-frame-v1.schema.json",
    "schemas/offline-linear-srgb-frame-v1.schema.json",
    "schemas/offline-scientific-spectral-frame-v1.schema.json",
    "scripts/convert_offline_cie_xyz_to_linear_srgb.py",
    "scripts/verify_nr_contract.py",
    "scripts/verify_offline_cie_xyz.py",
    "scripts/verify_offline_linear_srgb.py",
    "scripts/verify_offline_spectral_frame.py",
)

XYZ_TO_LINEAR_SRGB_D65: Final = (
    (3.2409699419045226, -1.537383177570094, -0.4986107602930034),
    (-0.9692436362808796, 1.8759675015077202, 0.04155505740717559),
    (0.05563007969699366, -0.20397695888897652, 1.0569715142428786),
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
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
)


class LinearRgbProductError(RuntimeError):
    """A linear-sRGB derivative failed its closed artifact contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _canonical_object(
    value: Any,
    label: str,
    *,
    implementation_id: bool = False,
) -> dict[str, Any]:
    try:
        result = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise LinearRgbProductError(
            f"{label} must be finite canonical JSON: {error}"
        ) from error
    if not isinstance(result, dict):
        raise LinearRgbProductError(f"{label} must be an object")
    if implementation_id and (
        not isinstance(result.get("implementationId"), str)
        or not result["implementationId"]
    ):
        raise LinearRgbProductError(
            f"{label} requires a non-empty implementationId"
        )
    return result


def _finite(value: Any, label: str, *, non_negative: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if non_negative and result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class LinearSrgbPixelRecord:
    linear_srgb: tuple[float, float, float]
    estimated_absolute_error_linear_srgb: tuple[float, float, float]
    input_cie_xyz_record_sha256: bytes
    input_spectral_record_sha256: bytes
    source_mask: int
    convergence_mask: int

    def __post_init__(self) -> None:
        for name, non_negative in (
            ("linear_srgb", False),
            ("estimated_absolute_error_linear_srgb", True),
        ):
            try:
                values = tuple(getattr(self, name))
            except TypeError as error:
                raise ValueError(f"{name} must contain three numbers") from error
            if len(values) != 3:
                raise ValueError(f"{name} must contain three numbers")
            object.__setattr__(
                self,
                name,
                tuple(
                    _finite(
                        value,
                        f"{name}[{index}]",
                        non_negative=non_negative,
                    )
                    for index, value in enumerate(values)
                ),
            )
        for name in (
            "input_cie_xyz_record_sha256",
            "input_spectral_record_sha256",
        ):
            digest = bytes(getattr(self, name))
            if len(digest) != 32:
                raise ValueError(f"{name} must contain 32 bytes")
            object.__setattr__(self, name, digest)
        for name in ("source_mask", "convergence_mask"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 0xFFFFFFFF:
                raise ValueError(f"{name} must fit uint32")
        if self.convergence_mask & REQUIRED_CONVERGENCE_MASK != (
            REQUIRED_CONVERGENCE_MASK
        ):
            raise ValueError("input convergence mask lacks a required gate")


def linear_srgb_pixel_layout_descriptor() -> dict[str, Any]:
    return {
        "convergenceMaskOffsetBytes": 116,
        "endianness": "little",
        "estimatedAbsoluteErrorLinearSrgbOffsetBytes": 24,
        "errorSemantics": "abs-D65-matrix-propagated-non-rigorous-estimate",
        "floatEncoding": "IEEE-754-binary64",
        "id": PIXEL_LAYOUT_ID,
        "inputCieXyzRecordSha256OffsetBytes": 48,
        "inputSpectralRecordSha256OffsetBytes": 80,
        "linearSrgbOffsetBytes": 0,
        "recordBytes": RECORD_BYTES,
        "sourceMaskOffsetBytes": 112,
    }


def pack_linear_srgb_pixel(record: LinearSrgbPixelRecord) -> bytes:
    if not isinstance(record, LinearSrgbPixelRecord):
        raise TypeError("record must be LinearSrgbPixelRecord")
    return RECORD_STRUCT.pack(
        *record.linear_srgb,
        *record.estimated_absolute_error_linear_srgb,
        record.input_cie_xyz_record_sha256,
        record.input_spectral_record_sha256,
        record.source_mask,
        record.convergence_mask,
    )


def unpack_linear_srgb_pixel(
    payload: bytes | bytearray | memoryview,
) -> LinearSrgbPixelRecord:
    raw = bytes(payload)
    if len(raw) != RECORD_BYTES:
        raise LinearRgbProductError("linear-sRGB record has the wrong length")
    values = RECORD_STRUCT.unpack(raw)
    try:
        return LinearSrgbPixelRecord(
            linear_srgb=tuple(values[:3]),  # type: ignore[arg-type]
            estimated_absolute_error_linear_srgb=(
                tuple(values[3:6])  # type: ignore[arg-type]
            ),
            input_cie_xyz_record_sha256=values[6],
            input_spectral_record_sha256=values[7],
            source_mask=values[8],
            convergence_mask=values[9],
        )
    except (TypeError, ValueError) as error:
        raise LinearRgbProductError(
            f"linear-sRGB record is invalid: {error}"
        ) from error


def propagated_linear_srgb_absolute_error(
    estimated_absolute_error_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply ``abs(M)`` to a non-negative XYZ error estimate."""

    if len(estimated_absolute_error_xyz) != 3:
        raise ValueError("estimated_absolute_error_xyz must contain three values")
    errors = tuple(
        _finite(value, f"estimated_absolute_error_xyz[{index}]", non_negative=True)
        for index, value in enumerate(estimated_absolute_error_xyz)
    )
    result = tuple(
        math.fsum(
            abs(XYZ_TO_LINEAR_SRGB_D65[row][column]) * errors[column]
            for column in range(3)
        )
        for row in range(3)
    )
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise LinearRgbProductError("linear-sRGB error propagation overflowed")
    return result  # type: ignore[return-value]


def apply_declared_xyz_to_linear_srgb(
    xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Apply the exact matrix carried by this product's descriptor.

    The scientific XYZ helper owns the general colour API, but this versioned
    artifact must not compute pixels through one implementation while merely
    *claiming* a second matrix in its manifest.  Both the value transform and
    error propagation therefore use :data:`XYZ_TO_LINEAR_SRGB_D65` directly.
    """

    if len(xyz) != 3:
        raise ValueError("xyz must contain three values")
    source = tuple(
        _finite(value, f"xyz[{index}]", non_negative=True)
        for index, value in enumerate(xyz)
    )
    result = tuple(
        math.fsum(
            XYZ_TO_LINEAR_SRGB_D65[row][column] * source[column]
            for column in range(3)
        )
        for row in range(3)
    )
    if any(not math.isfinite(value) for value in result):
        raise LinearRgbProductError("declared XYZ-to-linear-sRGB matrix overflowed")
    return result  # type: ignore[return-value]


def _source_file_descriptor(path: Path, module_uri: str) -> dict[str, Any]:
    payload = cie_product_module._read_stable_regular(
        path,
        f"converter source {module_uri}",
        4 << 20,
    )
    return {
        "byteLength": len(payload),
        "moduleUri": module_uri,
        "sha256": _sha256(payload),
    }


def converter_descriptor(numeric_backend: Mapping[str, Any]) -> dict[str, Any]:
    backend = _canonical_object(
        numeric_backend,
        "numeric backend",
        implementation_id=True,
    )
    source_root = Path(__file__).resolve(strict=True).parents[1]
    return {
        "errorPropagation": "componentwise abs(D65 matrix) times XYZ estimate",
        "implementationId": CONVERTER_IMPLEMENTATION_ID,
        "matrix": [list(row) for row in XYZ_TO_LINEAR_SRGB_D65],
        "numericBackend": backend,
        "operation": "unclamped CIE XYZ to scene-linear sRGB D65",
        "sourceFiles": [
            _source_file_descriptor(source_root / module_uri, module_uri)
            for module_uri in CONVERTER_SOURCE_FILES
        ],
    }


class _Summary:
    def __init__(self) -> None:
        self.records = 0
        self.minimum = [math.inf, math.inf, math.inf]
        self.maximum = [-math.inf, -math.inf, -math.inf]
        self.maximum_error = [0.0, 0.0, 0.0]
        self.negative_channel_mask_union = 0
        self.records_with_negative = 0
        self.source_union = 0
        self.convergence_intersection: int | None = None
        self.xyz_hash_chain = hashlib.sha256()
        self.spectral_hash_chain = hashlib.sha256()

    def add(self, record: LinearSrgbPixelRecord) -> None:
        self.records += 1
        negative_mask = 0
        for index, value in enumerate(record.linear_srgb):
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)
            self.maximum_error[index] = max(
                self.maximum_error[index],
                record.estimated_absolute_error_linear_srgb[index],
            )
            if value < 0.0:
                negative_mask |= 1 << index
        self.negative_channel_mask_union |= negative_mask
        if negative_mask:
            self.records_with_negative += 1
        self.source_union |= record.source_mask
        self.convergence_intersection = (
            record.convergence_mask
            if self.convergence_intersection is None
            else self.convergence_intersection & record.convergence_mask
        )
        self.xyz_hash_chain.update(record.input_cie_xyz_record_sha256)
        self.spectral_hash_chain.update(record.input_spectral_record_sha256)

    def descriptor(self) -> dict[str, Any]:
        return {
            "convergenceMaskIntersection": self.convergence_intersection,
            "inputCieXyzRecordSha256Chain": self.xyz_hash_chain.hexdigest(),
            "inputSpectralRecordSha256Chain": (
                self.spectral_hash_chain.hexdigest()
            ),
            "maximumEstimatedAbsoluteErrorLinearSrgb": self.maximum_error,
            "maximumLinearSrgb": self.maximum,
            "minimumLinearSrgb": self.minimum,
            "negativeChannelMaskUnion": self.negative_channel_mask_union,
            "recordCount": self.records,
            "recordsWithNegativeChannel": self.records_with_negative,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_union,
        }


def _strict_xyz_verification(
    xyz_manifest_path: Path,
    spectral_manifest_path: Path,
) -> dict[str, Any]:
    try:
        from scripts.verify_offline_cie_xyz import (
            validate_scientific_cie_xyz_frame,
        )
        from scripts.verify_nr_contract import ContractError

        return validate_scientific_cie_xyz_frame(
            xyz_manifest_path,
            spectral_manifest_path,
        )
    except ContractError as error:
        raise LinearRgbProductError(
            f"input CIE XYZ product failed strict verification: {error}"
        ) from error


def _input_xyz_identity(
    manifest: Mapping[str, Any],
    payload: bytes,
    report: Mapping[str, Any],
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


def _output_tile_uri(input_uri: str) -> str:
    pure = cie_product_module._normalized_uri(
        input_uri,
        "$.tiles[].inputCieXyzPayload.uri",
    )
    if pure.parent != PurePosixPath("tiles") or pure.suffix != ".cxyz":
        raise LinearRgbProductError("unsupported CIE XYZ tile URI")
    return f"tiles/{pure.stem}.lsrgb"


@dataclass(frozen=True, slots=True)
class LinearRgbProductPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    product_id: str
    product_sha256: str
    tile_count: int
    record_count: int


def convert_cie_xyz_product_to_linear_srgb(
    input_cie_xyz_manifest_path: Path | str,
    input_spectral_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    numeric_backend: Mapping[str, Any] | None = None,
) -> LinearRgbProductPublication:
    xyz_path = Path(input_cie_xyz_manifest_path).absolute()
    spectral_path = Path(input_spectral_manifest_path).absolute()
    if xyz_path.name != MANIFEST_NAME or spectral_path.name != MANIFEST_NAME:
        raise LinearRgbProductError("both input manifests must be named manifest.json")
    verification = _strict_xyz_verification(xyz_path, spectral_path)
    xyz_payload = cie_product_module._read_stable_regular(
        xyz_path,
        "input CIE XYZ manifest",
        16 << 20,
    )
    xyz_manifest = cie_product_module._strict_json(
        xyz_payload,
        "input CIE XYZ manifest",
    )
    if canonical_json_bytes(xyz_manifest) != xyz_payload:
        raise LinearRgbProductError("input CIE XYZ manifest is not canonical JSON")
    try:
        selected_backend = (
            numeric_backend
            if numeric_backend is not None
            else default_numeric_backend_descriptor()
        )
        backend = _canonical_object(
            selected_backend,
            "numeric backend",
            implementation_id=True,
        )
    except (CieProductError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise LinearRgbProductError(
            f"numeric backend authentication failed: {error}"
        ) from error
    converter = converter_descriptor(backend)
    input_xyz_identity = _input_xyz_identity(
        xyz_manifest,
        xyz_payload,
        verification,
    )
    spectral_identity = dict(xyz_manifest["inputSpectralProduct"])
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise LinearRgbProductError(
            f"refusing to overwrite existing output {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    summary = _Summary()
    output_tiles: list[dict[str, Any]] = []
    try:
        staging.mkdir(mode=0o700)
        (staging / "tiles").mkdir(mode=0o700)
        for tile_index, input_entry in enumerate(xyz_manifest["tiles"]):
            xyz_artifact = input_entry["outputPayload"]
            record_count = input_entry["recordCount"]
            expected_bytes = record_count * CIE_XYZ_RECORD_BYTES
            input_tile = cie_product_module._read_relative_regular(
                xyz_path.parent,
                xyz_artifact["uri"],
                f"input CIE XYZ tile {tile_index}",
                expected_bytes,
            )
            if (
                len(input_tile) != expected_bytes
                or xyz_artifact["byteLength"] != expected_bytes
                or _sha256(input_tile) != xyz_artifact["sha256"]
            ):
                raise LinearRgbProductError(
                    f"input CIE XYZ tile {tile_index} changed after verification"
                )
            output_payload = bytearray()
            for record_index in range(record_count):
                offset = record_index * CIE_XYZ_RECORD_BYTES
                input_record_payload = input_tile[
                    offset : offset + CIE_XYZ_RECORD_BYTES
                ]
                try:
                    input_record = unpack_cie_xyz_pixel(input_record_payload)
                    linear = apply_declared_xyz_to_linear_srgb(
                        input_record.mean_cie_xyz
                    )
                    errors = propagated_linear_srgb_absolute_error(
                        input_record.mean_estimated_absolute_error_xyz
                    )
                except (
                    CieProductError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise LinearRgbProductError(
                        f"input XYZ tile {tile_index} record {record_index} "
                        f"cannot produce linear sRGB: {error}"
                    ) from error
                output_record = LinearSrgbPixelRecord(
                    linear_srgb=linear,
                    estimated_absolute_error_linear_srgb=errors,
                    input_cie_xyz_record_sha256=hashlib.sha256(
                        input_record_payload
                    ).digest(),
                    input_spectral_record_sha256=(
                        input_record.input_record_sha256
                    ),
                    source_mask=input_record.source_mask,
                    convergence_mask=input_record.convergence_mask,
                )
                output_payload.extend(pack_linear_srgb_pixel(output_record))
                summary.add(output_record)
            output_uri = _output_tile_uri(xyz_artifact["uri"])
            output_bytes = bytes(output_payload)
            cie_product_module._atomic_write_no_replace(
                staging / output_uri,
                output_bytes,
            )
            output_tiles.append(
                {
                    "inputCieXyzPayload": dict(xyz_artifact),
                    "inputSpectralPayload": dict(input_entry["inputPayload"]),
                    "outputPayload": {
                        "byteLength": len(output_bytes),
                        "sha256": _sha256(output_bytes),
                        "uri": output_uri,
                    },
                    "recordCount": record_count,
                    "recordOrder": input_entry["recordOrder"],
                    "sampleIndex": input_entry["sampleIndex"],
                    "tile": dict(input_entry["tile"]),
                }
            )
        if summary.records != verification["recordCount"]:
            raise LinearRgbProductError("converted tiles do not cover every XYZ record")
        second_verification = _strict_xyz_verification(xyz_path, spectral_path)
        second_xyz_payload = cie_product_module._read_stable_regular(
            xyz_path,
            "input CIE XYZ manifest final snapshot",
            16 << 20,
        )
        if (
            second_verification != verification
            or second_xyz_payload != xyz_payload
        ):
            raise LinearRgbProductError(
                "input CIE XYZ product changed during conversion"
            )
        if converter_descriptor(backend) != converter:
            raise LinearRgbProductError("linear-sRGB converter source changed")

        layout = linear_srgb_pixel_layout_descriptor()
        configuration = {
            "cieDataset": xyz_manifest["cieDataset"],
            "converter": converter,
            "frame": xyz_manifest["frame"],
            "inputCieXyzProduct": input_xyz_identity,
            "inputSpectralProduct": spectral_identity,
            "pixelLayout": layout,
            "schema": PRODUCT_SCHEMA,
        }
        configuration_hash = _canonical_hash(configuration)
        summary_descriptor = summary.descriptor()
        product_identity = {
            "configurationSha256": configuration_hash,
            "schema": PRODUCT_SCHEMA,
            "summary": summary_descriptor,
            "tiles": output_tiles,
        }
        product_hash = _canonical_hash(product_identity)
        product_id = f"scientific-linear-srgb-frame-{product_hash[:24]}"
        manifest = {
            "cieDataset": xyz_manifest["cieDataset"],
            "converter": {
                "descriptor": converter,
                "descriptorSha256": _canonical_hash(converter),
            },
            "frame": xyz_manifest["frame"],
            "id": product_id,
            "inputCieXyzProduct": input_xyz_identity,
            "inputSpectralProduct": spectral_identity,
            "integrity": {
                "configurationSha256": configuration_hash,
                "manifestSidecar": SIDECAR_NAME,
                "productSha256": product_hash,
            },
            "pixelLayout": layout,
            "schema": PRODUCT_SCHEMA,
            "scientificStatus": dict(SCIENTIFIC_STATUS),
            "summary": summary_descriptor,
            "tiles": output_tiles,
        }
        manifest_payload = canonical_json_bytes(manifest)
        manifest_hash = _sha256(manifest_payload)
        sidecar = f"{manifest_hash}  {MANIFEST_NAME}\n".encode("ascii")
        cie_product_module._atomic_write_no_replace(
            staging / SIDECAR_NAME,
            sidecar,
        )
        cie_product_module._atomic_write_no_replace(
            staging / MANIFEST_NAME,
            manifest_payload,
        )
        cie_product_module._fsync_directory(staging)
        cie_product_module._promote_directory_no_replace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            cie_product_module._fsync_directory(output.parent)
        raise
    return LinearRgbProductPublication(
        output_directory=output,
        manifest_path=output / MANIFEST_NAME,
        manifest_sha256=manifest_hash,
        product_id=product_id,
        product_sha256=product_hash,
        tile_count=len(output_tiles),
        record_count=summary.records,
    )


__all__ = (
    "CONVERTER_IMPLEMENTATION_ID",
    "CONVERTER_SOURCE_FILES",
    "LinearRgbProductError",
    "LinearRgbProductPublication",
    "LinearSrgbPixelRecord",
    "MANIFEST_NAME",
    "PIXEL_LAYOUT_ID",
    "PRODUCT_SCHEMA",
    "PRODUCT_SCHEMA_ID",
    "RECORD_BYTES",
    "SCIENTIFIC_STATUS",
    "SIDECAR_NAME",
    "XYZ_TO_LINEAR_SRGB_D65",
    "apply_declared_xyz_to_linear_srgb",
    "convert_cie_xyz_product_to_linear_srgb",
    "converter_descriptor",
    "linear_srgb_pixel_layout_descriptor",
    "pack_linear_srgb_pixel",
    "propagated_linear_srgb_absolute_error",
    "unpack_linear_srgb_pixel",
)
