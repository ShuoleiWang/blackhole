"""Transactional SDR PPM16 quicklooks derived from linear-sRGB v1.

This module is deliberately downstream of the scientific colour products.  It
never rewrites the signed, unclamped linear-sRGB records.  Instead it publishes
a separate, explicitly display-referred product with a fixed manual exposure,
the versioned Rec.709-luminance Reinhard/gamut transform from
``offline.cie_color``, the IEC sRGB OETF, and deterministic RGB16 PPM encoding.

Passing the input verifier authenticates the source artifact and its lineage;
it does not make this quicklook an HDR product, a physics verification, or a
scientific primary quantity.
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
from offline.cie_color import CieColorError, LinearSrgb, derive_display_srgb
from offline.job import canonical_json_bytes
from offline.linear_rgb_product import (
    RECORD_BYTES as LINEAR_RECORD_BYTES,
    unpack_linear_srgb_pixel,
)
from offline.spectral_product import default_numeric_backend_descriptor


PRODUCT_SCHEMA: Final = "blackhole.sdr-display-quicklook/v1"
PRODUCT_SCHEMA_ID: Final = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-sdr-display-quicklook-v1.schema.json"
)
TRANSFORM_IMPLEMENTATION_ID: Final = (
    "blackhole.linear-srgb-to-sdr-display-quicklook/v1"
)
ENCODING_ID: Final = "blackhole.ppm16-rgb-srgb/big-endian-v1"
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
MAX_INPUT_TILE_BYTES: Final = 1 << 26
MAX_INPUT_TILE_PAYLOAD_BYTES: Final = (
    MAX_INPUT_TILE_BYTES // LINEAR_RECORD_BYTES * LINEAR_RECORD_BYTES
)
MAX_FRAME_PIXELS: Final = 1 << 23
MAX_TOTAL_PIXELS: Final = 1 << 24
MAX_SAMPLE_COUNT: Final = 64
RGB16_BYTES_PER_PIXEL: Final = 6
COVERAGE_BYTES_PER_PIXEL: Final = 1
PPM_HEADER_UPPER_BOUND_BYTES: Final = 19
DISPLAY_SOURCE_FILES: Final = (
    "assets/science/cie/CIE_xyz_1931_2deg.csv",
    "assets/science/cie/CIE_xyz_1931_2deg.csv_metadata.json",
    "offline/__init__.py",
    "offline/adaptive_frame.py",
    "offline/cie_color.py",
    "offline/cie_product.py",
    "offline/display_product.py",
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
    "schemas/offline-sdr-display-quicklook-v1.schema.json",
    "scripts/convert_offline_linear_srgb_to_sdr_display.py",
    "scripts/verify_nr_contract.py",
    "scripts/verify_offline_cie_xyz.py",
    "scripts/verify_offline_linear_srgb.py",
    "scripts/verify_offline_sdr_display.py",
    "scripts/verify_offline_spectral_frame.py",
)
PRODUCER_BULK_BUFFER_UPPER_BOUND_BYTES: Final = (
    (RGB16_BYTES_PER_PIXEL + COVERAGE_BYTES_PER_PIXEL) * MAX_TOTAL_PIXELS
    + MAX_INPUT_TILE_PAYLOAD_BYTES
    + LINEAR_RECORD_BYTES
)
VERIFIER_BULK_BUFFER_UPPER_BOUND_BYTES: Final = (
    (RGB16_BYTES_PER_PIXEL + COVERAGE_BYTES_PER_PIXEL) * MAX_TOTAL_PIXELS
    + 2
    * (
        RGB16_BYTES_PER_PIXEL * MAX_FRAME_PIXELS
        + PPM_HEADER_UPPER_BOUND_BYTES
    )
)
RGB16_STRUCT: Final = struct.Struct(">HHH")

DISPLAY_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "deterministic SDR display quicklook derived from authenticated "
            "signed unclamped linear-sRGB"
        ),
        "primaryQuantity": "display-referred-sRGB-code-values",
        "isDisplayImage": True,
        "isScientificPrimaryQuantity": False,
        "scientificLinearSrgbModified": False,
        "inputPhysicsRecomputed": False,
        "inputPhysicsVerified": False,
        "isHdr": False,
        "hdrTransferApplied": False,
        "manualExposureApplied": True,
        "negativeLinearChannelsClippedAtDisplayBoundary": True,
        "toneMappingApplied": True,
        "gamutMappingApplied": True,
        "srgbTransferCurveApplied": True,
        "integerQuantizationApplied": True,
        "isCameraSensorModel": False,
        "isAbsoluteHumanAppearanceModel": False,
        "prohibitedClaim": (
            "Do not describe this quicklook as HDR, a scientific linear-RGB "
            "replacement, a camera or appearance model, or a verification of "
            "the input physics."
        ),
    }
)


class DisplayProductError(RuntimeError):
    """An SDR quicklook product failed its closed artifact contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _strict_exposure(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("exposure must be a built-in int or float")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("exposure exceeds binary64 range") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("exposure must be finite and positive")
    return result


def _source_file_descriptor(path: Path, module_uri: str) -> dict[str, Any]:
    payload = cie_product_module._read_stable_regular(
        path,
        f"display producer source {module_uri}",
        4 << 20,
    )
    return {
        "byteLength": len(payload),
        "moduleUri": module_uri,
        "sha256": _sha256(payload),
    }


def _canonical_backend(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DisplayProductError(
            f"numeric backend must be finite canonical JSON: {error}"
        ) from error
    if not isinstance(result, dict) or not isinstance(
        result.get("implementationId"), str
    ) or not result["implementationId"]:
        raise DisplayProductError(
            "numeric backend requires a non-empty implementationId"
        )
    return result


def display_transform_descriptor(
    exposure: float,
    numeric_backend: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_exposure = _strict_exposure(exposure)
    backend = _canonical_backend(numeric_backend)
    current_backend = _canonical_backend(default_numeric_backend_descriptor())
    if backend != current_backend:
        raise DisplayProductError(
            "numeric backend must equal the authenticated current default v2"
        )
    source_root = Path(__file__).resolve(strict=True).parents[1]
    return {
        "colourspace": "sRGB-D65",
        "derivedDisplayOutput": True,
        "exposure": normalized_exposure,
        "exposureControl": "manual-fixed-scalar",
        "gamutPolicy": "uniform-max-channel-scale-if-needed",
        "implementationId": TRANSFORM_IMPLEMENTATION_ID,
        "luminanceCoefficients": [0.2126, 0.7152, 0.0722],
        "mappedLuminance": "Y/(1+Y)",
        "negativeLinearPolicy": "clip-to-zero-at-display-boundary",
        "nonNegativeLinearRgbRatioPreservedBeforeSrgbEncoding": True,
        "numericBackend": backend,
        "resourceLimits": {
            "bulkBufferAccounting": (
                "counts only RGB16 frame buffers, coverage buffers, one "
                "authenticated tile or up to two PPM payloads, and producer "
                "record scratch; excludes raw manifest/schema bytes, parsed "
                "JSON objects, and upstream verifier runtime memory"
            ),
            "coverageBytesPerPixel": COVERAGE_BYTES_PER_PIXEL,
            "maxFramePixels": MAX_FRAME_PIXELS,
            "maxInputTileBytes": MAX_INPUT_TILE_BYTES,
            "maxInputTilePayloadBytes": MAX_INPUT_TILE_PAYLOAD_BYTES,
            "maxSampleCount": MAX_SAMPLE_COUNT,
            "maxTotalPixels": MAX_TOTAL_PIXELS,
            "ppmHeaderUpperBoundBytes": PPM_HEADER_UPPER_BOUND_BYTES,
            "producerBulkBufferUpperBoundBytes": (
                PRODUCER_BULK_BUFFER_UPPER_BOUND_BYTES
            ),
            "rgb16BytesPerPixel": RGB16_BYTES_PER_PIXEL,
            "verifierBulkBufferUpperBoundBytes": (
                VERIFIER_BULK_BUFFER_UPPER_BOUND_BYTES
            ),
        },
        "sourceFiles": [
            _source_file_descriptor(source_root / uri, uri)
            for uri in DISPLAY_SOURCE_FILES
        ],
        "toneMapper": "reinhard-rec709-luminance-uniform-gamut/v2",
        "toneMappingDomain": "Rec.709-linear-luminance",
        "transferCurve": "IEC-61966-2-1-sRGB",
        "uniformRgbToneScale": "mappedY/Y = 1/(1+Y)",
    }


def ppm16_encoding_descriptor() -> dict[str, Any]:
    return {
        "channelOrder": "RGB",
        "encodedColourspace": "sRGB-D65",
        "header": "P6\n<width> <height>\n65535\n",
        "id": ENCODING_ID,
        "inputRowOrder": "bottom-to-top-local-y-then-x",
        "magic": "P6",
        "maxValue": 65535,
        "mediaType": "image/x-portable-pixmap",
        "outputRowOrder": "top-to-bottom-local-y-then-x",
        "quantization": "floor(encoded-sRGB*65535+0.5)",
        "sampleBits": 16,
        "sampleByteOrder": "big-endian",
    }


def _quantize_channel(value: float) -> int:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise DisplayProductError("encoded sRGB channel lies outside [0, 1]")
    result = math.floor(value * 65535.0 + 0.5)
    if result < 0 or result > 65535:
        raise DisplayProductError("RGB16 quantization overflowed")
    return result


def derive_display_rgb16(
    linear_rgb: tuple[float, float, float],
    *,
    exposure: float,
) -> tuple[int, int, int]:
    """Apply the frozen display transform and quantize to unsigned RGB16."""

    if len(linear_rgb) != 3:
        raise ValueError("linear_rgb must contain three values")
    try:
        source = LinearSrgb(*(float(value) for value in linear_rgb))
        display = derive_display_srgb(source, exposure=_strict_exposure(exposure))
    except (CieColorError, TypeError, ValueError, OverflowError) as error:
        raise DisplayProductError(f"display transform failed: {error}") from error
    return tuple(
        _quantize_channel(value) for value in (display.r, display.g, display.b)
    )  # type: ignore[return-value]


def encode_ppm16(
    width: int,
    height: int,
    rgb16_bottom_up: list[tuple[int, int, int]],
) -> bytes:
    """Encode lower-left-origin row-major pixels as canonical top-down PPM16."""

    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("PPM dimensions must be positive built-in integers")
    if len(rgb16_bottom_up) != width * height:
        raise ValueError("PPM pixel count does not match dimensions")
    compact = bytearray(width * height * RGB16_BYTES_PER_PIXEL)
    for index, pixel in enumerate(rgb16_bottom_up):
        if len(pixel) != 3 or any(
            type(channel) is not int or channel < 0 or channel > 65535
            for channel in pixel
        ):
            raise ValueError("PPM RGB16 pixels must contain three uint16 values")
        RGB16_STRUCT.pack_into(compact, index * RGB16_BYTES_PER_PIXEL, *pixel)
    return bytes(_encode_ppm16_buffer(width, height, compact))


def _encode_ppm16_buffer(
    width: int,
    height: int,
    rgb16_bottom_up: bytes | bytearray | memoryview,
) -> bytearray:
    """Encode one compact lower-left-origin RGB16 buffer without pixel objects."""

    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("PPM dimensions must be positive built-in integers")
    view = memoryview(rgb16_bottom_up).cast("B")
    expected = width * height * RGB16_BYTES_PER_PIXEL
    if len(view) != expected:
        raise ValueError("compact PPM buffer length does not match dimensions")
    header = f"P6\n{width} {height}\n65535\n".encode("ascii")
    if len(header) > PPM_HEADER_UPPER_BOUND_BYTES:
        raise ValueError("PPM header exceeds its declared byte limit")
    result = bytearray(header)
    row_bytes = width * RGB16_BYTES_PER_PIXEL
    for y in range(height - 1, -1, -1):
        start = y * row_bytes
        result.extend(view[start : start + row_bytes])
    return result


def _checked_frame_topology(
    width: Any,
    height: Any,
    sample_indices: tuple[Any, ...],
) -> int:
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise DisplayProductError("frame dimensions must be positive integers")
    if not sample_indices or len(sample_indices) > MAX_SAMPLE_COUNT:
        raise DisplayProductError("display sample count exceeds the limit")
    if len(set(sample_indices)) != len(sample_indices):
        raise DisplayProductError("frame sample indices must be unique")
    if any(
        type(sample_index) is not int
        or sample_index < 0
        or sample_index > 999999
        for sample_index in sample_indices
    ):
        raise DisplayProductError("frame contains an unsupported sample index")
    pixels_per_frame = width * height
    if pixels_per_frame > MAX_FRAME_PIXELS:
        raise DisplayProductError("display frame exceeds the 2^23-pixel limit")
    if pixels_per_frame * len(sample_indices) > MAX_TOTAL_PIXELS:
        raise DisplayProductError("display product exceeds the 2^24-pixel limit")
    return pixels_per_frame


def _checked_input_tile_bytes(record_count: Any) -> int:
    if type(record_count) is not int or record_count < 1:
        raise DisplayProductError("tile record count must be a positive integer")
    expected_bytes = record_count * LINEAR_RECORD_BYTES
    if expected_bytes > MAX_INPUT_TILE_BYTES:
        raise DisplayProductError("input tile exceeds the 2^26-byte limit")
    return expected_bytes


def _strict_linear_verification(
    linear_manifest_path: Path,
    xyz_manifest_path: Path,
    spectral_manifest_path: Path,
) -> dict[str, Any]:
    try:
        from scripts.verify_nr_contract import ContractError
        from scripts.verify_offline_linear_srgb import (
            validate_scientific_linear_srgb_frame,
        )

        return validate_scientific_linear_srgb_frame(
            linear_manifest_path,
            xyz_manifest_path,
            spectral_manifest_path,
        )
    except ContractError as error:
        raise DisplayProductError(
            f"input linear-sRGB product failed strict verification: {error}"
        ) from error


def _input_linear_identity(
    manifest: Mapping[str, Any],
    manifest_payload: bytes,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": manifest["id"],
        "inputPhysicsVerified": report["inputPhysicsVerified"],
        "manifestSha256": _sha256(manifest_payload),
        "matrixTransformVerified": report["matrixTransformVerified"],
        "maximumUlpDifference": report["maximumUlpDifference"],
        "productSha256": manifest["integrity"]["productSha256"],
        "recordCount": report["recordCount"],
        "schema": manifest["schema"],
        "structuralStatus": report["status"],
        "tileCount": report["tileCount"],
    }


def _validate_input_uri(input_uri: str) -> None:
    pure = cie_product_module._normalized_uri(
        input_uri,
        "$.tiles[].inputLinearSrgbPayload.uri",
    )
    if pure.parent != PurePosixPath("tiles") or pure.suffix != ".lsrgb":
        raise DisplayProductError("unsupported linear-sRGB tile URI")
    return None


def _frame_output_uri(sample_index: int) -> str:
    if type(sample_index) is not int or sample_index < 0 or sample_index > 999999:
        raise DisplayProductError("sample index cannot be represented canonically")
    return f"images/sample-{sample_index:06d}.ppm"


class _Summary:
    def __init__(self) -> None:
        self.records = 0
        self.minimum = [65535, 65535, 65535]
        self.maximum = [0, 0, 0]
        self.negative_inputs = 0
        self.black_pixels = 0
        self.saturated_pixels = 0
        self.input_record_chain = hashlib.sha256()
        self.output_payload_chain = hashlib.sha256()

    def add_record(
        self,
        input_record_payload: bytes,
        linear_rgb: tuple[float, float, float],
        rgb16: tuple[int, int, int],
    ) -> None:
        self.records += 1
        for index, value in enumerate(rgb16):
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)
        self.negative_inputs += int(any(value < 0.0 for value in linear_rgb))
        self.black_pixels += int(rgb16 == (0, 0, 0))
        self.saturated_pixels += int(any(value == 65535 for value in rgb16))
        self.input_record_chain.update(hashlib.sha256(input_record_payload).digest())

    def add_output(self, payload: bytes) -> None:
        self.output_payload_chain.update(hashlib.sha256(payload).digest())

    def descriptor(self) -> dict[str, Any]:
        return {
            "inputLinearSrgbRecordSha256Chain": self.input_record_chain.hexdigest(),
            "maximumRgb16": self.maximum,
            "minimumRgb16": self.minimum,
            "outputPpmSha256Chain": self.output_payload_chain.hexdigest(),
            "pixelCount": self.records,
            "pixelsAtBlack": self.black_pixels,
            "pixelsWithAnyNegativeLinearInput": self.negative_inputs,
            "pixelsWithAnySaturatedCode": self.saturated_pixels,
        }


@dataclass(frozen=True, slots=True)
class DisplayProductPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    product_id: str
    product_sha256: str
    image_count: int
    tile_count: int
    pixel_count: int


def convert_linear_srgb_to_sdr_display(
    input_linear_srgb_manifest_path: Path | str,
    input_cie_xyz_manifest_path: Path | str,
    input_spectral_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    exposure: float,
) -> DisplayProductPublication:
    """Publish a no-overwrite, versioned SDR PPM16 display derivative."""

    linear_path = Path(input_linear_srgb_manifest_path).absolute()
    xyz_path = Path(input_cie_xyz_manifest_path).absolute()
    spectral_path = Path(input_spectral_manifest_path).absolute()
    if any(path.name != MANIFEST_NAME for path in (linear_path, xyz_path, spectral_path)):
        raise DisplayProductError("all input manifests must be named manifest.json")
    normalized_exposure = _strict_exposure(exposure)
    verification = _strict_linear_verification(linear_path, xyz_path, spectral_path)
    linear_payload = cie_product_module._read_stable_regular(
        linear_path,
        "input linear-sRGB manifest",
        16 << 20,
    )
    linear_manifest = cie_product_module._strict_json(
        linear_payload,
        "input linear-sRGB manifest",
    )
    if canonical_json_bytes(linear_manifest) != linear_payload:
        raise DisplayProductError("input linear-sRGB manifest is not canonical JSON")
    backend = _canonical_backend(default_numeric_backend_descriptor())
    transform = display_transform_descriptor(normalized_exposure, backend)
    encoding = ppm16_encoding_descriptor()
    input_identity = _input_linear_identity(
        linear_manifest,
        linear_payload,
        verification,
    )

    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise DisplayProductError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    summary = _Summary()
    input_tiles: list[dict[str, Any]] = []
    output_images: list[dict[str, Any]] = []
    frame_width = linear_manifest["frame"]["widthPixels"]
    frame_height = linear_manifest["frame"]["heightPixels"]
    sample_indices = tuple(linear_manifest["frame"]["sampleIndices"])
    pixels_per_frame = _checked_frame_topology(
        frame_width, frame_height, sample_indices
    )
    frame_pixels: dict[int, bytearray] = {
        sample_index: bytearray(pixels_per_frame * RGB16_BYTES_PER_PIXEL)
        for sample_index in sample_indices
    }
    frame_coverage: dict[int, bytearray] = {
        sample_index: bytearray(pixels_per_frame)
        for sample_index in sample_indices
    }
    try:
        staging.mkdir(mode=0o700)
        (staging / "images").mkdir(mode=0o700)
        for tile_index, input_entry in enumerate(linear_manifest["tiles"]):
            input_artifact = input_entry["outputPayload"]
            record_count = input_entry["recordCount"]
            tile = input_entry["tile"]
            if record_count != tile["width"] * tile["height"]:
                raise DisplayProductError(
                    f"input tile {tile_index} record count does not match dimensions"
                )
            try:
                expected_bytes = _checked_input_tile_bytes(record_count)
            except DisplayProductError as error:
                raise DisplayProductError(
                    f"input tile {tile_index}: {error}"
                ) from error
            _validate_input_uri(input_artifact["uri"])
            sample_index = input_entry["sampleIndex"]
            if sample_index not in frame_pixels:
                raise DisplayProductError(
                    f"input tile {tile_index} has undeclared sample index"
                )
            if (
                tile["x"] + tile["width"] > frame_width
                or tile["y"] + tile["height"] > frame_height
            ):
                raise DisplayProductError(
                    f"input tile {tile_index} lies outside the declared frame"
                )
            target = frame_pixels[sample_index]
            target_coverage = frame_coverage[sample_index]
            raw_tile = cie_product_module._read_relative_regular(
                linear_path.parent,
                input_artifact["uri"],
                f"input linear-sRGB tile {tile_index}",
                expected_bytes,
            )
            if (
                len(raw_tile) != expected_bytes
                or input_artifact["byteLength"] != expected_bytes
                or _sha256(raw_tile) != input_artifact["sha256"]
            ):
                raise DisplayProductError(
                    f"input linear-sRGB tile {tile_index} changed after verification"
                )
            for record_index in range(record_count):
                offset = record_index * LINEAR_RECORD_BYTES
                raw_record = raw_tile[offset : offset + LINEAR_RECORD_BYTES]
                try:
                    record = unpack_linear_srgb_pixel(raw_record)
                    rgb16 = derive_display_rgb16(
                        record.linear_srgb,
                        exposure=normalized_exposure,
                    )
                except (TypeError, ValueError, OverflowError) as error:
                    raise DisplayProductError(
                        f"input tile {tile_index} record {record_index} cannot be "
                        f"display encoded: {error}"
                    ) from error
                local_y, local_x = divmod(record_index, tile["width"])
                target_index = (
                    (tile["y"] + local_y) * frame_width
                    + tile["x"]
                    + local_x
                )
                if target_coverage[target_index]:
                    raise DisplayProductError(
                        f"input tile {tile_index} overlaps another tile"
                    )
                RGB16_STRUCT.pack_into(
                    target,
                    target_index * RGB16_BYTES_PER_PIXEL,
                    *rgb16,
                )
                target_coverage[target_index] = 1
                summary.add_record(raw_record, record.linear_srgb, rgb16)
            input_tiles.append(
                {
                    "inputLinearSrgbPayload": dict(input_artifact),
                    "recordCount": record_count,
                    "recordOrder": input_entry["recordOrder"],
                    "sampleIndex": input_entry["sampleIndex"],
                    "tile": dict(tile),
                }
            )
            del raw_tile
            del raw_record
        if summary.records != verification["recordCount"]:
            raise DisplayProductError(
                "display tiles do not cover every linear-sRGB record"
            )
        for sample_index in sample_indices:
            if 0 in frame_coverage[sample_index]:
                raise DisplayProductError(
                    f"tiles do not cover every pixel of sample {sample_index}"
                )
            ppm = _encode_ppm16_buffer(
                frame_width,
                frame_height,
                frame_pixels[sample_index],
            )
            output_uri = _frame_output_uri(sample_index)
            cie_product_module._atomic_write_no_replace(staging / output_uri, ppm)
            summary.add_output(ppm)
            output_images.append(
                {
                    "heightPixels": frame_height,
                    "outputPayload": {
                        "byteLength": len(ppm),
                        "sha256": _sha256(ppm),
                        "uri": output_uri,
                    },
                    "pixelCount": frame_width * frame_height,
                    "sampleIndex": sample_index,
                    "widthPixels": frame_width,
                }
            )
            del ppm
        frame_pixels.clear()
        frame_coverage.clear()
        del target, target_coverage
        second_verification = _strict_linear_verification(
            linear_path, xyz_path, spectral_path
        )
        second_linear_payload = cie_product_module._read_stable_regular(
            linear_path,
            "input linear-sRGB manifest final snapshot",
            16 << 20,
        )
        if second_verification != verification or second_linear_payload != linear_payload:
            raise DisplayProductError(
                "input linear-sRGB product changed during conversion"
            )
        if display_transform_descriptor(normalized_exposure, backend) != transform:
            raise DisplayProductError("display producer source changed")

        configuration = {
            "displayTransform": transform,
            "encoding": encoding,
            "frame": linear_manifest["frame"],
            "inputCieXyzProduct": linear_manifest["inputCieXyzProduct"],
            "inputLinearSrgbProduct": input_identity,
            "inputSpectralProduct": linear_manifest["inputSpectralProduct"],
            "schema": PRODUCT_SCHEMA,
        }
        configuration_hash = _canonical_hash(configuration)
        summary_descriptor = summary.descriptor()
        identity = {
            "configurationSha256": configuration_hash,
            "images": output_images,
            "schema": PRODUCT_SCHEMA,
            "summary": summary_descriptor,
            "tiles": input_tiles,
        }
        product_hash = _canonical_hash(identity)
        product_id = f"sdr-display-quicklook-{product_hash[:24]}"
        manifest = {
            "displayStatus": dict(DISPLAY_STATUS),
            "displayTransform": {
                "descriptor": transform,
                "descriptorSha256": _canonical_hash(transform),
            },
            "encoding": encoding,
            "frame": linear_manifest["frame"],
            "id": product_id,
            "images": output_images,
            "inputCieXyzProduct": linear_manifest["inputCieXyzProduct"],
            "inputLinearSrgbProduct": input_identity,
            "inputSpectralProduct": linear_manifest["inputSpectralProduct"],
            "integrity": {
                "configurationSha256": configuration_hash,
                "manifestSidecar": SIDECAR_NAME,
                "productSha256": product_hash,
            },
            "schema": PRODUCT_SCHEMA,
            "summary": summary_descriptor,
            "tiles": input_tiles,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_hash = _sha256(manifest_bytes)
        sidecar = f"{manifest_hash}  {MANIFEST_NAME}\n".encode("ascii")
        cie_product_module._atomic_write_no_replace(staging / SIDECAR_NAME, sidecar)
        cie_product_module._atomic_write_no_replace(staging / MANIFEST_NAME, manifest_bytes)
        cie_product_module._fsync_directory(staging)
        cie_product_module._promote_directory_no_replace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            cie_product_module._fsync_directory(output.parent)
        raise
    return DisplayProductPublication(
        output_directory=output,
        manifest_path=output / MANIFEST_NAME,
        manifest_sha256=manifest_hash,
        product_id=product_id,
        product_sha256=product_hash,
        image_count=len(output_images),
        tile_count=len(input_tiles),
        pixel_count=summary.records,
    )


__all__ = (
    "DISPLAY_STATUS",
    "DISPLAY_SOURCE_FILES",
    "DisplayProductError",
    "DisplayProductPublication",
    "ENCODING_ID",
    "MANIFEST_NAME",
    "PRODUCT_SCHEMA",
    "PRODUCT_SCHEMA_ID",
    "SIDECAR_NAME",
    "TRANSFORM_IMPLEMENTATION_ID",
    "convert_linear_srgb_to_sdr_display",
    "derive_display_rgb16",
    "display_transform_descriptor",
    "encode_ppm16",
    "ppm16_encoding_descriptor",
)
