#!/usr/bin/env python3
"""Independently verify an SDR PPM16 quicklook against linear-sRGB v1.

The verifier authenticates the complete upstream linear/XYZ/spectral lineage,
then locally reimplements the display transform, quantization, row inversion,
and PPM16 byte stream.  It does not call the display producer or its colour
helpers.  Success verifies this display derivative, not HDR or input physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import struct
import sys
from typing import Any, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.job import canonical_json_bytes
from offline.display_product import DISPLAY_SOURCE_FILES
from offline.spectral_product import default_numeric_backend_descriptor
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
)
from scripts.verify_offline_linear_srgb import (
    validate_scientific_linear_srgb_frame,
)


DEFAULT_SCHEMA = ROOT / "schemas" / "offline-sdr-display-quicklook-v1.schema.json"
SCHEMA_ID = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-sdr-display-quicklook-v1.schema.json"
)
PRODUCT_SCHEMA = "blackhole.sdr-display-quicklook/v1"
TRANSFORM_IMPLEMENTATION_ID = "blackhole.linear-srgb-to-sdr-display-quicklook/v1"
LINEAR_RECORD_STRUCT = struct.Struct("<6d32s32sII")
LINEAR_RECORD_BYTES = LINEAR_RECORD_STRUCT.size
MAX_INPUT_TILE_BYTES = 1 << 26
MAX_INPUT_TILE_PAYLOAD_BYTES = (
    MAX_INPUT_TILE_BYTES // LINEAR_RECORD_BYTES * LINEAR_RECORD_BYTES
)
MAX_FRAME_PIXELS = 1 << 23
MAX_TOTAL_PIXELS = 1 << 24
MAX_SAMPLE_COUNT = 64
RGB16_BYTES_PER_PIXEL = 6
COVERAGE_BYTES_PER_PIXEL = 1
PPM_HEADER_UPPER_BOUND_BYTES = 19
PRODUCER_BULK_BUFFER_UPPER_BOUND_BYTES = (
    (RGB16_BYTES_PER_PIXEL + COVERAGE_BYTES_PER_PIXEL) * MAX_TOTAL_PIXELS
    + MAX_INPUT_TILE_PAYLOAD_BYTES
    + LINEAR_RECORD_BYTES
)
VERIFIER_BULK_BUFFER_UPPER_BOUND_BYTES = (
    (RGB16_BYTES_PER_PIXEL + COVERAGE_BYTES_PER_PIXEL) * MAX_TOTAL_PIXELS
    + 2
    * (
        RGB16_BYTES_PER_PIXEL * MAX_FRAME_PIXELS
        + PPM_HEADER_UPPER_BOUND_BYTES
    )
)
RGB16_STRUCT = struct.Struct(">HHH")

EXPECTED_STATUS = {
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

EXPECTED_ENCODING = {
    "channelOrder": "RGB",
    "encodedColourspace": "sRGB-D65",
    "header": "P6\n<width> <height>\n65535\n",
    "id": "blackhole.ppm16-rgb-srgb/big-endian-v1",
    "inputRowOrder": "bottom-to-top-local-y-then-x",
    "magic": "P6",
    "maxValue": 65535,
    "mediaType": "image/x-portable-pixmap",
    "outputRowOrder": "top-to-bottom-local-y-then-x",
    "quantization": "floor(encoded-sRGB*65535+0.5)",
    "sampleBits": 16,
    "sampleByteOrder": "big-endian",
}


class OfflineSdrDisplayContractError(ContractError):
    """An SDR display derivative contract failed closed."""


def fail(path: str, message: str) -> NoReturn:
    raise OfflineSdrDisplayContractError(f"{path}: {message}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    try:
        return _sha256(canonical_json_bytes(value))
    except (OverflowError, TypeError, ValueError) as error:
        fail("$", f"value is not finite canonical JSON: {error}")


def _source_descriptor() -> list[dict[str, Any]]:
    result = []
    for module_uri in DISPLAY_SOURCE_FILES:
        payload = _read_stable_file(
            ROOT / module_uri,
            f"display producer source {module_uri}",
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


def _current_numeric_backend() -> dict[str, Any]:
    try:
        value = json.loads(canonical_json_bytes(default_numeric_backend_descriptor()))
    except (
        json.JSONDecodeError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        fail(
            "$.displayTransform.descriptor.numericBackend",
            f"unable to authenticate current numeric backend: {error}",
        )
    if not isinstance(value, dict):
        fail(
            "$.displayTransform.descriptor.numericBackend",
            "current numeric backend is not an object",
        )
    return value


def _input_linear_identity(
    manifest: dict[str, Any],
    payload: bytes,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": manifest["id"],
        "inputPhysicsVerified": report["inputPhysicsVerified"],
        "manifestSha256": _sha256(payload),
        "matrixTransformVerified": report["matrixTransformVerified"],
        "maximumUlpDifference": report["maximumUlpDifference"],
        "productSha256": manifest["integrity"]["productSha256"],
        "recordCount": report["recordCount"],
        "schema": manifest["schema"],
        "structuralStatus": report["status"],
        "tileCount": report["tileCount"],
    }


def _validate_input_uri(input_uri: str) -> None:
    if not input_uri.startswith("tiles/") or not input_uri.endswith(".lsrgb"):
        fail("$.tiles[].inputLinearSrgbPayload.uri", "unsupported input URI")
    stem = input_uri.removeprefix("tiles/").removesuffix(".lsrgb")
    if not stem or "/" in stem or ".." in stem:
        fail("$.tiles[].inputLinearSrgbPayload.uri", "non-canonical input URI")


def _expected_frame_uri(sample_index: int) -> str:
    if sample_index < 0 or sample_index > 999999:
        fail("$.images[].sampleIndex", "sample index cannot be represented")
    return f"images/sample-{sample_index:06d}.ppm"


def _srgb_encode(linear: float) -> float:
    if linear <= 0.0031308:
        return 12.92 * linear
    return 1.055 * linear ** (1.0 / 2.4) - 0.055


def _derive_rgb16(
    linear_rgb: tuple[float, float, float],
    exposure: float,
    path: str,
) -> tuple[int, int, int]:
    if not math.isfinite(exposure) or exposure <= 0.0:
        fail(path, "manual exposure must be finite and positive")
    if any(not math.isfinite(channel) for channel in linear_rgb):
        fail(path, "input linear-sRGB channels must be finite")
    exposed_values: list[float] = []
    for channel in linear_rgb:
        product = channel * exposure
        if not math.isfinite(product):
            fail(path, "manual exposure overflowed before display clipping")
        exposed_values.append(max(0.0, product))
    exposed = tuple(exposed_values)
    try:
        luminance = math.fsum(
            coefficient * channel
            for coefficient, channel in zip((0.2126, 0.7152, 0.0722), exposed)
        )
    except OverflowError:
        fail(path, "Rec.709 display luminance overflowed")
    if not math.isfinite(luminance) or luminance < 0.0:
        fail(path, "Rec.709 display luminance is invalid")
    if luminance == 0.0:
        if any(channel > 0.0 for channel in exposed):
            fail(path, "positive RGB has zero representable luminance")
        tone_mapped = (0.0, 0.0, 0.0)
    else:
        scale = 1.0 / (1.0 + luminance)
        tone_mapped = tuple(channel * scale for channel in exposed)
    maximum = max(tone_mapped)
    display_linear = (
        tuple(channel / maximum for channel in tone_mapped)
        if maximum > 1.0
        else tone_mapped
    )
    encoded = tuple(
        min(1.0, max(0.0, _srgb_encode(channel))) for channel in display_linear
    )
    if any(not math.isfinite(channel) for channel in encoded):
        fail(path, "sRGB encoding failed")
    return tuple(
        math.floor(channel * 65535.0 + 0.5) for channel in encoded
    )  # type: ignore[return-value]


def _encode_ppm16_buffer(
    width: int,
    height: int,
    pixels_bottom_up: bytes | bytearray | memoryview,
) -> bytearray:
    view = memoryview(pixels_bottom_up).cast("B")
    if len(view) != width * height * RGB16_BYTES_PER_PIXEL:
        fail("$.images", "compact RGB16 buffer length disagrees with frame")
    header = f"P6\n{width} {height}\n65535\n".encode("ascii")
    if len(header) > PPM_HEADER_UPPER_BOUND_BYTES:
        fail("$.images", "PPM header exceeds its declared byte limit")
    result = bytearray(header)
    row_bytes = width * RGB16_BYTES_PER_PIXEL
    for y in range(height - 1, -1, -1):
        offset = y * row_bytes
        result.extend(view[offset : offset + row_bytes])
    return result


def _checked_frame_topology(
    width: Any,
    height: Any,
    sample_indices: tuple[Any, ...],
) -> int:
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        fail("$.frame", "frame dimensions must be positive integers")
    if not sample_indices or len(sample_indices) > MAX_SAMPLE_COUNT:
        fail("$.frame.sampleIndices", "display sample count exceeds the limit")
    if len(set(sample_indices)) != len(sample_indices):
        fail("$.frame.sampleIndices", "sample indices must be unique")
    if any(
        type(sample_index) is not int
        or sample_index < 0
        or sample_index > 999999
        for sample_index in sample_indices
    ):
        fail("$.frame.sampleIndices", "unsupported sample index")
    pixels_per_frame = width * height
    if pixels_per_frame > MAX_FRAME_PIXELS:
        fail("$.frame", "display frame exceeds the 2^23-pixel limit")
    if pixels_per_frame * len(sample_indices) > MAX_TOTAL_PIXELS:
        fail("$.frame", "display product exceeds the 2^24-pixel limit")
    return pixels_per_frame


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_open_stable(
    descriptor: int,
    expected_bytes: int,
    path: str,
) -> bytes:
    if expected_bytes < 0:
        fail(path, "negative expected byte length")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(path, "final snapshot requires a regular file")
        if before.st_size != expected_bytes:
            fail(path, "final snapshot byte length changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except ContractError:
        raise
    except OSError as error:
        fail(path, f"unable to read anchored final snapshot: {error}")
    if _stat_identity(before) != _stat_identity(after):
        fail(path, "file identity changed during final snapshot read")
    if len(payload) != expected_bytes:
        fail(path, "final snapshot read length changed")
    return payload


def _validate_final_self_closure(
    root: Path,
    manifest: dict[str, Any],
    manifest_payload: bytes,
    images: list[dict[str, Any]],
    allowed: set[str],
) -> None:
    """Close the final self-observation window over anchored product FDs.

    The product and image directories, manifest, sidecar, and every PPM are
    opened before the final no-extra scan.  After that scan returns, all files
    are read again through those same descriptors and both directory-entry and
    inode/stat snapshots must remain identical.  The caller returns
    immediately after this unified closure.
    """

    expected_sidecar = f"{_sha256(manifest_payload)}  manifest.json\n".encode(
        "ascii"
    )
    expectations: dict[str, tuple[int, str, bytes | None]] = {
        "manifest.json": (
            len(manifest_payload),
            _sha256(manifest_payload),
            manifest_payload,
        ),
        manifest["integrity"]["manifestSidecar"]: (
            len(expected_sidecar),
            _sha256(expected_sidecar),
            expected_sidecar,
        ),
    }
    for image_index, image_entry in enumerate(images):
        artifact = image_entry["outputPayload"]
        uri = artifact["uri"]
        pure = PurePosixPath(uri)
        if pure.parent != PurePosixPath("images") or len(pure.parts) != 2:
            fail(f"$.images[{image_index}].outputPayload.uri", "invalid final URI")
        if uri in expectations:
            fail(f"$.images[{image_index}].outputPayload.uri", "duplicate final URI")
        header = (
            f"P6\n{image_entry['widthPixels']} "
            f"{image_entry['heightPixels']}\n65535\n"
        ).encode("ascii")
        if len(header) > PPM_HEADER_UPPER_BOUND_BYTES:
            fail(
                f"$.images[{image_index}].outputPayload",
                "PPM header exceeds its declared byte limit",
            )
        expected_bytes = len(header) + (
            image_entry["pixelCount"] * RGB16_BYTES_PER_PIXEL
        )
        if artifact["byteLength"] != expected_bytes:
            fail(
                f"$.images[{image_index}].outputPayload.byteLength",
                "final PPM length disagrees with topology",
            )
        expectations[uri] = (expected_bytes, artifact["sha256"], None)
    if set(expectations) != allowed:
        fail("$files", "final self-closure does not cover every allowed file")

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    root_descriptor: int | None = None
    images_descriptor: int | None = None
    file_descriptors: dict[str, int] = {}
    try:
        root_descriptor = os.open(root, directory_flags)
        root_path_identity = _stat_identity(os.stat(root, follow_symlinks=False))
        if root_path_identity != _stat_identity(os.fstat(root_descriptor)):
            fail("$files", "product root path is not the anchored directory")
        expected_root_entries = {
            "images",
            "manifest.json",
            manifest["integrity"]["manifestSidecar"],
        }
        if set(os.listdir(root_descriptor)) != expected_root_entries:
            fail("$files", "final product root entries differ from the contract")
        images_descriptor = os.open(
            "images",
            directory_flags,
            dir_fd=root_descriptor,
        )
        expected_image_names = {
            PurePosixPath(uri).name
            for uri in expectations
            if PurePosixPath(uri).parent == PurePosixPath("images")
        }
        if set(os.listdir(images_descriptor)) != expected_image_names:
            fail("$files", "final images entries differ from the contract")

        initial_identities: dict[str, tuple[int, int, int, int, int, int]] = {
            ".": _stat_identity(os.fstat(root_descriptor)),
            "images": _stat_identity(os.fstat(images_descriptor)),
        }
        for uri in sorted(expectations):
            pure = PurePosixPath(uri)
            parent_descriptor = (
                root_descriptor
                if pure.parent == PurePosixPath(".")
                else images_descriptor
            )
            descriptor = os.open(
                pure.name,
                file_flags,
                dir_fd=parent_descriptor,
            )
            file_descriptors[uri] = descriptor
            initial_identities[uri] = _stat_identity(os.fstat(descriptor))

        # This deliberately remains inside the unified closure.  A mutation
        # injected after its directory walk is observed by the anchored reread
        # and identity comparison below.
        _validate_no_extra_files(root, allowed)

        for uri in sorted(expectations):
            expected_bytes, expected_sha256, exact_payload = expectations[uri]
            payload = _read_open_stable(
                file_descriptors[uri],
                expected_bytes,
                f"$files.final[{uri!r}]",
            )
            if _sha256(payload) != expected_sha256:
                fail(f"$files.final[{uri!r}]", "final payload hash changed")
            if exact_payload is not None and payload != exact_payload:
                fail(f"$files.final[{uri!r}]", "final exact payload changed")
            del payload

        if set(os.listdir(root_descriptor)) != expected_root_entries:
            fail("$files", "product root entries changed during final closure")
        if set(os.listdir(images_descriptor)) != expected_image_names:
            fail("$files", "images entries changed during final closure")
        final_identities = {
            ".": _stat_identity(os.fstat(root_descriptor)),
            "images": _stat_identity(os.fstat(images_descriptor)),
            **{
                uri: _stat_identity(os.fstat(descriptor))
                for uri, descriptor in file_descriptors.items()
            },
        }
        if final_identities != initial_identities:
            fail("$files", "directory or file identity changed during final closure")
        if _stat_identity(os.stat(root, follow_symlinks=False)) != root_path_identity:
            fail("$files", "product root path changed during final closure")
        anchored_images = os.stat(
            "images",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if _stat_identity(anchored_images) != initial_identities["images"]:
            fail("$files", "images directory path changed during final closure")
    except ContractError:
        raise
    except OSError as error:
        fail("$files", f"unable to establish anchored final closure: {error}")
    finally:
        for descriptor in file_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if images_descriptor is not None:
            try:
                os.close(images_descriptor)
            except OSError:
                pass
        if root_descriptor is not None:
            try:
                os.close(root_descriptor)
            except OSError:
                pass


class _Summary:
    def __init__(self) -> None:
        self.records = 0
        self.minimum = [65535, 65535, 65535]
        self.maximum = [0, 0, 0]
        self.negative_inputs = 0
        self.black_pixels = 0
        self.saturated_pixels = 0
        self.input_chain = hashlib.sha256()
        self.output_chain = hashlib.sha256()

    def add_record(
        self,
        raw: bytes,
        linear: tuple[float, float, float],
        rgb16: tuple[int, int, int],
    ) -> None:
        self.records += 1
        for index, value in enumerate(rgb16):
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)
        self.negative_inputs += int(any(value < 0.0 for value in linear))
        self.black_pixels += int(rgb16 == (0, 0, 0))
        self.saturated_pixels += int(any(value == 65535 for value in rgb16))
        self.input_chain.update(hashlib.sha256(raw).digest())

    def add_output(self, payload: bytes) -> None:
        self.output_chain.update(hashlib.sha256(payload).digest())

    def descriptor(self) -> dict[str, Any]:
        return {
            "inputLinearSrgbRecordSha256Chain": self.input_chain.hexdigest(),
            "maximumRgb16": self.maximum,
            "minimumRgb16": self.minimum,
            "outputPpmSha256Chain": self.output_chain.hexdigest(),
            "pixelCount": self.records,
            "pixelsAtBlack": self.black_pixels,
            "pixelsWithAnyNegativeLinearInput": self.negative_inputs,
            "pixelsWithAnySaturatedCode": self.saturated_pixels,
        }


def validate_sdr_display_quicklook(
    manifest_path: Path | str,
    input_linear_srgb_manifest_path: Path | str,
    input_cie_xyz_manifest_path: Path | str,
    input_spectral_manifest_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    path = Path(manifest_path).absolute()
    linear_path = Path(input_linear_srgb_manifest_path).absolute()
    xyz_path = Path(input_cie_xyz_manifest_path).absolute()
    spectral_path = Path(input_spectral_manifest_path).absolute()
    if any(item.name != "manifest.json" for item in (path, linear_path, xyz_path, spectral_path)):
        fail("$", "all v1 manifests must be named manifest.json")

    schema_payload = _read_stable_file(Path(schema_path).absolute(), "$schema", 4 << 20)
    default_schema_payload = _read_stable_file(
        DEFAULT_SCHEMA, "$defaultSchema", 4 << 20
    )
    if schema_payload != default_schema_payload:
        fail("$schema", "schema must byte-match the repository default")
    schema = _strict_json(schema_payload, "$schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        fail("$schema.$schema", "only Draft 2020-12 is supported")
    if schema.get("$id") != SCHEMA_ID:
        fail("$schema.$id", "unexpected SDR quicklook schema id")
    audit_schema_dialect(schema)

    manifest_payload = _read_stable_file(path, "$", 16 << 20)
    try:
        manifest = _strict_json(manifest_payload, "$")
    except ContractError:
        raise
    except (OverflowError, ValueError) as error:
        fail("$", f"JSON integer exceeds verifier numeric limits: {error}")
    try:
        validate_json_schema(manifest, schema, schema)
    except OverflowError as error:
        fail("$", f"JSON number exceeds verifier binary64 range: {error}")
    try:
        canonical_manifest = canonical_json_bytes(manifest)
    except (OverflowError, TypeError, ValueError) as error:
        fail("$", f"manifest cannot be encoded as finite canonical JSON: {error}")
    if canonical_manifest != manifest_payload:
        fail("$", "manifest is not canonical JSON")
    _validate_sidecar(path.parent, manifest, manifest_payload)
    if manifest["schema"] != PRODUCT_SCHEMA:
        fail("$.schema", "unsupported display product schema")
    if manifest["displayStatus"] != EXPECTED_STATUS:
        fail("$.displayStatus", "display/scientific boundary drifted")
    if manifest["encoding"] != EXPECTED_ENCODING:
        fail("$.encoding", "PPM16 encoding contract drifted")

    try:
        linear_report = validate_scientific_linear_srgb_frame(
            linear_path, xyz_path, spectral_path
        )
    except ContractError as error:
        fail("$.inputLinearSrgbProduct", f"input verification failed: {error}")
    linear_payload = _read_stable_file(linear_path, "$linear", 16 << 20)
    linear_manifest = _strict_json(linear_payload, "$linear")
    if manifest["inputLinearSrgbProduct"] != _input_linear_identity(
        linear_manifest, linear_payload, linear_report
    ):
        fail("$.inputLinearSrgbProduct", "linear-sRGB identity mismatch")
    if manifest["inputCieXyzProduct"] != linear_manifest["inputCieXyzProduct"]:
        fail("$.inputCieXyzProduct", "CIE XYZ lineage mismatch")
    if manifest["inputSpectralProduct"] != linear_manifest["inputSpectralProduct"]:
        fail("$.inputSpectralProduct", "spectral lineage mismatch")
    if manifest["frame"] != linear_manifest["frame"]:
        fail("$.frame", "frame differs from linear-sRGB input")

    descriptor = manifest["displayTransform"]["descriptor"]
    if descriptor["implementationId"] != TRANSFORM_IMPLEMENTATION_ID:
        fail("$.displayTransform.descriptor.implementationId", "unsupported transform")
    expected_transform_fields = {
        "colourspace": "sRGB-D65",
        "derivedDisplayOutput": True,
        "exposureControl": "manual-fixed-scalar",
        "gamutPolicy": "uniform-max-channel-scale-if-needed",
        "luminanceCoefficients": [0.2126, 0.7152, 0.0722],
        "mappedLuminance": "Y/(1+Y)",
        "negativeLinearPolicy": "clip-to-zero-at-display-boundary",
        "nonNegativeLinearRgbRatioPreservedBeforeSrgbEncoding": True,
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
        "toneMapper": "reinhard-rec709-luminance-uniform-gamut/v2",
        "toneMappingDomain": "Rec.709-linear-luminance",
        "transferCurve": "IEC-61966-2-1-sRGB",
        "uniformRgbToneScale": "mappedY/Y = 1/(1+Y)",
    }
    for key, expected in expected_transform_fields.items():
        if descriptor[key] != expected:
            fail(f"$.displayTransform.descriptor.{key}", "transform field drifted")
    exposure = descriptor["exposure"]
    if isinstance(exposure, bool) or not isinstance(exposure, (int, float)):
        fail("$.displayTransform.descriptor.exposure", "invalid manual exposure")
    try:
        exposure = float(exposure)
    except (OverflowError, TypeError, ValueError) as error:
        fail(
            "$.displayTransform.descriptor.exposure",
            f"manual exposure exceeds binary64 range: {error}",
        )
    if not math.isfinite(exposure) or exposure <= 0.0:
        fail("$.displayTransform.descriptor.exposure", "invalid manual exposure")
    initial_sources = _source_descriptor()
    if descriptor["sourceFiles"] != initial_sources:
        fail("$.displayTransform.descriptor.sourceFiles", "producer source hash mismatch")
    if manifest["displayTransform"]["descriptorSha256"] != _canonical_hash(descriptor):
        fail("$.displayTransform.descriptorSha256", "transform descriptor hash mismatch")
    initial_backend = _current_numeric_backend()
    if descriptor["numericBackend"] != initial_backend:
        fail(
            "$.displayTransform.descriptor.numericBackend",
            "numeric backend differs from the authenticated current default v2",
        )

    entries = manifest["tiles"]
    linear_entries = linear_manifest["tiles"]
    if len(entries) != len(linear_entries):
        fail("$.tiles", "tile count differs from linear-sRGB input")
    allowed = {"manifest.json", manifest["integrity"]["manifestSidecar"]}
    summary = _Summary()
    frame_width = manifest["frame"]["widthPixels"]
    frame_height = manifest["frame"]["heightPixels"]
    sample_indices = tuple(manifest["frame"]["sampleIndices"])
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
    for tile_index, (linear_entry, entry) in enumerate(zip(linear_entries, entries)):
        tile_path = f"$.tiles[{tile_index}]"
        for key in ("recordCount", "recordOrder", "sampleIndex", "tile"):
            if entry[key] != linear_entry[key]:
                fail(tile_path, "tile topology differs from linear-sRGB input")
        if entry["inputLinearSrgbPayload"] != linear_entry["outputPayload"]:
            fail(f"{tile_path}.inputLinearSrgbPayload", "input artifact mismatch")
        _validate_input_uri(linear_entry["outputPayload"]["uri"])
        count = entry["recordCount"]
        width = entry["tile"]["width"]
        height = entry["tile"]["height"]
        if count != width * height:
            fail(tile_path, "record count does not match tile dimensions")
        expected_input_bytes = count * LINEAR_RECORD_BYTES
        if expected_input_bytes > MAX_INPUT_TILE_BYTES:
            fail(tile_path, "input tile exceeds the 2^26-byte limit")
        sample_index = entry["sampleIndex"]
        if sample_index not in frame_pixels:
            fail(f"{tile_path}.sampleIndex", "undeclared sample index")
        tile_x = entry["tile"]["x"]
        tile_y = entry["tile"]["y"]
        if tile_x + width > frame_width or tile_y + height > frame_height:
            fail(f"{tile_path}.tile", "tile lies outside declared frame")
        target = frame_pixels[sample_index]
        target_coverage = frame_coverage[sample_index]
        input_tile = _read_relative_file(
            linear_path.parent,
            linear_entry["outputPayload"]["uri"],
            f"{tile_path}.inputLinearSrgbPayload",
            expected_input_bytes,
        )
        if _sha256(input_tile) != linear_entry["outputPayload"]["sha256"]:
            fail(f"{tile_path}.inputLinearSrgbPayload", "input tile hash mismatch")
        for record_index in range(count):
            record_path = f"{tile_path}.records[{record_index}]"
            raw = input_tile[
                record_index * LINEAR_RECORD_BYTES :
                (record_index + 1) * LINEAR_RECORD_BYTES
            ]
            values = LINEAR_RECORD_STRUCT.unpack(raw)
            linear = tuple(values[:3])
            rgb16 = _derive_rgb16(linear, exposure, record_path)  # type: ignore[arg-type]
            local_y, local_x = divmod(record_index, width)
            target_index = (
                (tile_y + local_y) * frame_width + tile_x + local_x
            )
            if target_coverage[target_index]:
                fail(f"{tile_path}.tile", "tile overlaps another tile")
            RGB16_STRUCT.pack_into(
                target,
                target_index * RGB16_BYTES_PER_PIXEL,
                *rgb16,
            )
            target_coverage[target_index] = 1
            summary.add_record(raw, linear, rgb16)  # type: ignore[arg-type]
        del input_tile
        del raw

    images = manifest["images"]
    if len(images) != len(sample_indices):
        fail("$.images", "image count differs from frame sample count")
    for image_index, (sample_index, image_entry) in enumerate(
        zip(sample_indices, images)
    ):
        image_path = f"$.images[{image_index}]"
        if image_entry["sampleIndex"] != sample_index:
            fail(f"{image_path}.sampleIndex", "images are not in sample order")
        if (
            image_entry["widthPixels"] != frame_width
            or image_entry["heightPixels"] != frame_height
            or image_entry["pixelCount"] != frame_width * frame_height
        ):
            fail(image_path, "image topology differs from frame")
        if 0 in frame_coverage[sample_index]:
            fail(image_path, "tiles do not cover the complete frame")
        expected_ppm = _encode_ppm16_buffer(
            frame_width,
            frame_height,
            frame_pixels[sample_index],
        )
        output_artifact = image_entry["outputPayload"]
        if output_artifact["uri"] != _expected_frame_uri(sample_index):
            fail(f"{image_path}.outputPayload.uri", "non-canonical output URI")
        output_image = _read_relative_file(
            path.parent,
            output_artifact["uri"],
            f"{image_path}.outputPayload",
            len(expected_ppm),
        )
        if output_artifact["byteLength"] != len(expected_ppm):
            fail(f"{image_path}.outputPayload.byteLength", "PPM length mismatch")
        if _sha256(output_image) != output_artifact["sha256"]:
            fail(f"{image_path}.outputPayload.sha256", "PPM payload hash mismatch")
        if output_image != expected_ppm:
            mismatch = next(
                index
                for index, pair in enumerate(zip(output_image, expected_ppm))
                if pair[0] != pair[1]
            )
            fail(
                f"{image_path}.outputPayload",
                f"independent PPM replay differs at byte {mismatch}",
            )
        allowed.add(output_artifact["uri"])
        summary.add_output(expected_ppm)
        del expected_ppm, output_image

    if manifest["summary"] != summary.descriptor():
        fail("$.summary", "summary differs from independent display replay")
    frame_pixels.clear()
    frame_coverage.clear()
    del target, target_coverage
    configuration = {
        "displayTransform": descriptor,
        "encoding": manifest["encoding"],
        "frame": manifest["frame"],
        "inputCieXyzProduct": manifest["inputCieXyzProduct"],
        "inputLinearSrgbProduct": manifest["inputLinearSrgbProduct"],
        "inputSpectralProduct": manifest["inputSpectralProduct"],
        "schema": PRODUCT_SCHEMA,
    }
    configuration_hash = _canonical_hash(configuration)
    if manifest["integrity"]["configurationSha256"] != configuration_hash:
        fail("$.integrity.configurationSha256", "configuration hash mismatch")
    identity = {
        "configurationSha256": configuration_hash,
        "images": manifest["images"],
        "schema": PRODUCT_SCHEMA,
        "summary": manifest["summary"],
        "tiles": manifest["tiles"],
    }
    product_hash = _canonical_hash(identity)
    if manifest["integrity"]["productSha256"] != product_hash:
        fail("$.integrity.productSha256", "product hash mismatch")
    if manifest["id"] != f"sdr-display-quicklook-{product_hash[:24]}":
        fail("$.id", "product id does not match product hash")
    _validate_no_extra_files(path.parent, allowed)
    report = {
        "displayReplayVerified": True,
        "hdrVerified": False,
        "id": manifest["id"],
        "imageCount": len(images),
        "inputPhysicsVerified": False,
        "pixelCount": summary.records,
        "ppm16EncodingVerified": True,
        "scientificLinearSrgbModified": False,
        "status": "sdr-display-quicklook-contract-conformant",
        "tileCount": len(entries),
    }

    try:
        final_report = validate_scientific_linear_srgb_frame(
            linear_path, xyz_path, spectral_path
        )
    except ContractError as error:
        fail("$.inputLinearSrgbProduct", f"final input verification failed: {error}")
    if final_report != linear_report or _read_stable_file(
        linear_path, "$linear", 16 << 20
    ) != linear_payload:
        fail("$.inputLinearSrgbProduct", "input changed during verification")
    if _source_descriptor() != initial_sources:
        fail("$.displayTransform.descriptor.sourceFiles", "producer source changed")
    final_backend = _current_numeric_backend()
    if final_backend != initial_backend or descriptor["numericBackend"] != final_backend:
        fail(
            "$.displayTransform.descriptor.numericBackend",
            "numeric backend changed during verification",
        )

    _validate_final_self_closure(
        path.parent,
        manifest,
        manifest_payload,
        images,
        allowed,
    )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input_linear_srgb_manifest", type=Path)
    parser.add_argument("input_cie_xyz_manifest", type=Path)
    parser.add_argument("input_spectral_manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_sdr_display_quicklook(
            arguments.manifest,
            arguments.input_linear_srgb_manifest,
            arguments.input_cie_xyz_manifest,
            arguments.input_spectral_manifest,
            schema_path=arguments.schema,
        )
    except ContractError as error:
        print(f"offline SDR display validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
