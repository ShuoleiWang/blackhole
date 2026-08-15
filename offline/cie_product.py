"""Transactional CIE XYZ products derived from exact-grid spectral frames.

The converter accepts only a structurally conformant v1 scientific spectral
frame whose observer-frequency grid is bit-for-bit the authenticated 471-bin
CIE grid.  Each source tile is decoded independently and converted to a small
little-endian binary64 XYZ record.  The source record SHA-256 and its source and
convergence masks are copied into that record, preserving a direct audit link
without duplicating the 471-bin spectrum.

XYZ is the unnormalised CIE 1931 2-degree standard-observer integral.  The
error triplet is the same positive linear integral applied to the input
finite-stencil estimated absolute errors; it is not a rigorous bound.  No
exposure, tone mapping, gamut operation, sRGB transfer curve, camera model, or
absolute appearance model enters this scientific product.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import sys
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn
import uuid

from offline.cie_color import (
    CIE_ROW_COUNT,
    Cie1931Table,
    CieColorError,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
    cie_1931_frequency_grid_hz,
    load_authenticated_cie_1931_2deg,
    spectral_i_nu_to_cie_xyz,
)
from offline.job import canonical_json_bytes
from offline.spectral_frame import (
    REQUIRED_CONVERGENCE_MASK,
    SpectralFrameError,
    SpectralPixelLayout,
    unpack_spectral_pixel,
)
from offline.spectral_product import default_numeric_backend_descriptor


PRODUCT_SCHEMA: Final = "blackhole.scientific-cie-xyz-frame/v1"
PRODUCT_SCHEMA_ID: Final = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-cie-xyz-frame-v1.schema.json"
)
CONVERTER_IMPLEMENTATION_ID: Final = "blackhole.spectral-to-cie-xyz/v1"
PIXEL_LAYOUT_ID: Final = "blackhole.scientific-cie-xyz-pixel/le-f64-v1"
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
RECORD_STRUCT: Final = struct.Struct("<6d32sII")
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
    "offline/radiative_transfer.py",
    "offline/spacetime.py",
    "offline/spectral_frame.py",
    "offline/spectral_product.py",
    "schemas/offline-cie-xyz-frame-v1.schema.json",
    "schemas/offline-scientific-spectral-frame-v1.schema.json",
    "scripts/convert_offline_spectral_to_cie_xyz.py",
    "scripts/verify_nr_contract.py",
    "scripts/verify_offline_cie_xyz.py",
    "scripts/verify_offline_spectral_frame.py",
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
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
)


class CieProductError(RuntimeError):
    """A spectral-to-XYZ product failed its closed publication contract."""


def _fail(path: str, message: str) -> NoReturn:
    raise CieProductError(f"{path}: {message}")


def _finite_non_negative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


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
        canonical = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be finite canonical JSON: {error}") from error
    if not isinstance(canonical, dict):
        raise ValueError(f"{label} must be an object")
    if implementation_id and (
        not isinstance(canonical.get("implementationId"), str)
        or not canonical["implementationId"]
    ):
        raise ValueError(f"{label} requires a non-empty implementationId")
    return canonical


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                _fail(label, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(label, f"invalid UTF-8 JSON: {error}")
    if not isinstance(parsed, dict):
        _fail(label, "JSON root must be an object")
    return parsed


def _read_stable_regular(path: Path, label: str, maximum_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        if path.is_symlink():
            _fail(label, "symlinked files are forbidden")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(label, "expected a regular file")
        if before.st_size > maximum_bytes:
            _fail(label, f"file exceeds the {maximum_bytes}-byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    except CieProductError:
        raise
    except OSError as error:
        _fail(label, f"unable to read file: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        _fail(label, "file changed while it was being read")
    return payload


def _normalized_uri(value: Any, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(path, "URI must be a normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        _fail(path, "URI must be normalized, relative, and traversal-free")
    return pure


def _read_relative_regular(
    root: Path,
    uri: Any,
    label: str,
    maximum_bytes: int,
) -> bytes:
    pure = _normalized_uri(uri, label)
    if root.is_symlink():
        _fail(label, "symlinked input root is forbidden")
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
            _fail(label, "artifact must be a regular file")
        if before.st_size > maximum_bytes:
            _fail(label, f"artifact exceeds the {maximum_bytes}-byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    except CieProductError:
        raise
    except OSError as error:
        _fail(label, f"unable to read traversal-safe artifact: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        _fail(label, "artifact changed while it was being read")
    return payload


@dataclass(frozen=True, slots=True)
class CieXyzPixelRecord:
    """One XYZ triplet, propagated error, and exact source-record binding."""

    mean_cie_xyz: tuple[float, float, float]
    mean_estimated_absolute_error_xyz: tuple[float, float, float]
    input_record_sha256: bytes
    source_mask: int
    convergence_mask: int

    def __post_init__(self) -> None:
        for name in ("mean_cie_xyz", "mean_estimated_absolute_error_xyz"):
            try:
                values = tuple(getattr(self, name))
            except TypeError as error:
                raise ValueError(f"{name} must contain three numbers") from error
            if len(values) != 3:
                raise ValueError(f"{name} must contain three numbers")
            normalized = tuple(
                _finite_non_negative(value, f"{name}[{index}]")
                for index, value in enumerate(values)
            )
            object.__setattr__(self, name, normalized)
        digest = bytes(self.input_record_sha256)
        if len(digest) != 32:
            raise ValueError("input_record_sha256 must contain 32 bytes")
        object.__setattr__(self, "input_record_sha256", digest)
        for name in ("source_mask", "convergence_mask"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 0xFFFFFFFF:
                raise ValueError(f"{name} must fit uint32")
        if self.convergence_mask & REQUIRED_CONVERGENCE_MASK != (
            REQUIRED_CONVERGENCE_MASK
        ):
            raise ValueError("input convergence mask lacks a required gate")


def cie_xyz_pixel_layout_descriptor() -> dict[str, Any]:
    return {
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


def pack_cie_xyz_pixel(record: CieXyzPixelRecord) -> bytes:
    if not isinstance(record, CieXyzPixelRecord):
        raise TypeError("record must be CieXyzPixelRecord")
    payload = RECORD_STRUCT.pack(
        *record.mean_cie_xyz,
        *record.mean_estimated_absolute_error_xyz,
        record.input_record_sha256,
        record.source_mask,
        record.convergence_mask,
    )
    if len(payload) != RECORD_BYTES:
        raise AssertionError("packed CIE XYZ record has the wrong length")
    return payload


def unpack_cie_xyz_pixel(
    payload: bytes | bytearray | memoryview,
) -> CieXyzPixelRecord:
    raw = bytes(payload)
    if len(raw) != RECORD_BYTES:
        raise CieProductError("CIE XYZ record has the wrong byte length")
    values = RECORD_STRUCT.unpack(raw)
    try:
        return CieXyzPixelRecord(
            mean_cie_xyz=tuple(values[:3]),  # type: ignore[arg-type]
            mean_estimated_absolute_error_xyz=(
                tuple(values[3:6])  # type: ignore[arg-type]
            ),
            input_record_sha256=values[6],
            source_mask=values[7],
            convergence_mask=values[8],
        )
    except (TypeError, ValueError) as error:
        raise CieProductError(f"CIE XYZ record is invalid: {error}") from error


def _source_file_descriptor(path: Path, module_uri: str) -> dict[str, Any]:
    payload = _read_stable_regular(path, f"converter source {module_uri}", 4 << 20)
    return {
        "byteLength": len(payload),
        "moduleUri": module_uri,
        "sha256": _sha256(payload),
    }


def converter_descriptor(numeric_backend: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the numerical algorithm, source bytes, and declared backend."""

    backend = _canonical_object(
        numeric_backend,
        "numeric backend",
        implementation_id=True,
    )
    source_root = Path(__file__).resolve(strict=True).parents[1]
    return {
        "algorithm": (
            "I_nu-to-I_lambda Jacobian and 1-nm trapezoidal integration; "
            "same positive weights applied to estimated absolute errors"
        ),
        "implementationId": CONVERTER_IMPLEMENTATION_ID,
        "numericBackend": backend,
        "sourceFiles": [
            _source_file_descriptor(source_root / module_uri, module_uri)
            for module_uri in CONVERTER_SOURCE_FILES
        ],
    }


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
        self.maximum_mean_xyz = [0.0, 0.0, 0.0]
        self.maximum_mean_errors = [0.0, 0.0, 0.0]
        self.source_mask_union = 0
        self.convergence_mask_intersection: int | None = None
        self.record_hash_chain = hashlib.sha256()

    def add(
        self,
        output: CieXyzPixelRecord,
        pixel_solid_angle_sr: float,
    ) -> None:
        self.record_count += 1
        for index in range(3):
            self.integrated_xyz[index].add(
                output.mean_cie_xyz[index] * pixel_solid_angle_sr
            )
            self.integrated_errors[index].add(
                output.mean_estimated_absolute_error_xyz[index]
                * pixel_solid_angle_sr
            )
            self.maximum_mean_xyz[index] = max(
                self.maximum_mean_xyz[index],
                output.mean_cie_xyz[index],
            )
            self.maximum_mean_errors[index] = max(
                self.maximum_mean_errors[index],
                output.mean_estimated_absolute_error_xyz[index],
            )
        self.source_mask_union |= output.source_mask
        self.convergence_mask_intersection = (
            output.convergence_mask
            if self.convergence_mask_intersection is None
            else self.convergence_mask_intersection & output.convergence_mask
        )
        self.record_hash_chain.update(output.input_record_sha256)

    def descriptor(self) -> dict[str, Any]:
        return {
            "convergenceMaskIntersection": self.convergence_mask_intersection,
            "estimatedAbsoluteErrorCieXyzOverFrame": [
                value.value() for value in self.integrated_errors
            ],
            "inputRecordSha256Chain": self.record_hash_chain.hexdigest(),
            "integratedCieXyzOverFrame": [
                value.value() for value in self.integrated_xyz
            ],
            "maximumMeanCieXyz": self.maximum_mean_xyz,
            "maximumMeanEstimatedAbsoluteErrorXyz": self.maximum_mean_errors,
            "recordCount": self.record_count,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_mask_union,
        }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_no_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CieProductError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _promote_directory_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise CieProductError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        _fsync_directory(destination.parent)
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise CieProductError(f"refusing to overwrite existing output {destination}")
    raise CieProductError(
        f"unable to atomically publish {destination}: {os.strerror(error_number)}"
    )


def _output_tile_uri(input_uri: str) -> str:
    pure = _normalized_uri(input_uri, "$.tiles[].inputPayload.uri")
    if pure.parent.as_posix() != "tiles" or pure.suffix != ".spx":
        _fail("$.tiles[].inputPayload.uri", "unsupported spectral tile URI")
    return f"tiles/{pure.stem}.cxyz"


def _strict_input_verification(manifest_path: Path) -> dict[str, Any]:
    try:
        from scripts.verify_offline_spectral_frame import (
            validate_scientific_spectral_frame,
        )
        from scripts.verify_nr_contract import ContractError

        return validate_scientific_spectral_frame(manifest_path)
    except ContractError as error:
        raise CieProductError(
            f"input spectral product failed strict structural verification: {error}"
        ) from error


def _input_identity(
    manifest: Mapping[str, Any],
    manifest_payload: bytes,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": manifest["id"],
        "manifestSha256": _sha256(manifest_payload),
        "physicsVerified": verification["physicsVerified"],
        "productSha256": manifest["integrity"]["productSha256"],
        "provenanceScope": verification["provenanceScope"],
        "recordCount": verification["recordCount"],
        "schema": manifest["schema"],
        "structuralStatus": verification["status"],
        "tileCount": verification["tileCount"],
    }


@dataclass(frozen=True, slots=True)
class CieProductPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    product_id: str
    product_sha256: str
    tile_count: int
    record_count: int


def convert_spectral_product_to_cie_xyz(
    input_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    cie_csv_path: Path | str = DEFAULT_CIE_CSV,
    cie_metadata_path: Path | str = DEFAULT_CIE_METADATA,
    numeric_backend: Mapping[str, Any] | None = None,
) -> CieProductPublication:
    """Publish one non-overwriting v1 XYZ product from a strict spectral input."""

    input_manifest_path = Path(input_manifest_path).absolute()
    if input_manifest_path.name != MANIFEST_NAME:
        raise CieProductError("input spectral manifest must be named manifest.json")
    verification = _strict_input_verification(input_manifest_path)
    input_manifest_payload = _read_stable_regular(
        input_manifest_path,
        "input spectral manifest",
        64 << 20,
    )
    input_manifest = _strict_json(
        input_manifest_payload,
        "input spectral manifest",
    )
    if canonical_json_bytes(input_manifest) != input_manifest_payload:
        raise CieProductError("input spectral manifest is not canonical JSON")
    input_root = input_manifest_path.parent

    try:
        table = load_authenticated_cie_1931_2deg(
            cie_csv_path,
            cie_metadata_path,
        )
    except (CieColorError, OSError, TypeError, ValueError) as error:
        raise CieProductError(f"CIE resource authentication failed: {error}") from error
    if not isinstance(table, Cie1931Table):
        raise AssertionError("authenticated CIE loader returned an invalid table")
    expected_frequencies = cie_1931_frequency_grid_hz(table)
    frequencies = tuple(input_manifest["observerFrequencyBinsHz"])
    if len(frequencies) != CIE_ROW_COUNT or frequencies != expected_frequencies:
        raise CieProductError(
            "input spectral frame must use the exact authenticated 471-bin CIE grid"
        )
    layout = SpectralPixelLayout(frequencies)
    if dict(layout.descriptor()) != input_manifest["pixelLayout"]:
        raise CieProductError("input spectral pixel layout is inconsistent")

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
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CieProductError(
            f"numeric backend authentication failed: {error}"
        ) from error
    converter = converter_descriptor(backend)
    cie_descriptor = table.descriptor()
    input_identity = _input_identity(
        input_manifest,
        input_manifest_payload,
        verification,
    )
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise CieProductError(f"refusing to overwrite existing output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    summary = _Summary()
    output_tiles: list[dict[str, Any]] = []
    try:
        staging.mkdir(mode=0o700)
        (staging / "tiles").mkdir(mode=0o700)
        for tile_index, input_entry in enumerate(input_manifest["tiles"]):
            input_artifact = input_entry["payload"]
            input_uri = input_artifact["uri"]
            expected_bytes = input_entry["recordCount"] * layout.record_bytes
            input_payload = _read_relative_regular(
                input_root,
                input_uri,
                f"input spectral tile {tile_index}",
                expected_bytes,
            )
            if (
                len(input_payload) != expected_bytes
                or input_artifact["byteLength"] != expected_bytes
                or _sha256(input_payload) != input_artifact["sha256"]
            ):
                raise CieProductError(
                    f"input spectral tile {tile_index} changed after verification"
                )
            output_payload = bytearray()
            for record_index in range(input_entry["recordCount"]):
                offset = record_index * layout.record_bytes
                input_record_payload = input_payload[
                    offset : offset + layout.record_bytes
                ]
                try:
                    input_record = unpack_spectral_pixel(
                        layout,
                        input_record_payload,
                    )
                    xyz = spectral_i_nu_to_cie_xyz(
                        frequencies,
                        input_record.mean_specific_intensities_nu,
                        table=table,
                    )
                    error_xyz = spectral_i_nu_to_cie_xyz(
                        frequencies,
                        input_record.mean_estimated_absolute_errors_nu,
                        table=table,
                    )
                except (
                    CieColorError,
                    SpectralFrameError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise CieProductError(
                        f"input tile {tile_index} record {record_index} cannot "
                        f"produce CIE XYZ: {error}"
                    ) from error
                output_record = CieXyzPixelRecord(
                    mean_cie_xyz=(xyz.x, xyz.y, xyz.z),
                    mean_estimated_absolute_error_xyz=(
                        error_xyz.x,
                        error_xyz.y,
                        error_xyz.z,
                    ),
                    input_record_sha256=hashlib.sha256(
                        input_record_payload
                    ).digest(),
                    source_mask=input_record.source_mask,
                    convergence_mask=input_record.convergence_mask,
                )
                output_payload.extend(pack_cie_xyz_pixel(output_record))
                summary.add(output_record, input_record.pixel_solid_angle_sr)

            output_uri = _output_tile_uri(input_uri)
            output_bytes = bytes(output_payload)
            _atomic_write_no_replace(staging / output_uri, output_bytes)
            output_tiles.append(
                {
                    "inputPayload": {
                        "byteLength": input_artifact["byteLength"],
                        "sha256": input_artifact["sha256"],
                        "uri": input_uri,
                    },
                    "outputPayload": {
                        "byteLength": len(output_bytes),
                        "sha256": _sha256(output_bytes),
                        "uri": output_uri,
                    },
                    "recordCount": input_entry["recordCount"],
                    "recordOrder": input_entry["recordOrder"],
                    "sampleIndex": input_entry["sampleIndex"],
                    "tile": dict(input_entry["tile"]),
                }
            )

        if summary.record_count != verification["recordCount"]:
            raise CieProductError("converted tiles do not cover every input record")
        second_verification = _strict_input_verification(input_manifest_path)
        second_manifest_payload = _read_stable_regular(
            input_manifest_path,
            "input spectral manifest final snapshot",
            64 << 20,
        )
        if (
            second_verification != verification
            or second_manifest_payload != input_manifest_payload
        ):
            raise CieProductError("input spectral product changed during conversion")
        if converter_descriptor(backend) != converter:
            raise CieProductError("converter source changed during conversion")

        output_layout = cie_xyz_pixel_layout_descriptor()
        configuration = {
            "cieDataset": cie_descriptor,
            "converter": converter,
            "frame": input_manifest["frame"],
            "inputSpectralProduct": input_identity,
            "pixelLayout": output_layout,
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
        product_id = f"scientific-cie-xyz-frame-{product_hash[:24]}"
        manifest = {
            "cieDataset": cie_descriptor,
            "converter": {
                "descriptor": converter,
                "descriptorSha256": _canonical_hash(converter),
            },
            "frame": input_manifest["frame"],
            "id": product_id,
            "inputSpectralProduct": input_identity,
            "integrity": {
                "configurationSha256": configuration_hash,
                "manifestSidecar": SIDECAR_NAME,
                "productSha256": product_hash,
            },
            "pixelLayout": output_layout,
            "schema": PRODUCT_SCHEMA,
            "scientificStatus": dict(SCIENTIFIC_STATUS),
            "summary": summary_descriptor,
            "tiles": output_tiles,
        }
        manifest_payload = canonical_json_bytes(manifest)
        manifest_hash = _sha256(manifest_payload)
        sidecar = f"{manifest_hash}  {MANIFEST_NAME}\n".encode("ascii")
        _atomic_write_no_replace(staging / SIDECAR_NAME, sidecar)
        _atomic_write_no_replace(staging / MANIFEST_NAME, manifest_payload)
        _fsync_directory(staging)
        _promote_directory_no_replace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            _fsync_directory(output.parent)
        raise

    return CieProductPublication(
        output_directory=output,
        manifest_path=output / MANIFEST_NAME,
        manifest_sha256=manifest_hash,
        product_id=product_id,
        product_sha256=product_hash,
        tile_count=len(output_tiles),
        record_count=summary.record_count,
    )


__all__ = (
    "CONVERTER_IMPLEMENTATION_ID",
    "CONVERTER_SOURCE_FILES",
    "CieProductError",
    "CieProductPublication",
    "CieXyzPixelRecord",
    "MANIFEST_NAME",
    "PIXEL_LAYOUT_ID",
    "PRODUCT_SCHEMA",
    "PRODUCT_SCHEMA_ID",
    "RECORD_BYTES",
    "SCIENTIFIC_STATUS",
    "SIDECAR_NAME",
    "cie_xyz_pixel_layout_descriptor",
    "convert_spectral_product_to_cie_xyz",
    "converter_descriptor",
    "pack_cie_xyz_pixel",
    "unpack_cie_xyz_pixel",
)
