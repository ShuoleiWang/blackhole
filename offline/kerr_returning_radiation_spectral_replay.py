"""Live-authority replay for returning-radiation spectral products.

This verifier first delegates the public artifact contract to the generic
spectral-frame verifier.  It then binds the manifest to the current returning
tile producer, the current source/runtime closure, and one caller-supplied
*live* ``KerrReturningRadiationFiniteThicknessRaySampler``.  Every pixel is
rerun through the frozen production whole-pixel integrator and packer and is
required to reproduce the published bytes exactly.

The result is deliberately a same-code replay.  It replays frame geodesics
against the already authenticated frozen thermal snapshot; it does not rerun
the thermal fixed point, direction-cache records, or direction rays, and it is
not an independent physics oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Final, Mapping, NoReturn, TypedDict

from offline.adaptive_frame import AdaptivePixelOptions
from offline.cie_color import DEFAULT_CIE_CSV, DEFAULT_CIE_METADATA
from offline.job import InputArtifact, JobSpec, canonical_json_bytes
import offline.kerr_returning_radiation_finite_thickness_frame as frame_module
from offline.kerr_returning_radiation_finite_thickness_frame import (
    IMPLEMENTATION_ID as RETURNING_SAMPLER_IMPLEMENTATION_ID,
    KerrReturningRadiationFiniteThicknessRaySampler,
    integrate_returning_thermal_spectral_pixel,
)
from offline.kerr_returning_radiation_frame_context import (
    MAXIMUM_AUTHENTICATED_ANNULUS_COUNT,
    ReturningThermalEmissionSnapshotV1,
    ValidatedReturningThermalAuthority,
)
import offline.kerr_returning_radiation_spectral_product as product_module
from offline.kerr_returning_radiation_spectral_product import (
    RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION,
    RETURNING_ADAPTIVE_TILE_PRODUCER_ID,
    assert_returning_radiation_spectral_runtime_bindings,
    build_returning_radiation_spectral_job_spec,
)
import offline.spectral_frame as spectral_frame_module
from offline.spectral_frame import (
    SpectralPixelLayout,
    pack_adaptive_pixel,
    pack_spectral_pixel,
)
import offline.spectral_product as spectral_product_module
from offline.spectral_product import (
    PRODUCT_SCHEMA,
    SpectralFrameGrid,
    default_numeric_backend_descriptor,
)
import scripts.render_offline_kerr_returning_radiation_frame as renderer_module
from scripts.render_offline_kerr_returning_radiation_frame import (
    CIE_CSV_INPUT_URI,
    CIE_METADATA_INPUT_URI,
    PRODUCER_SOURCE_FILES,
)
import scripts.verify_offline_spectral_frame as verifier_module
from scripts.verify_offline_spectral_frame import (
    DEFAULT_SCHEMA,
    validate_scientific_spectral_frame,
)


ROOT: Final = Path(__file__).resolve().parents[1]
_READ_CHUNK_BYTES: Final = 1024 * 1024

MAXIMUM_REPLAY_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_REPLAY_SCHEMA_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_REPLAY_SIDECAR_BYTES: Final = 4096
MAXIMUM_REPLAY_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_REPLAY_SOURCE_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_REPLAY_TILE_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_REPLAY_TOTAL_TILE_BYTES: Final = 1024 * 1024 * 1024
MAXIMUM_REPLAY_TILES: Final = 65_536
MAXIMUM_REPLAY_RECORDS: Final = 16_777_216
MAXIMUM_REPLAY_FREQUENCY_BINS: Final = 4096
MAXIMUM_REPLAY_ANNULI: Final = MAXIMUM_AUTHENTICATED_ANNULUS_COUNT
MAXIMUM_REPLAY_TOTAL_RAY_EVALUATIONS: Final = 10_000_000
MAXIMUM_REPLAY_ADAPTIVE_DEPTH: Final = 16


class KerrReturningRadiationSpectralReplayError(RuntimeError):
    """A fail-closed live returning-spectral replay failure."""


class ReturningRadiationSpectralLiveReplayReport(TypedDict):
    """Typed evidence returned after a complete byte-exact live replay."""

    id: str
    status: str
    structuralContractVerified: bool
    producerIdentityCurrentMatch: bool
    jobSpecVerified: bool
    samplerDescriptorLiveMatch: bool
    numericBackendCurrentMatch: bool
    sourceArtifactsCurrentMatch: bool
    frameGeodesicsReplayed: bool
    frozenThermalSnapshotBound: bool
    thermalFixedPointReplayed: bool
    directionCacheRecordsReplayed: bool
    directionRaysRetraced: bool
    independentPhysicsOracle: bool
    pixelBytesExact: bool
    recordCount: int
    tileCount: int
    totalRaySamples: int
    totalFrameGeodesicsReplayed: int
    replayScope: str
    scientificScope: str
    sourceHashScope: str


@dataclass(frozen=True, slots=True)
class ReturningRadiationSpectralReplayLimits:
    """Caller-selectable limits that may only tighten fixed hard maxima."""

    maximum_manifest_bytes: int = MAXIMUM_REPLAY_MANIFEST_BYTES
    maximum_schema_bytes: int = MAXIMUM_REPLAY_SCHEMA_BYTES
    maximum_source_file_bytes: int = MAXIMUM_REPLAY_SOURCE_FILE_BYTES
    maximum_source_total_bytes: int = MAXIMUM_REPLAY_SOURCE_TOTAL_BYTES
    maximum_tile_bytes: int = MAXIMUM_REPLAY_TILE_BYTES
    maximum_total_tile_bytes: int = MAXIMUM_REPLAY_TOTAL_TILE_BYTES
    maximum_tiles: int = MAXIMUM_REPLAY_TILES
    maximum_records: int = MAXIMUM_REPLAY_RECORDS
    maximum_frequency_bins: int = MAXIMUM_REPLAY_FREQUENCY_BINS
    maximum_annuli: int = MAXIMUM_REPLAY_ANNULI
    maximum_total_ray_evaluations: int = MAXIMUM_REPLAY_TOTAL_RAY_EVALUATIONS
    maximum_adaptive_depth: int = MAXIMUM_REPLAY_ADAPTIVE_DEPTH

    def __post_init__(self) -> None:
        hard = {
            "maximum_manifest_bytes": MAXIMUM_REPLAY_MANIFEST_BYTES,
            "maximum_schema_bytes": MAXIMUM_REPLAY_SCHEMA_BYTES,
            "maximum_source_file_bytes": MAXIMUM_REPLAY_SOURCE_FILE_BYTES,
            "maximum_source_total_bytes": MAXIMUM_REPLAY_SOURCE_TOTAL_BYTES,
            "maximum_tile_bytes": MAXIMUM_REPLAY_TILE_BYTES,
            "maximum_total_tile_bytes": MAXIMUM_REPLAY_TOTAL_TILE_BYTES,
            "maximum_tiles": MAXIMUM_REPLAY_TILES,
            "maximum_records": MAXIMUM_REPLAY_RECORDS,
            "maximum_frequency_bins": MAXIMUM_REPLAY_FREQUENCY_BINS,
            "maximum_annuli": MAXIMUM_REPLAY_ANNULI,
            "maximum_total_ray_evaluations": (
                MAXIMUM_REPLAY_TOTAL_RAY_EVALUATIONS
            ),
        }
        for name, ceiling in hard.items():
            value = object.__getattribute__(self, name)
            if type(value) is not int or value < 1 or value > ceiling:
                raise ValueError(
                    f"{name} must be an exact positive integer no greater than "
                    f"{ceiling}"
                )
        depth = object.__getattribute__(self, "maximum_adaptive_depth")
        if type(depth) is not int or depth < 0 or depth > MAXIMUM_REPLAY_ADAPTIVE_DEPTH:
            raise ValueError(
                "maximum_adaptive_depth must be an exact non-negative integer "
                f"no greater than {MAXIMUM_REPLAY_ADAPTIVE_DEPTH}"
            )


DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS: Final = (
    ReturningRadiationSpectralReplayLimits()
)


@dataclass(frozen=True, slots=True)
class _ProductSnapshot:
    manifest_payload: bytes
    schema_payload: bytes
    sidecar_uri: str
    sidecar_payload: bytes
    tile_payloads: tuple[tuple[str, bytes], ...]


# Freeze every public computation boundary used by replay.  The public and
# owning-module bindings are checked before an alternative implementation can
# run and again before success is reported.
_PIXEL_INTEGRATOR_ENTRY: Final = integrate_returning_thermal_spectral_pixel
_ADAPTIVE_PIXEL_INTEGRATOR_ENTRY: Final = frame_module.integrate_spectral_pixel
_TRUSTED_PIXEL_SAMPLER_TYPE_ENTRY: Final = frame_module._TrustedPixelSampler
_TRUSTED_PIXEL_SAMPLE_ENTRY: Final = frame_module._TrustedPixelSampler.sample
_TRUSTED_RAY_SAMPLE_ENTRY: Final = (
    KerrReturningRadiationFiniteThicknessRaySampler._sample_trusted
)
_TRUSTED_RAY_SAMPLE_FROZEN_ENTRY: Final = frame_module._SAMPLER_TRUSTED_ENTRY
_PACK_PIXEL_ENTRY: Final = pack_adaptive_pixel
_PACK_SPECTRAL_PIXEL_ENTRY: Final = pack_spectral_pixel
_SOURCE_COVERAGE_ENTRY: Final = spectral_frame_module._source_coverage
_PIXEL_BOUNDS_ENTRY: Final = SpectralFrameGrid.pixel_bounds
_SAMPLER_DESCRIPTOR_ENTRY: Final = (
    KerrReturningRadiationFiniteThicknessRaySampler.descriptor
)
_AUTHORITY_REQUIRE_LIVE_ENTRY: Final = ValidatedReturningThermalAuthority.require_live
_DEFAULT_BACKEND_ENTRY: Final = default_numeric_backend_descriptor
_PRODUCT_RUNTIME_GATE_ENTRY: Final = (
    assert_returning_radiation_spectral_runtime_bindings
)
_RETURNING_JOB_SPEC_BUILDER_ENTRY: Final = (
    build_returning_radiation_spectral_job_spec
)
_STRUCTURAL_VERIFIER_ENTRY: Final = validate_scientific_spectral_frame
_LAYOUT_ENTRY: Final = verifier_module._layout
_OPTIONS_ENTRY: Final = verifier_module._adaptive_options
_JOB_SPEC_ENTRY: Final = verifier_module._job_spec
_PRODUCER_SOURCE_FILES_REFERENCE: Final = PRODUCER_SOURCE_FILES
_PRODUCER_SOURCE_FILES: Final = tuple(PRODUCER_SOURCE_FILES)


def _fail(path: str, message: str) -> NoReturn:
    raise KerrReturningRadiationSpectralReplayError(f"{path}: {message}")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    values = tuple(getattr(os, name, None) for name in required)
    if any(type(value) is not int for value in values):
        _fail("$runtime", "secure directory open primitives are unavailable")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )


def _file_flags() -> int:
    required = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    values = tuple(getattr(os, name, None) for name in required)
    if any(type(value) is not int for value in values):
        _fail("$runtime", "secure regular-file open primitives are unavailable")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _open_absolute_directory(path: Path, label: str) -> int:
    """Open an absolute directory one non-symlink component at a time."""

    candidate = Path(path)
    if not candidate.is_absolute():
        _fail(label, "path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, _directory_flags())
        for part in candidate.parts[1:]:
            if part in ("", ".", ".."):
                _fail(label, "path is not lexically normalized")
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(label, "expected a directory")
        result = descriptor
        descriptor = None
        return result
    except OSError as error:
        _fail(label, f"cannot open non-symlink path: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _normalized_relative(value: Any, label: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        _fail(label, "artifact URI must be an exact non-empty POSIX string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(label, "artifact URI must be normalized and traversal-free")
    return pure


def _read_bounded_fd(descriptor: int, maximum_bytes: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _fail(label, "artifact must be a regular file")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        _fail(label, f"artifact exceeds the fixed {maximum_bytes}-byte limit")
    chunks: list[bytes] = []
    length = 0
    while True:
        remaining = maximum_bytes - length
        block = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining + 1))
        if not block:
            break
        length += len(block)
        if length > maximum_bytes:
            _fail(label, "artifact grew beyond its fixed byte limit")
        chunks.append(block)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after) or length != before.st_size:
        _fail(label, "artifact changed while it was being read")
    return b"".join(chunks)


def _read_at(
    root_descriptor: int,
    relative: PurePosixPath | str,
    maximum_bytes: int,
    label: str,
) -> bytes:
    pure = (
        relative
        if type(relative) is PurePosixPath
        else _normalized_relative(relative, label)
    )
    directories: list[int] = []
    file_descriptor: int | None = None
    try:
        cursor = root_descriptor
        for part in pure.parts[:-1]:
            directory = os.open(part, _directory_flags(), dir_fd=cursor)
            directories.append(directory)
            cursor = directory
        file_descriptor = os.open(pure.parts[-1], _file_flags(), dir_fd=cursor)
        return _read_bounded_fd(file_descriptor, maximum_bytes, label)
    except OSError as error:
        _fail(label, f"cannot open traversal-safe regular artifact: {error}")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)


def _regular_size_at(root_descriptor: int, relative: str, label: str) -> int:
    pure = _normalized_relative(relative, label)
    directories: list[int] = []
    descriptor: int | None = None
    try:
        cursor = root_descriptor
        for part in pure.parts[:-1]:
            directory = os.open(part, _directory_flags(), dir_fd=cursor)
            directories.append(directory)
            cursor = directory
        descriptor = os.open(pure.parts[-1], _file_flags(), dir_fd=cursor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "artifact must be a regular file")
        return metadata.st_size
    except OSError as error:
        _fail(label, f"cannot stat traversal-safe regular artifact: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def _read_absolute_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    candidate = Path(path).absolute()
    parent_descriptor = _open_absolute_directory(candidate.parent, label)
    try:
        return _read_at(
            parent_descriptor,
            PurePosixPath(candidate.name),
            maximum_bytes,
            label,
        )
    finally:
        os.close(parent_descriptor)


def _strict_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                _fail(label, f"duplicate or foreign JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> NoReturn:
        _fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"invalid UTF-8 JSON: {error}")
    if type(value) is not dict:
        _fail(label, "JSON root must be an exact object")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        _fail(label, f"JSON is not finite canonical data: {error}")
    if canonical != payload:
        _fail(label, "JSON is not in canonical encoding")
    return value


def _assert_structural_verifier_binding() -> None:
    """Reject replacement of the generic authority before calling it."""

    if (
        verifier_module.validate_scientific_spectral_frame
        is not _STRUCTURAL_VERIFIER_ENTRY
        or validate_scientific_spectral_frame is not _STRUCTURAL_VERIFIER_ENTRY
    ):
        _fail("$runtime", "generic structural verifier identity changed")


def _assert_runtime_bindings() -> None:
    if (
        frame_module.integrate_returning_thermal_spectral_pixel
        is not _PIXEL_INTEGRATOR_ENTRY
        or integrate_returning_thermal_spectral_pixel is not _PIXEL_INTEGRATOR_ENTRY
        or frame_module.integrate_spectral_pixel
        is not _ADAPTIVE_PIXEL_INTEGRATOR_ENTRY
        or frame_module._TrustedPixelSampler
        is not _TRUSTED_PIXEL_SAMPLER_TYPE_ENTRY
        or frame_module._TrustedPixelSampler.sample
        is not _TRUSTED_PIXEL_SAMPLE_ENTRY
        or KerrReturningRadiationFiniteThicknessRaySampler._sample_trusted
        is not _TRUSTED_RAY_SAMPLE_ENTRY
        or frame_module._SAMPLER_TRUSTED_ENTRY
        is not _TRUSTED_RAY_SAMPLE_FROZEN_ENTRY
        or spectral_frame_module.pack_adaptive_pixel is not _PACK_PIXEL_ENTRY
        or pack_adaptive_pixel is not _PACK_PIXEL_ENTRY
        or spectral_frame_module.pack_spectral_pixel
        is not _PACK_SPECTRAL_PIXEL_ENTRY
        or pack_spectral_pixel is not _PACK_SPECTRAL_PIXEL_ENTRY
        or spectral_frame_module._source_coverage is not _SOURCE_COVERAGE_ENTRY
        or SpectralFrameGrid.pixel_bounds is not _PIXEL_BOUNDS_ENTRY
        or KerrReturningRadiationFiniteThicknessRaySampler.descriptor
        is not _SAMPLER_DESCRIPTOR_ENTRY
        or ValidatedReturningThermalAuthority.require_live
        is not _AUTHORITY_REQUIRE_LIVE_ENTRY
    ):
        _fail("$runtime", "live replay computation callable identity changed")
    if (
        spectral_product_module.default_numeric_backend_descriptor
        is not _DEFAULT_BACKEND_ENTRY
        or default_numeric_backend_descriptor is not _DEFAULT_BACKEND_ENTRY
        or product_module.assert_returning_radiation_spectral_runtime_bindings
        is not _PRODUCT_RUNTIME_GATE_ENTRY
        or product_module.build_returning_radiation_spectral_job_spec
        is not _RETURNING_JOB_SPEC_BUILDER_ENTRY
        or build_returning_radiation_spectral_job_spec
        is not _RETURNING_JOB_SPEC_BUILDER_ENTRY
        or verifier_module._layout is not _LAYOUT_ENTRY
        or verifier_module._adaptive_options is not _OPTIONS_ENTRY
        or verifier_module._job_spec is not _JOB_SPEC_ENTRY
        or renderer_module.PRODUCER_SOURCE_FILES
        is not _PRODUCER_SOURCE_FILES_REFERENCE
    ):
        _fail("$runtime", "live replay verification callable identity changed")
    _assert_structural_verifier_binding()
    try:
        _PRODUCT_RUNTIME_GATE_ENTRY()
    except Exception as error:
        _fail("$runtime", f"returning producer runtime binding failed: {error}")


def _preflight_manifest(
    root_descriptor: int,
    payload: bytes,
    limits: ReturningRadiationSpectralReplayLimits,
) -> tuple[dict[str, Any], tuple[str, ...], int, int]:
    manifest = _strict_canonical_json(payload, "$")
    try:
        tiles = manifest["tiles"]
        frequencies = manifest["observerFrequencyBinsHz"]
        options = manifest["adaptivePixelOptions"]
    except (KeyError, TypeError) as error:
        _fail("$", f"manifest lacks a replay resource field: {error}")
    if type(tiles) is not list or not tiles or len(tiles) > limits.maximum_tiles:
        _fail("$.tiles", "tile count exceeds the fixed replay limit")
    if (
        type(frequencies) is not list
        or not frequencies
        or len(frequencies) > limits.maximum_frequency_bins
    ):
        _fail(
            "$.observerFrequencyBinsHz",
            "frequency-bin count exceeds the fixed replay limit",
        )
    record_count = 0
    total_bytes = 0
    uris: list[str] = []
    for index, entry in enumerate(tiles):
        label = f"$.tiles[{index}]"
        try:
            records = entry["recordCount"]
            artifact = entry["payload"]
            uri = artifact["uri"]
            declared_bytes = artifact["byteLength"]
        except (KeyError, TypeError) as error:
            _fail(label, f"malformed tile resource declaration: {error}")
        if type(records) is not int or records < 1:
            _fail(f"{label}.recordCount", "must be an exact positive integer")
        if type(declared_bytes) is not int or declared_bytes < 1:
            _fail(f"{label}.payload.byteLength", "must be exactly positive")
        pure = _normalized_relative(uri, f"{label}.payload.uri")
        if len(pure.parts) != 2 or pure.parts[0] != "tiles":
            _fail(f"{label}.payload.uri", "tile must be a direct child of tiles/")
        size = _regular_size_at(root_descriptor, uri, f"{label}.payload.uri")
        if size != declared_bytes:
            _fail(f"{label}.payload.byteLength", "declared and actual bytes differ")
        if size > limits.maximum_tile_bytes:
            _fail(f"{label}.payload.byteLength", "tile exceeds its fixed byte cap")
        record_count += records
        total_bytes += size
        uris.append(uri)
    if len(set(uris)) != len(uris):
        _fail("$.tiles", "tile URIs must be unique")
    if record_count > limits.maximum_records:
        _fail("$.tiles", "record count exceeds the fixed replay limit")
    if total_bytes > limits.maximum_total_tile_bytes:
        _fail("$.tiles", "total tile bytes exceed the fixed replay limit")
    try:
        maximum_rays = options["maximumRayEvaluations"]
        maximum_depth = options["maximumDepth"]
    except (KeyError, TypeError) as error:
        _fail("$.adaptivePixelOptions", f"malformed replay budget: {error}")
    if type(maximum_rays) is not int or maximum_rays < 1:
        _fail(
            "$.adaptivePixelOptions.maximumRayEvaluations",
            "must be an exact positive integer",
        )
    if type(maximum_depth) is not int or maximum_depth < 0:
        _fail(
            "$.adaptivePixelOptions.maximumDepth",
            "must be an exact non-negative integer",
        )
    if maximum_depth > limits.maximum_adaptive_depth:
        _fail("$.adaptivePixelOptions.maximumDepth", "adaptive depth exceeds cap")
    # Each returning sample owns independent fine and coarse frame geodesics.
    if record_count * maximum_rays * 2 > limits.maximum_total_ray_evaluations:
        _fail(
            "$.adaptivePixelOptions.maximumRayEvaluations",
            "fine-plus-coarse frame-geodesic budget exceeds the fixed limit",
        )
    return manifest, tuple(uris), record_count, total_bytes


def _capture_product_snapshot(
    root_descriptor: int,
    manifest: Mapping[str, Any],
    manifest_payload: bytes,
    schema_payload: bytes,
    tile_uris: tuple[str, ...],
    limits: ReturningRadiationSpectralReplayLimits,
) -> _ProductSnapshot:
    """Capture the exact bounded tree addressed by the held product dirfd."""

    try:
        sidecar_uri = manifest["integrity"]["manifestSidecar"]
    except (KeyError, TypeError) as error:
        _fail("$.integrity.manifestSidecar", f"malformed sidecar URI: {error}")
    sidecar_pure = _normalized_relative(
        sidecar_uri,
        "$.integrity.manifestSidecar",
    )
    if len(sidecar_pure.parts) != 1:
        _fail("$.integrity.manifestSidecar", "sidecar must be at product root")
    expected_root = {"manifest.json", sidecar_uri, "tiles"}
    expected_tile_names = {PurePosixPath(uri).name for uri in tile_uris}
    root_before = os.fstat(root_descriptor)
    tiles_descriptor: int | None = None
    try:
        observed_root = set(os.listdir(root_descriptor))
        if observed_root != expected_root:
            _fail("$files", "product root differs from the closed declared tree")
        for name in ("manifest.json", sidecar_uri):
            metadata = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                _fail("$files", f"root artifact {name!r} is not a regular file")
        tiles_metadata = os.stat(
            "tiles",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(tiles_metadata.st_mode):
            _fail("$files", "tiles is not a non-symlink directory")
        tiles_descriptor = os.open(
            "tiles",
            _directory_flags(),
            dir_fd=root_descriptor,
        )
        tiles_before = os.fstat(tiles_descriptor)
        observed_tiles = set(os.listdir(tiles_descriptor))
        if observed_tiles != expected_tile_names:
            _fail("$files", "tiles directory differs from the declared tile set")
        tile_payloads: list[tuple[str, bytes]] = []
        total_tile_bytes = 0
        for index, uri in enumerate(tile_uris):
            name = PurePosixPath(uri).name
            payload = _read_at(
                tiles_descriptor,
                PurePosixPath(name),
                limits.maximum_tile_bytes,
                f"$.tiles[{index}].payload.uri",
            )
            total_tile_bytes += len(payload)
            if total_tile_bytes > limits.maximum_total_tile_bytes:
                _fail("$.tiles", "captured tile bytes exceed the fixed total cap")
            tile_payloads.append((uri, payload))
        sidecar_payload = _read_at(
            root_descriptor,
            sidecar_pure,
            MAXIMUM_REPLAY_SIDECAR_BYTES,
            "$.integrity.manifestSidecar",
        )
        tiles_after = os.fstat(tiles_descriptor)
        root_after = os.fstat(root_descriptor)
    except OSError as error:
        _fail("$files", f"cannot capture anchored product tree: {error}")
    finally:
        if tiles_descriptor is not None:
            os.close(tiles_descriptor)
    if _identity(tiles_before) != _identity(tiles_after):
        _fail("$files", "tiles directory changed during snapshot capture")
    if _identity(root_before) != _identity(root_after):
        _fail("$files", "product root changed during snapshot capture")
    return _ProductSnapshot(
        manifest_payload=manifest_payload,
        schema_payload=schema_payload,
        sidecar_uri=sidecar_uri,
        sidecar_payload=sidecar_payload,
        tile_payloads=tuple(tile_payloads),
    )


def _write_exclusive_regular(path: Path, payload: bytes) -> None:
    required = ("O_NOFOLLOW", "O_CLOEXEC")
    values = tuple(getattr(os, name, None) for name in required)
    if any(type(value) is not int for value in values):
        _fail("$runtime", "secure snapshot-write primitives are unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                _fail("$snapshot", "short write while creating verifier snapshot")
            offset += written
        os.fsync(descriptor)
    except OSError as error:
        _fail("$snapshot", f"cannot create private verifier snapshot: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_structural_snapshot(snapshot: _ProductSnapshot) -> dict[str, Any]:
    """Run the generic verifier only over the bounded anchored byte snapshot."""

    _assert_structural_verifier_binding()
    try:
        with tempfile.TemporaryDirectory(
            prefix="blackhole-returning-spectral-replay-"
        ) as temporary:
            root = Path(temporary)
            product = root / "product"
            tiles = product / "tiles"
            product.mkdir(mode=0o700)
            tiles.mkdir(mode=0o700)
            schema = root / "schema.json"
            _write_exclusive_regular(schema, snapshot.schema_payload)
            _write_exclusive_regular(
                product / "manifest.json",
                snapshot.manifest_payload,
            )
            _write_exclusive_regular(
                product / snapshot.sidecar_uri,
                snapshot.sidecar_payload,
            )
            for uri, payload in snapshot.tile_payloads:
                _write_exclusive_regular(product / uri, payload)
            return _STRUCTURAL_VERIFIER_ENTRY(product / "manifest.json", schema)
    except KerrReturningRadiationSpectralReplayError:
        raise
    except Exception as error:
        _fail("$", f"generic structural verification failed: {error}")


def _assert_named_root_identity(path: Path, expected: os.stat_result) -> None:
    descriptor = _open_absolute_directory(path, "$root")
    try:
        if _identity(os.fstat(descriptor)) != _identity(expected):
            _fail("$root", "manifest path no longer names the held product root")
    finally:
        os.close(descriptor)


def _grid(manifest: Mapping[str, Any]) -> SpectralFrameGrid:
    try:
        raw = manifest["frame"]
        bounds = raw["screenBounds"]
        grid = SpectralFrameGrid(
            width_pixels=raw["widthPixels"],
            height_pixels=raw["heightPixels"],
            screen_x_min=bounds["xMin"],
            screen_x_max=bounds["xMax"],
            screen_y_min=bounds["yMin"],
            screen_y_max=bounds["yMax"],
            sample_indices=tuple(raw["sampleIndices"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail("$.frame", f"cannot reconstruct exact frame grid: {error}")
    if grid.descriptor() != raw:
        _fail("$.frame", "frame grid does not round-trip exactly")
    return grid


def _source_snapshot(
    manifest: Mapping[str, Any],
    limits: ReturningRadiationSpectralReplayLimits,
) -> tuple[InputArtifact, ...]:
    artifacts: list[InputArtifact] = []
    total = 0
    root_descriptor = _open_absolute_directory(ROOT, "source:$root")
    try:
        for relative in _PRODUCER_SOURCE_FILES:
            label = f"source:{relative.as_posix()}"
            remaining = limits.maximum_source_total_bytes - total
            if remaining < 1:
                _fail(label, "source closure exceeds its fixed total-byte cap")
            payload = _read_at(
                root_descriptor,
                PurePosixPath(relative.as_posix()),
                min(limits.maximum_source_file_bytes, remaining),
                label,
            )
            total += len(payload)
            artifacts.append(
                InputArtifact(
                    f"repo-source://{relative.as_posix()}",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
    finally:
        os.close(root_descriptor)
    for uri, path in (
        (CIE_CSV_INPUT_URI, Path(DEFAULT_CIE_CSV)),
        (CIE_METADATA_INPUT_URI, Path(DEFAULT_CIE_METADATA)),
    ):
        remaining = limits.maximum_source_total_bytes - total
        if remaining < 1:
            _fail(f"science:{uri}", "input closure exceeds its total-byte cap")
        payload = _read_absolute_file(
            path,
            min(limits.maximum_source_file_bytes, remaining),
            f"science:{uri}",
        )
        total += len(payload)
        artifacts.append(
            InputArtifact(uri, len(payload), hashlib.sha256(payload).hexdigest())
        )
    current = tuple(sorted(artifacts))
    try:
        raw_spec = manifest["producer"]["jobSpec"]
        declared_inputs = raw_spec["inputs"]
        declared_source_hashes = raw_spec["producerSourceHashes"]
    except (KeyError, TypeError) as error:
        _fail("$.producer.jobSpec.inputs", f"cannot audit source closure: {error}")
    expected_inputs = [artifact.as_dict() for artifact in current]
    if declared_inputs != expected_inputs:
        _fail(
            "$.producer.jobSpec.inputs",
            "declared producer/CIE artifacts do not match current exact bytes",
        )
    source_uris = {f"repo-source://{path.as_posix()}" for path in _PRODUCER_SOURCE_FILES}
    expected_hashes = sorted(
        {artifact.sha256 for artifact in current if artifact.uri in source_uris}
    )
    if declared_source_hashes != expected_hashes:
        _fail(
            "$.producer.jobSpec.producerSourceHashes",
            "declared source hashes do not match the complete current closure",
        )
    return current


def _strict_configuration(
    manifest: dict[str, Any],
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
) -> tuple[
    SpectralPixelLayout,
    SpectralFrameGrid,
    AdaptivePixelOptions,
    JobSpec,
    dict[str, Any],
    bytes,
]:
    if manifest.get("schema") != PRODUCT_SCHEMA:
        _fail("$.schema", "unsupported spectral product schema")
    producer = manifest.get("producer")
    if type(producer) is not dict or (
        producer.get("id") != RETURNING_ADAPTIVE_TILE_PRODUCER_ID
        or producer.get("algorithmVersion")
        != RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION
    ):
        _fail("$.producer", "unsupported returning tile producer/version")
    try:
        layout = _LAYOUT_ENTRY(manifest)
        options = _OPTIONS_ENTRY(manifest["adaptivePixelOptions"])
        spec = _JOB_SPEC_ENTRY(manifest)
    except Exception as error:
        _fail("$.producer.jobSpec", f"strict JobSpec reconstruction failed: {error}")
    grid = _grid(manifest)
    if (
        type(layout) is not SpectralPixelLayout
        or type(options) is not AdaptivePixelOptions
        or type(spec) is not JobSpec
        or spec.producer != RETURNING_ADAPTIVE_TILE_PRODUCER_ID
        or spec.algorithm_version != RETURNING_ADAPTIVE_TILE_ALGORITHM_VERSION
    ):
        _fail("$.producer.jobSpec", "JobSpec has a foreign exact type or identity")
    try:
        descriptor = _SAMPLER_DESCRIPTOR_ENTRY(sampler)
        descriptor_bytes = canonical_json_bytes(descriptor)
        declared_descriptor_bytes = canonical_json_bytes(
            manifest["sampler"]["descriptor"]
        )
    except Exception as error:
        _fail("$.sampler.descriptor", f"invalid sampler descriptor: {error}")
    if (
        manifest["sampler"]["descriptor"].get("implementationId")
        != RETURNING_SAMPLER_IMPLEMENTATION_ID
        or descriptor_bytes != declared_descriptor_bytes
    ):
        _fail(
            "$.sampler.descriptor",
            "manifest is not bound to the supplied live sampler descriptor",
        )
    current_backend = _DEFAULT_BACKEND_ENTRY()
    try:
        backend_bytes = canonical_json_bytes(current_backend)
        declared_backend_bytes = canonical_json_bytes(
            manifest["runtimeNumericBackend"]["descriptor"]
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail("$.runtimeNumericBackend.descriptor", f"invalid backend: {error}")
    if backend_bytes != declared_backend_bytes:
        _fail(
            "$.runtimeNumericBackend.descriptor",
            "published numeric backend differs from the current runtime",
        )
    try:
        tile_width = max(task.width for task in spec.tasks)
        tile_height = max(task.height for task in spec.tasks)
        expected_spec = _RETURNING_JOB_SPEC_BUILDER_ENTRY(
            sampler,
            layout,
            grid,
            options,
            tile_width=tile_width,
            tile_height=tile_height,
            numeric_backend=current_backend,
            inputs=spec.inputs,
            producer_source_hashes=spec.producer_source_hashes,
        )
        expected_spec_bytes = canonical_json_bytes(expected_spec.as_dict())
        declared_spec_bytes = canonical_json_bytes(spec.as_dict())
    except Exception as error:
        _fail(
            "$.producer.jobSpec",
            f"current returning JobSpec builder rejected the declaration: {error}",
        )
    if expected_spec_bytes != declared_spec_bytes:
        _fail(
            "$.producer.jobSpec",
            "declaration is not the exact output of the current returning builder",
        )
    return layout, grid, options, spec, current_backend, descriptor_bytes


def validate_returning_radiation_spectral_live_replay(
    manifest_path: Path | str,
    *,
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
    schema_path: Path | str = DEFAULT_SCHEMA,
    limits: ReturningRadiationSpectralReplayLimits = (
        DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS
    ),
) -> ReturningRadiationSpectralLiveReplayReport:
    """Structurally authenticate and byte-replay a live returning frame."""

    if type(sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
        raise TypeError(
            "sampler must have the exact "
            "KerrReturningRadiationFiniteThicknessRaySampler type"
        )
    if type(limits) is not ReturningRadiationSpectralReplayLimits:
        raise TypeError("limits must have exact ReturningRadiationSpectralReplayLimits type")
    path = Path(manifest_path).absolute()
    if path.name != "manifest.json":
        _fail("$", "live replay input must be named 'manifest.json'")
    schema = Path(schema_path).absolute()
    root_descriptor = _open_absolute_directory(path.parent, "$root")
    root_before = os.fstat(root_descriptor)
    try:
        manifest_payload = _read_at(
            root_descriptor,
            PurePosixPath("manifest.json"),
            limits.maximum_manifest_bytes,
            "$",
        )
        schema_payload = _read_absolute_file(
            schema, limits.maximum_schema_bytes, "$schema"
        )
        manifest, tile_uris, declared_records, _declared_bytes = _preflight_manifest(
            root_descriptor,
            manifest_payload,
            limits,
        )
        product_snapshot_before = _capture_product_snapshot(
            root_descriptor,
            manifest,
            manifest_payload,
            schema_payload,
            tile_uris,
            limits,
        )

        # Only bounded path/resource snapshotting and the generic entry's own
        # identity gate occur before the generic structural authority.
        structural = _verify_structural_snapshot(product_snapshot_before)
        if structural.get("status") != (
            "scientific-spectral-frame-structural-contract-conformant"
        ):
            _fail("$", "generic structural verifier returned a bad status")
        manifest_after_structural = _read_at(
            root_descriptor,
            PurePosixPath("manifest.json"),
            limits.maximum_manifest_bytes,
            "$",
        )
        schema_after_structural = _read_absolute_file(
            schema,
            limits.maximum_schema_bytes,
            "$schema",
        )
        if manifest_after_structural != manifest_payload:
            _fail("$", "manifest changed after generic structural verification")
        if schema_after_structural != schema_payload:
            _fail("$schema", "schema changed after generic structural verification")
        if _capture_product_snapshot(
            root_descriptor,
            manifest,
            manifest_after_structural,
            schema_after_structural,
            tile_uris,
            limits,
        ) != product_snapshot_before:
            _fail("$files", "product bytes changed after structural verification")
        if _identity(os.fstat(root_descriptor)) != _identity(root_before):
            _fail("$root", "product root changed during structural verification")
        _assert_named_root_identity(path.parent, root_before)
        default_schema_payload = _read_absolute_file(
            Path(DEFAULT_SCHEMA), limits.maximum_schema_bytes, "$defaultSchema"
        )
        if schema_payload != default_schema_payload:
            _fail("$schema", "live replay requires the repository's exact schema")
        _assert_runtime_bindings()

        layout, grid, options, spec, backend_before, sampler_descriptor_before = (
            _strict_configuration(manifest, sampler)
        )
        if structural.get("recordCount") != declared_records:
            _fail("$.tiles", "structural and preflight record counts disagree")
        sources_before = _source_snapshot(manifest, limits)

        try:
            start_snapshot = _AUTHORITY_REQUIRE_LIVE_ENTRY(sampler.authority)
        except Exception as error:
            _fail("$.sampler.descriptor.authority", f"live authority failed: {error}")
        if (
            type(start_snapshot) is not ReturningThermalEmissionSnapshotV1
            or start_snapshot is not sampler._authenticated_snapshot
        ):
            _fail(
                "$.sampler.descriptor.authority",
                "supplied sampler does not own the live authenticated snapshot",
            )
        annulus_count = object.__getattribute__(start_snapshot, "annulus_count")
        if type(annulus_count) is not int or annulus_count < 1:
            _fail("$.sampler.descriptor.authority", "annulus count is malformed")
        if annulus_count > limits.maximum_annuli:
            _fail(
                "$.sampler.descriptor.authority",
                "thermal annulus count exceeds the fixed replay limit",
            )

        tile_payloads: dict[str, bytes] = {}
        record_count = 0
        total_ray_samples = 0
        replay_error: BaseException | None = None
        try:
            for tile_index, (entry, uri) in enumerate(zip(manifest["tiles"], tile_uris)):
                label = f"$.tiles[{tile_index}]"
                payload = _read_at(
                    root_descriptor,
                    uri,
                    limits.maximum_tile_bytes,
                    f"{label}.payload.uri",
                )
                if (
                    len(payload) != entry["payload"]["byteLength"]
                    or hashlib.sha256(payload).hexdigest()
                    != entry["payload"]["sha256"]
                ):
                    _fail(f"{label}.payload", "tile changed after authentication")
                tile_payloads[uri] = payload
                tile = entry["tile"]
                expected = bytearray()
                for local_index in range(entry["recordCount"]):
                    local_y, local_x = divmod(local_index, tile["width"])
                    x = tile["x"] + local_x
                    y = tile["y"] + local_y
                    x_min, x_max, y_min, y_max = _PIXEL_BOUNDS_ENTRY(grid, x, y)
                    try:
                        result = _PIXEL_INTEGRATOR_ENTRY(
                            sampler,
                            layout.observer_frequencies_hz,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            options=options,
                        )
                        encoded = _PACK_PIXEL_ENTRY(layout, result, options)
                    except Exception as error:
                        _fail(
                            f"{label}.records[{local_index}]",
                            f"live same-code replay failed closed: {error}",
                        )
                    expected.extend(encoded)
                    record_count += 1
                    total_ray_samples += result.sample_count
                if bytes(expected) != payload:
                    first = next(
                        index
                        for index, (actual, wanted) in enumerate(
                            zip(payload, expected)
                        )
                        if actual != wanted
                    )
                    _fail(
                        f"{label}.payload",
                        "published tile is not byte-identical to live replay "
                        f"(first differing byte {first})",
                    )
        except BaseException as error:
            replay_error = error
        finally:
            try:
                end_snapshot = _AUTHORITY_REQUIRE_LIVE_ENTRY(sampler.authority)
                if end_snapshot is not start_snapshot:
                    _fail(
                        "$.sampler.descriptor.authority",
                        "authority snapshot changed during live frame replay",
                    )
            except KerrReturningRadiationSpectralReplayError as error:
                replay_error = error
            except Exception as error:
                replay_error = KerrReturningRadiationSpectralReplayError(
                    "$.sampler.descriptor.authority: live authority end gate "
                    f"failed: {error}"
                )
        if replay_error is not None:
            raise replay_error

        if record_count != declared_records:
            _fail("$.tiles", "replayed record count disagrees with the manifest")
        manifest_after_replay = _read_at(
            root_descriptor,
            PurePosixPath("manifest.json"),
            limits.maximum_manifest_bytes,
            "$",
        )
        schema_after_replay = _read_absolute_file(
            schema,
            limits.maximum_schema_bytes,
            "$schema",
        )
        product_snapshot_after = _capture_product_snapshot(
            root_descriptor,
            manifest,
            manifest_after_replay,
            schema_after_replay,
            tile_uris,
            limits,
        )
        if product_snapshot_after != product_snapshot_before:
            _fail("$files", "manifest/schema/sidecar/tile bytes changed during replay")
        sources_after = _source_snapshot(manifest, limits)
        if sources_after != sources_before:
            _fail("$.producer.jobSpec.inputs", "source/CIE inputs changed during replay")
        if canonical_json_bytes(_DEFAULT_BACKEND_ENTRY()) != canonical_json_bytes(
            backend_before
        ):
            _fail("$.runtimeNumericBackend.descriptor", "numeric backend changed during replay")
        if canonical_json_bytes(_SAMPLER_DESCRIPTOR_ENTRY(sampler)) != sampler_descriptor_before:
            _fail("$.sampler.descriptor", "live sampler descriptor changed during replay")
        if _identity(os.fstat(root_descriptor)) != _identity(root_before):
            _fail("$root", "product root changed during replay")
        _assert_named_root_identity(path.parent, root_before)
        _assert_runtime_bindings()
        final_structural = _verify_structural_snapshot(product_snapshot_after)
        if final_structural != structural:
            _fail("$", "structural evidence changed during replay")

        # The final generic pass runs on the bounded snapshot.  Re-read every
        # external boundary afterwards so mutation during that pass cannot
        # escape the success result.
        manifest_final = _read_at(
            root_descriptor,
            PurePosixPath("manifest.json"),
            limits.maximum_manifest_bytes,
            "$",
        )
        schema_final = _read_absolute_file(
            schema,
            limits.maximum_schema_bytes,
            "$schema",
        )
        product_snapshot_final = _capture_product_snapshot(
            root_descriptor,
            manifest,
            manifest_final,
            schema_final,
            tile_uris,
            limits,
        )
        if product_snapshot_final != product_snapshot_before:
            _fail("$files", "product bytes changed during final structural pass")
        sources_final = _source_snapshot(manifest, limits)
        if sources_final != sources_before:
            _fail(
                "$.producer.jobSpec.inputs",
                "source/CIE inputs changed during final structural pass",
            )
        if canonical_json_bytes(_DEFAULT_BACKEND_ENTRY()) != canonical_json_bytes(
            backend_before
        ):
            _fail(
                "$.runtimeNumericBackend.descriptor",
                "numeric backend changed during final structural pass",
            )
        if canonical_json_bytes(_SAMPLER_DESCRIPTOR_ENTRY(sampler)) != (
            sampler_descriptor_before
        ):
            _fail(
                "$.sampler.descriptor",
                "live sampler descriptor changed during final structural pass",
            )
        if _identity(os.fstat(root_descriptor)) != _identity(root_before):
            _fail("$root", "product root changed during final structural pass")
        _assert_named_root_identity(path.parent, root_before)
        _assert_runtime_bindings()
        try:
            final_live_snapshot = _AUTHORITY_REQUIRE_LIVE_ENTRY(sampler.authority)
        except Exception as error:
            _fail(
                "$.sampler.descriptor.authority",
                f"final live authority gate failed: {error}",
            )
        if final_live_snapshot is not start_snapshot:
            _fail(
                "$.sampler.descriptor.authority",
                "authority snapshot changed before live replay publication",
            )

        # The parsed spec is retained as explicit evidence that the complete
        # canonical JobSpec was reconstructed, not merely its wrapper strings.
        if spec.job_key != manifest["producer"]["jobKey"]:
            _fail("$.producer.jobKey", "strict JobSpec key changed during replay")
        return {
            "id": manifest["id"],
            "status": "returning-radiation-live-sampler-byte-replay-conformant",
            "structuralContractVerified": True,
            "producerIdentityCurrentMatch": True,
            "jobSpecVerified": True,
            "samplerDescriptorLiveMatch": True,
            "numericBackendCurrentMatch": True,
            "sourceArtifactsCurrentMatch": True,
            "frameGeodesicsReplayed": True,
            "frozenThermalSnapshotBound": True,
            "thermalFixedPointReplayed": False,
            "directionCacheRecordsReplayed": False,
            "directionRaysRetraced": False,
            "independentPhysicsOracle": False,
            "pixelBytesExact": True,
            "recordCount": record_count,
            "tileCount": len(tile_uris),
            "totalRaySamples": total_ray_samples,
            "totalFrameGeodesicsReplayed": total_ray_samples * 2,
            "replayScope": (
                "same-code production whole-pixel integration and packing; exact "
                "published tile bytes with live authority at frame and pixel boundaries"
            ),
            "scientificScope": (
                "finite-grid piecewise-annulus returning-thermal frame geodesics "
                "against the already authenticated frozen thermal snapshot"
            ),
            "sourceHashScope": (
                "exact current bytes for the complete declared returning producer "
                "closure and official CIE inputs; no external signature is implied"
            ),
        }
    finally:
        os.close(root_descriptor)


# Renderer code imports this module lazily to avoid the intentional renderer
# provenance import above.  Preserve the definition-time public entry here so
# that a replacement made before that first lazy import cannot become trusted.
_CANONICAL_LIVE_REPLAY_VALIDATOR_ENTRY: Final = (
    validate_returning_radiation_spectral_live_replay
)


__all__ = (
    "DEFAULT_RETURNING_RADIATION_SPECTRAL_REPLAY_LIMITS",
    "KerrReturningRadiationSpectralReplayError",
    "ReturningRadiationSpectralLiveReplayReport",
    "ReturningRadiationSpectralReplayLimits",
    "validate_returning_radiation_spectral_live_replay",
)
