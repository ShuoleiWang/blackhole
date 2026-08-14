"""Deterministic publication of adaptive scientific spectral frame tiles.

The adaptive integrator and the fixed pixel ABI deliberately remain separate
from this artifact layer.  A :class:`JobSpec` identifies cached tile work;
publication authenticates every completed payload, decodes every public pixel
record, and writes a new non-overwriting dataset whose manifest is published
last.  No display transform is applied.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import struct
import sys
from types import MappingProxyType
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from offline.adaptive_frame import (
    AdaptivePixelOptions,
    SpectralRaySampler,
    integrate_spectral_pixel,
)
from offline.job import (
    InputArtifact,
    JobRun,
    JobSpec,
    RECEIPT_SCHEMA,
    TaskKey,
    TaskResult,
    canonical_json_bytes,
)
from offline.spectral_frame import (
    HAS_ESCAPE_DIRECTION,
    HAS_FREQUENCY_SHIFT,
    PIXEL_LAYOUT_ID,
    REQUIRED_CONVERGENCE_MASK,
    SOURCE_DISK,
    SOURCE_ESCAPED_BOUNDARY,
    ScientificSpectralPixelRecord,
    SpectralFrameError,
    SpectralPixelLayout,
    pack_adaptive_pixel,
    unpack_spectral_pixel,
)


PRODUCT_SCHEMA: Final = "blackhole.scientific-spectral-frame/v1"
ADAPTIVE_TILE_PRODUCER_ID: Final = "blackhole.adaptive-spectral-tile"
ADAPTIVE_TILE_ALGORITHM_VERSION: Final = "1.0.0"
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
_MAXIMUM_CACHE_RECEIPT_BYTES: Final = 64 * 1024
_CACHE_READ_CHUNK_BYTES: Final = 1024 * 1024
_PATH_TYPE: Final = type(Path())
KERR_SAMPLER_IMPLEMENTATION_ID: Final = (
    "exact-kerr-nt-spectral-ray-sampler/v2"
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType({
    "classification": "authenticated adaptive scalar spectral frame",
    "description": (
        "Linear observer-frame I_nu pixels in a fixed binary64 ABI with "
        "finite-stencil convergence diagnostics."
    ),
    "isDisplayImage": False,
    "isGeneralRelativisticRadiativeTransfer": False,
    "isNumericalRelativitySolver": False,
    "isPhysicsRecomputedByArtifactVerifier": False,
    "isPolarized": False,
    "prohibitedClaim": (
        "Artifact conformance authenticates records and declared provenance; "
        "it does not independently rerun geodesics, prove caustic completeness, "
        "or establish NR, GRMHD, polarization, or returning radiation."
    ),
})


class SpectralProductError(RuntimeError):
    """Raised when a spectral product cannot be published safely."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _canonical_value(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be finite canonical JSON: {error}") from error


def _canonical_object(
    value: Any,
    label: str,
    *,
    require_implementation_id: bool = False,
) -> dict[str, Any]:
    canonical = _canonical_value(value, label)
    if not isinstance(canonical, dict):
        raise ValueError(f"{label} must be an object")
    if require_implementation_id and (
        not isinstance(canonical.get("implementationId"), str)
        or not canonical["implementationId"]
    ):
        raise ValueError(f"{label} needs a non-empty implementationId")
    return canonical


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _runtime_artifact_descriptor(
    path: str | os.PathLike[str],
    label: str,
) -> dict[str, Any]:
    """Bind one native runtime artifact without embedding its host path."""

    try:
        artifact = Path(path).resolve(strict=True)
        if not artifact.is_file():
            raise ValueError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        byte_length = 0
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                byte_length += len(block)
    except (OSError, TypeError, ValueError) as error:
        raise SpectralProductError(
            f"cannot authenticate the {label} runtime artifact: {error}"
        ) from error
    return {
        "artifactName": artifact.name,
        "byteLength": byte_length,
        "sha256": digest.hexdigest(),
    }


def adaptive_pixel_options_descriptor(
    options: AdaptivePixelOptions,
) -> dict[str, Any]:
    """Return every publication-relevant adaptive option as canonical JSON."""

    if not isinstance(options, AdaptivePixelOptions):
        raise TypeError("options must be an AdaptivePixelOptions")
    return {
        "maximumDepth": options.maximum_depth,
        "maximumRayEvaluations": options.maximum_ray_evaluations,
        "minimumDepth": options.minimum_depth,
        "radianceAbsoluteTolerances": list(
            options.radiance_absolute_tolerances
        ),
        "radianceGuardCeilings": list(options.radiance_guard_ceilings),
        "radianceRelativeTolerance": options.radiance_relative_tolerance,
        "stencilVersion": options.stencil_version,
        "unresolvedSolidAngleFractionTolerance": (
            options.unresolved_solid_angle_fraction_tolerance
        ),
        "weightedDirectionToleranceRad": (
            options.weighted_direction_tolerance_rad
        ),
        "weightedLogGTolerance": options.weighted_log_g_tolerance,
    }


def default_numeric_backend_descriptor() -> dict[str, Any]:
    """Describe the runtime whose binary64/libm results enter tile payloads."""

    math_extension = getattr(math, "__file__", None)
    if not isinstance(math_extension, str) or not math_extension:
        raise SpectralProductError(
            "the numeric backend needs an authenticated native math extension"
        )
    python_build_tag, python_build_date = platform.python_build()
    libc_name, libc_version = platform.libc_ver()
    return {
        "architectureBits": 8 * struct.calcsize("P"),
        "binary64MantissaBits": sys.float_info.mant_dig,
        "byteOrder": sys.byteorder,
        "floatRadix": sys.float_info.radix,
        "implementationId": "cpython-binary64-struct-libm/v2",
        "libc": {
            "implementation": libc_name,
            "version": libc_version,
        },
        "machine": platform.machine(),
        "mathExtension": _runtime_artifact_descriptor(
            math_extension,
            "math extension",
        ),
        "operatingSystem": platform.system(),
        "operatingSystemRelease": platform.release(),
        "operatingSystemVersion": platform.version(),
        "processor": platform.processor(),
        "pythonBuild": {
            "date": python_build_date,
            "tag": python_build_tag,
        },
        "pythonCacheTag": sys.implementation.cache_tag,
        "pythonCompiler": platform.python_compiler(),
        "pythonExecutable": _runtime_artifact_descriptor(
            sys.executable,
            "Python executable",
        ),
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "structDoubleBytes": struct.calcsize("d"),
    }


@dataclass(frozen=True, slots=True)
class SpectralFrameGrid:
    """Fixed pinhole screen grid shared by all authenticated frame tiles."""

    width_pixels: int
    height_pixels: int
    screen_x_min: float
    screen_x_max: float
    screen_y_min: float
    screen_y_max: float
    sample_indices: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if (
            isinstance(self.width_pixels, bool)
            or not isinstance(self.width_pixels, int)
            or self.width_pixels < 1
            or isinstance(self.height_pixels, bool)
            or not isinstance(self.height_pixels, int)
            or self.height_pixels < 1
        ):
            raise ValueError("frame width and height must be positive integers")
        bounds = tuple(
            _finite_number(value, f"screen bound {index}")
            for index, value in enumerate(
                (
                    self.screen_x_min,
                    self.screen_x_max,
                    self.screen_y_min,
                    self.screen_y_max,
                )
            )
        )
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("screen bounds must increase strictly")
        samples = tuple(self.sample_indices)
        if (
            not samples
            or any(type(value) is not int or value < 0 for value in samples)
            or samples != tuple(sorted(set(samples)))
        ):
            raise ValueError(
                "sample_indices must be unique sorted non-negative integers"
            )
        object.__setattr__(self, "screen_x_min", bounds[0])
        object.__setattr__(self, "screen_x_max", bounds[1])
        object.__setattr__(self, "screen_y_min", bounds[2])
        object.__setattr__(self, "screen_y_max", bounds[3])
        object.__setattr__(self, "sample_indices", samples)

    @property
    def record_count(self) -> int:
        return self.width_pixels * self.height_pixels * len(self.sample_indices)

    def descriptor(self) -> dict[str, Any]:
        return {
            "heightPixels": self.height_pixels,
            "imageOrigin": "lower-left",
            "pixelOrder": "sample-then-row-major-y-then-x",
            "projection": "pinhole-screen-z-equals-one",
            "sampleIndices": list(self.sample_indices),
            "screenBounds": {
                "xMax": self.screen_x_max,
                "xMin": self.screen_x_min,
                "yMax": self.screen_y_max,
                "yMin": self.screen_y_min,
            },
            "widthPixels": self.width_pixels,
        }

    def tasks(self, tile_width: int, tile_height: int) -> tuple[TaskKey, ...]:
        for name, value in (
            ("tile_width", tile_width),
            ("tile_height", tile_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        result: list[TaskKey] = []
        for sample_index in self.sample_indices:
            for y in range(0, self.height_pixels, tile_height):
                height = min(tile_height, self.height_pixels - y)
                for x in range(0, self.width_pixels, tile_width):
                    width = min(tile_width, self.width_pixels - x)
                    result.append(
                        TaskKey(sample_index, y, x, width, height)
                    )
        return tuple(result)

    def contains_task(self, key: TaskKey) -> bool:
        return (
            isinstance(key, TaskKey)
            and key.sample_index in self.sample_indices
            and key.x + key.width <= self.width_pixels
            and key.y + key.height <= self.height_pixels
        )

    def pixel_bounds(
        self,
        x: int,
        y: int,
    ) -> tuple[float, float, float, float]:
        if not (0 <= x < self.width_pixels and 0 <= y < self.height_pixels):
            raise ValueError("pixel coordinates lie outside the frame")
        x_span = self.screen_x_max - self.screen_x_min
        y_span = self.screen_y_max - self.screen_y_min
        x_min = self.screen_x_min + x_span * x / self.width_pixels
        x_max = self.screen_x_min + x_span * (x + 1) / self.width_pixels
        y_min = self.screen_y_min + y_span * y / self.height_pixels
        y_max = self.screen_y_min + y_span * (y + 1) / self.height_pixels
        return (x_min, x_max, y_min, y_max)


def spectral_job_parameters(
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    options: AdaptivePixelOptions,
    sampler_descriptor: Mapping[str, Any],
    numeric_backend: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact scientific parameter object embedded in ``JobSpec``."""

    if not isinstance(layout, SpectralPixelLayout):
        raise TypeError("layout must be a SpectralPixelLayout")
    if not isinstance(grid, SpectralFrameGrid):
        raise TypeError("grid must be a SpectralFrameGrid")
    sampler = _canonical_object(
        sampler_descriptor,
        "sampler descriptor",
        require_implementation_id=True,
    )
    backend = _canonical_object(
        numeric_backend,
        "numeric backend",
        require_implementation_id=True,
    )
    return {
        "adaptivePixelOptions": adaptive_pixel_options_descriptor(options),
        "frame": grid.descriptor(),
        "numericBackend": backend,
        "observerFrequencyBinsHz": list(layout.observer_frequencies_hz),
        "pixelLayout": dict(layout.descriptor()),
        "samplerDescriptor": sampler,
        "schema": PRODUCT_SCHEMA,
    }


def build_spectral_job_spec(
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    options: AdaptivePixelOptions,
    sampler_descriptor: Mapping[str, Any],
    *,
    tile_width: int,
    tile_height: int,
    numeric_backend: Mapping[str, Any],
    inputs: Sequence[InputArtifact] = (),
    producer_source_hashes: Sequence[str],
    producer: str = ADAPTIVE_TILE_PRODUCER_ID,
    algorithm_version: str = ADAPTIVE_TILE_ALGORITHM_VERSION,
) -> JobSpec:
    """Build a cache identity that binds every spectral computation input."""

    source_hashes = tuple(producer_source_hashes)
    if not source_hashes:
        raise ValueError("at least one producer source hash is required")
    return JobSpec(
        producer=producer,
        algorithm_version=algorithm_version,
        tasks=grid.tasks(tile_width, tile_height),
        parameters=spectral_job_parameters(
            layout,
            grid,
            options,
            sampler_descriptor,
            numeric_backend,
        ),
        inputs=inputs,
        producer_source_hashes=source_hashes,
        record_bytes=layout.record_bytes,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveSpectralTileProducer:
    """Small deterministic ``run_job`` producer for one fixed sampler.

    The sampler is fixed across every declared sample index.  A time-dependent
    sequence must construct a separate sampler-bound job per frame or provide a
    different producer with a correspondingly different descriptor.
    """

    sampler: SpectralRaySampler
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    options: AdaptivePixelOptions
    numeric_backend: Mapping[str, Any]
    _sampler_descriptor: dict[str, Any] = field(init=False, repr=False)
    _numeric_backend: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, SpectralPixelLayout):
            raise TypeError("layout must be a SpectralPixelLayout")
        if not isinstance(self.grid, SpectralFrameGrid):
            raise TypeError("grid must be a SpectralFrameGrid")
        if not isinstance(self.options, AdaptivePixelOptions):
            raise TypeError("options must be an AdaptivePixelOptions")
        descriptor_method = getattr(self.sampler, "descriptor", None)
        if not callable(descriptor_method):
            raise TypeError("sampler must provide descriptor()")
        sampler_descriptor = _canonical_object(
            descriptor_method(),
            "sampler descriptor",
            require_implementation_id=True,
        )
        backend = _canonical_object(
            self.numeric_backend,
            "numeric backend",
            require_implementation_id=True,
        )
        object.__setattr__(self, "_sampler_descriptor", sampler_descriptor)
        object.__setattr__(self, "_numeric_backend", backend)

    @property
    def sampler_descriptor(self) -> Mapping[str, Any]:
        return _canonical_object(self._sampler_descriptor, "sampler descriptor")

    def _assert_descriptor_stable(self) -> None:
        current = _canonical_object(
            self.sampler.descriptor(),
            "sampler descriptor",
            require_implementation_id=True,
        )
        if current != self._sampler_descriptor:
            raise SpectralProductError("sampler descriptor changed after construction")

    def __call__(self, spec: JobSpec, key: TaskKey) -> bytes:
        if not isinstance(spec, JobSpec) or not isinstance(key, TaskKey):
            raise TypeError("tile producer expects a JobSpec and TaskKey")
        if spec.producer != ADAPTIVE_TILE_PRODUCER_ID:
            raise SpectralProductError("JobSpec names a different tile producer")
        if spec.algorithm_version != ADAPTIVE_TILE_ALGORITHM_VERSION:
            raise SpectralProductError("JobSpec names a different algorithm version")
        if spec.record_bytes != self.layout.record_bytes:
            raise SpectralProductError("JobSpec record size disagrees with pixel ABI")
        self._assert_descriptor_stable()
        expected_parameters = spectral_job_parameters(
            self.layout,
            self.grid,
            self.options,
            self._sampler_descriptor,
            self._numeric_backend,
        )
        if spec.as_dict()["parameters"] != expected_parameters:
            raise SpectralProductError(
                "JobSpec does not bind the producer configuration"
            )
        if key not in spec.tasks or not self.grid.contains_task(key):
            raise SpectralProductError("tile key is outside the bound frame")

        payload = bytearray()
        for y in range(key.y, key.y + key.height):
            for x in range(key.x, key.x + key.width):
                x_min, x_max, y_min, y_max = self.grid.pixel_bounds(x, y)
                result = integrate_spectral_pixel(
                    self.sampler,
                    self.layout.observer_frequencies_hz,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    options=self.options,
                )
                payload.extend(
                    pack_adaptive_pixel(self.layout, result, self.options)
                )
        self._assert_descriptor_stable()
        return bytes(payload)


@dataclass(frozen=True, slots=True)
class SpectralProductPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    product_id: str
    product_sha256: str
    tile_count: int
    record_count: int


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


class _SummaryAccumulator:
    def __init__(self, frequency_count: int) -> None:
        self.records = 0
        self.solid_angle = _CompensatedSum()
        self.integrated = tuple(_CompensatedSum() for _ in range(frequency_count))
        self.errors = tuple(_CompensatedSum() for _ in range(frequency_count))
        self.disk_area = _CompensatedSum()
        self.captured_area = _CompensatedSum()
        self.escaped_area = _CompensatedSum()
        self.unresolved_area = _CompensatedSum()
        self.minimum_g: float | None = None
        self.maximum_g: float | None = None
        self.maximum_direction_span = 0.0
        self.maximum_weighted_log_g = 0.0
        self.maximum_weighted_direction = 0.0
        self.maximum_null = 0.0
        self.maximum_metric = 0.0
        self.maximum_terminal_event = 0.0
        self.maximum_terminal_covector = 0.0
        self.maximum_disk_radius = 0.0
        self.maximum_relative_g = 0.0
        self.maximum_surface_bracket = 0.0
        self.total_ray_samples = 0
        self.maximum_depth = 0
        self.maximum_accepted = 0
        self.maximum_rejected = 0
        self.source_mask_union = 0

    def add(self, record: ScientificSpectralPixelRecord) -> None:
        _validate_record_sentinels(record)
        self.records += 1
        solid_angle = record.pixel_solid_angle_sr
        self.solid_angle.add(solid_angle)
        for index, value in enumerate(record.mean_specific_intensities_nu):
            self.integrated[index].add(value * solid_angle)
        for index, value in enumerate(record.mean_estimated_absolute_errors_nu):
            self.errors[index].add(value * solid_angle)
        self.disk_area.add(record.disk_coverage_fraction * solid_angle)
        self.captured_area.add(
            record.captured_boundary_coverage_fraction * solid_angle
        )
        self.escaped_area.add(record.escaped_boundary_coverage_fraction * solid_angle)
        self.unresolved_area.add(record.unresolved_solid_angle_fraction * solid_angle)
        if record.convergence_mask & HAS_FREQUENCY_SHIFT:
            self.minimum_g = (
                record.minimum_frequency_shift_g
                if self.minimum_g is None
                else min(self.minimum_g, record.minimum_frequency_shift_g)
            )
            self.maximum_g = (
                record.maximum_frequency_shift_g
                if self.maximum_g is None
                else max(self.maximum_g, record.maximum_frequency_shift_g)
            )
        self.maximum_direction_span = max(
            self.maximum_direction_span,
            record.maximum_escape_direction_span_rad,
        )
        self.maximum_weighted_log_g = max(
            self.maximum_weighted_log_g,
            record.weighted_log_g_variation,
        )
        self.maximum_weighted_direction = max(
            self.maximum_weighted_direction,
            record.weighted_escape_direction_variation_rad,
        )
        self.maximum_null = max(self.maximum_null, record.maximum_null_residual)
        self.maximum_metric = max(
            self.maximum_metric,
            record.maximum_metric_interpolation_error,
        )
        self.maximum_terminal_event = max(
            self.maximum_terminal_event,
            record.maximum_terminal_event_difference_m,
        )
        self.maximum_terminal_covector = max(
            self.maximum_terminal_covector,
            record.maximum_terminal_covector_relative_difference,
        )
        self.maximum_disk_radius = max(
            self.maximum_disk_radius,
            record.maximum_disk_radius_difference_m,
        )
        self.maximum_relative_g = max(
            self.maximum_relative_g,
            record.maximum_relative_g_difference,
        )
        self.maximum_surface_bracket = max(
            self.maximum_surface_bracket,
            record.maximum_surface_bracket_affine_width,
        )
        self.total_ray_samples += record.sample_count
        self.maximum_depth = max(self.maximum_depth, record.maximum_depth_reached)
        self.maximum_accepted = max(
            self.maximum_accepted,
            record.maximum_accepted_steps,
        )
        self.maximum_rejected = max(
            self.maximum_rejected,
            record.maximum_rejected_steps,
        )
        self.source_mask_union |= record.source_mask

    def descriptor(self) -> dict[str, Any]:
        return {
            "estimatedAbsoluteErrorNuSr": [item.value() for item in self.errors],
            "integratedSpecificIntensityNuSr": [
                item.value() for item in self.integrated
            ],
            "maximumAcceptedSteps": self.maximum_accepted,
            "maximumAdaptiveDepth": self.maximum_depth,
            "maximumDiskRadiusDifferenceM": self.maximum_disk_radius,
            "maximumEscapeDirectionSpanRad": self.maximum_direction_span,
            "maximumMetricInterpolationError": self.maximum_metric,
            "maximumNullResidual": self.maximum_null,
            "maximumRejectedSteps": self.maximum_rejected,
            "maximumRelativeGDifference": self.maximum_relative_g,
            "maximumSurfaceBracketAffineWidth": self.maximum_surface_bracket,
            "maximumTerminalCovectorRelativeDifference": (
                self.maximum_terminal_covector
            ),
            "maximumTerminalEventDifferenceM": self.maximum_terminal_event,
            "maximumWeightedDirectionVariationRad": (
                self.maximum_weighted_direction
            ),
            "maximumWeightedLogGVariation": self.maximum_weighted_log_g,
            "maximumFrequencyShiftG": self.maximum_g,
            "minimumFrequencyShiftG": self.minimum_g,
            "recordCount": self.records,
            "requiredConvergenceMask": REQUIRED_CONVERGENCE_MASK,
            "sourceMaskUnion": self.source_mask_union,
            "sourceSolidAngleSr": {
                "capturedBoundary": self.captured_area.value(),
                "disk": self.disk_area.value(),
                "escapedBoundary": self.escaped_area.value(),
            },
            "totalPixelSolidAngleSr": self.solid_angle.value(),
            "totalRaySamples": self.total_ray_samples,
            "unresolvedSolidAngleSr": self.unresolved_area.value(),
        }


def _is_positive_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) > 0.0


def _validate_record_sentinels(record: ScientificSpectralPixelRecord) -> None:
    if (
        record.convergence_mask & REQUIRED_CONVERGENCE_MASK
        != REQUIRED_CONVERGENCE_MASK
    ):
        raise SpectralProductError("pixel record is missing a convergence gate")
    has_g = bool(record.convergence_mask & HAS_FREQUENCY_SHIFT)
    has_direction = bool(record.convergence_mask & HAS_ESCAPE_DIRECTION)
    if bool(record.source_mask & SOURCE_DISK) != has_g:
        raise SpectralProductError("pixel g sentinel disagrees with disk coverage")
    if bool(record.source_mask & SOURCE_ESCAPED_BOUNDARY) != has_direction:
        raise SpectralProductError(
            "pixel direction sentinel disagrees with escaped coverage"
        )
    if not has_g and not (
        _is_positive_zero(record.minimum_frequency_shift_g)
        and _is_positive_zero(record.maximum_frequency_shift_g)
    ):
        raise SpectralProductError("absent g must use positive-zero sentinels")
    if not has_direction and not (
        _is_positive_zero(record.maximum_escape_direction_span_rad)
        and _is_positive_zero(
            record.weighted_escape_direction_variation_rad
        )
    ):
        raise SpectralProductError(
            "absent escape direction must use positive-zero diagnostics"
        )


def _secure_directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise SpectralProductError(
            "platform lacks O_NOFOLLOW/O_DIRECTORY cache-consumer primitives"
        )
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_file_chain(path: Path, label: str) -> tuple[list[int], int]:
    """Open every absolute-path component without following a symlink."""

    if (
        type(path) is not _PATH_TYPE
        or not path.is_absolute()
        or path.anchor != os.sep
    ):
        raise SpectralProductError(f"{label} path must be an exact absolute Path")
    components = path.parts[1:]
    if not components or any(
        type(component) is not str
        or component in ("", ".", "..")
        or os.sep in component
        for component in components
    ):
        raise SpectralProductError(f"{label} path is not canonical")

    directory_flags = _secure_directory_open_flags()
    descriptors: list[int] = []
    file_descriptor = -1
    try:
        descriptors.append(os.open(os.sep, directory_flags))
        for component in components[:-1]:
            descriptors.append(
                os.open(
                    component,
                    directory_flags,
                    dir_fd=descriptors[-1],
                )
            )
        file_descriptor = os.open(
            components[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=descriptors[-1],
        )
        return descriptors, file_descriptor
    except OSError as error:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise SpectralProductError(
            f"unable to open {label} without following symlinks"
        ) from error


def _path_chain_identity(
    directory_descriptors: Sequence[int],
    file_descriptor: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode)
        for snapshot in (
            *(os.fstat(descriptor) for descriptor in directory_descriptors),
            os.fstat(file_descriptor),
        )
    )


def _read_stable_regular_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise TypeError("maximum_bytes must be an exact non-negative integer")

    directory_descriptors: list[int] = []
    descriptor = -1
    reopened_directories: list[int] = []
    reopened_descriptor = -1
    try:
        directory_descriptors, descriptor = _open_absolute_file_chain(path, label)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SpectralProductError(f"{label} must be a regular file")
        if before.st_size > maximum_bytes:
            raise SpectralProductError(f"{label} exceeds its hard byte limit")

        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(_CACHE_READ_CHUNK_BYTES, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > maximum_bytes:
            raise SpectralProductError(f"{label} exceeds its hard byte limit")

        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise SpectralProductError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise SpectralProductError(f"{label} changed while being read")

        original_chain = _path_chain_identity(
            directory_descriptors,
            descriptor,
        )
        reopened_directories, reopened_descriptor = _open_absolute_file_chain(
            path,
            label,
        )
        if (
            _path_chain_identity(reopened_directories, reopened_descriptor)
            != original_chain
        ):
            raise SpectralProductError(f"{label} path changed while being read")
        return payload
    except OSError as error:
        raise SpectralProductError(f"unable to read {label}: {error}") from error
    finally:
        if reopened_descriptor >= 0:
            os.close(reopened_descriptor)
        for current in reversed(reopened_directories):
            os.close(current)
        if descriptor >= 0:
            os.close(descriptor)
        for current in reversed(directory_descriptors):
            os.close(current)


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SpectralProductError(f"{label} is invalid JSON: {error}") from error


def _validate_task_receipt(
    job_spec: JobSpec,
    result: TaskResult,
    *,
    byte_length: int,
    record_count: int,
    sha256: str,
) -> None:
    key = result.key
    expected_payload_name = f"{key.file_stem}.bin"
    expected_receipt_name = f"{key.file_stem}.receipt.json"
    if (
        result.payload_path.name != expected_payload_name
        or result.receipt_path.name != expected_receipt_name
        or result.payload_path.parent != result.receipt_path.parent
    ):
        raise SpectralProductError(
            f"cached tile {key.file_stem} paths do not form a canonical task pair"
        )
    receipt_payload = _read_stable_regular_file(
        result.receipt_path,
        f"cached tile {key.file_stem} receipt",
        maximum_bytes=_MAXIMUM_CACHE_RECEIPT_BYTES,
    )
    receipt = _strict_json_bytes(
        receipt_payload,
        f"cached tile {key.file_stem} receipt",
    )
    expected = {
        "byteLength": byte_length,
        "jobKey": job_spec.job_key,
        "payload": expected_payload_name,
        "recordCount": record_count,
        "schema": RECEIPT_SCHEMA,
        "sha256": sha256,
        "task": key.as_dict(),
    }
    if receipt != expected or receipt_payload != canonical_json_bytes(expected):
        raise SpectralProductError(
            f"cached tile {key.file_stem} receipt does not bind its JobSpec payload"
        )


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
            raise SpectralProductError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        removed = False
        try:
            temporary.unlink()
            removed = True
        except FileNotFoundError:
            pass
        if removed:
            _fsync_directory(path.parent)


def _promote_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one complete directory without replacing a target."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    library = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        renamex = library.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        rename_exclusive = 0x00000004
        result = renamex(
            source_bytes,
            destination_bytes,
            rename_exclusive,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        renameat2 = library.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            source_bytes,
            at_fdcwd,
            destination_bytes,
            rename_noreplace,
        )
    else:
        raise SpectralProductError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        _fsync_directory(destination.parent)
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise SpectralProductError(
            f"refusing to overwrite existing output {destination}"
        )
    raise SpectralProductError(
        f"unable to atomically publish {destination}: "
        f"{os.strerror(error_number)}"
    )


def _tile_uri(key: TaskKey) -> str:
    return f"tiles/{key.file_stem}.spx"


def _validate_complete_tile_topology(
    tasks: Sequence[TaskKey],
    grid: SpectralFrameGrid,
) -> None:
    ordered = tuple(tasks)
    if ordered != tuple(sorted(ordered)):
        raise SpectralProductError("tiles are not in canonical TaskKey order")
    if {task.sample_index for task in ordered} != set(grid.sample_indices):
        raise SpectralProductError("tile sample indices do not match the frame")
    for task in ordered:
        if not grid.contains_task(task):
            raise SpectralProductError("a tile lies outside the frame")
    for sample_index in grid.sample_indices:
        sample_tasks = tuple(
            task for task in ordered if task.sample_index == sample_index
        )
        if not sample_tasks:
            raise SpectralProductError("a frame sample has no tiles")
        boundaries = sorted(
            {
                0,
                grid.height_pixels,
                *(task.y for task in sample_tasks),
                *(task.y + task.height for task in sample_tasks),
            }
        )
        for lower, upper in zip(boundaries, boundaries[1:]):
            if upper <= lower:
                raise SpectralProductError("tile y topology is not increasing")
            intervals = sorted(
                (task.x, task.x + task.width)
                for task in sample_tasks
                if task.y <= lower and task.y + task.height >= upper
            )
            cursor = 0
            for start, end in intervals:
                if start != cursor or end <= start:
                    raise SpectralProductError(
                        "tiles overlap or leave a horizontal coverage gap"
                    )
                cursor = end
            if cursor != grid.width_pixels:
                raise SpectralProductError(
                    "tiles do not completely cover a horizontal frame strip"
                )


def _validate_job_and_configuration(
    job_spec: JobSpec,
    job_run: JobRun,
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    options: AdaptivePixelOptions,
    sampler_descriptor: Mapping[str, Any],
    numeric_backend: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(job_spec, JobSpec) or not isinstance(job_run, JobRun):
        raise TypeError("publication requires a JobSpec and JobRun")
    if not isinstance(layout, SpectralPixelLayout):
        raise TypeError("layout must be a SpectralPixelLayout")
    if not isinstance(grid, SpectralFrameGrid):
        raise TypeError("grid must be a SpectralFrameGrid")
    if not isinstance(options, AdaptivePixelOptions):
        raise TypeError("options must be an AdaptivePixelOptions")
    if not job_spec.producer_source_hashes:
        raise SpectralProductError("JobSpec must bind producer source hashes")
    if job_run.job_key != job_spec.job_key:
        raise SpectralProductError("JobRun belongs to a different JobSpec")
    if tuple(result.key for result in job_run.results) != tuple(job_spec.tasks):
        raise SpectralProductError("JobRun tasks are not complete and canonical")
    _validate_complete_tile_topology(job_spec.tasks, grid)
    if job_spec.record_bytes != layout.record_bytes:
        raise SpectralProductError("JobSpec record size disagrees with pixel ABI")
    sampler = _canonical_object(
        sampler_descriptor,
        "sampler descriptor",
        require_implementation_id=True,
    )
    backend = _canonical_object(
        numeric_backend,
        "numeric backend",
        require_implementation_id=True,
    )
    expected_parameters = spectral_job_parameters(
        layout,
        grid,
        options,
        sampler,
        backend,
    )
    if job_spec.as_dict()["parameters"] != expected_parameters:
        raise SpectralProductError(
            "JobSpec parameters do not bind the publication configuration"
        )
    return sampler, backend, expected_parameters


def publish_spectral_product(
    output_directory: Path | str,
    *,
    job_spec: JobSpec,
    job_run: JobRun,
    layout: SpectralPixelLayout,
    grid: SpectralFrameGrid,
    options: AdaptivePixelOptions,
    sampler_descriptor: Mapping[str, Any],
    numeric_backend: Mapping[str, Any],
) -> SpectralProductPublication:
    """Publish a new authenticated dataset without replacing any existing path."""

    sampler, backend, configuration_parameters = _validate_job_and_configuration(
        job_spec,
        job_run,
        layout,
        grid,
        options,
        sampler_descriptor,
        numeric_backend,
    )
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise SpectralProductError(
            f"refusing to overwrite existing output {output}"
        )
    summary = _SummaryAccumulator(layout.frequency_count)
    validated_tiles: list[tuple[TaskResult, int, int, str]] = []
    for result in job_run.results:
        key = result.key
        if not grid.contains_task(key):
            raise SpectralProductError("JobSpec contains a tile outside the frame")
        expected_records = key.width * key.height
        expected_bytes = expected_records * layout.record_bytes
        payload = _read_stable_regular_file(
            result.payload_path,
            f"cached tile {key.file_stem}",
            maximum_bytes=expected_bytes,
        )
        actual_hash = hashlib.sha256(payload).hexdigest()
        if (
            result.record_count != expected_records
            or result.byte_length != expected_bytes
            or len(payload) != expected_bytes
            or result.sha256 != actual_hash
        ):
            raise SpectralProductError(
                f"cached tile {key.file_stem} metadata or hash is inconsistent"
            )
        _validate_task_receipt(
            job_spec,
            result,
            byte_length=expected_bytes,
            record_count=expected_records,
            sha256=actual_hash,
        )
        for record_index in range(expected_records):
            offset = record_index * layout.record_bytes
            try:
                record = unpack_spectral_pixel(
                    layout,
                    payload[offset : offset + layout.record_bytes],
                )
            except (SpectralFrameError, TypeError, ValueError) as error:
                raise SpectralProductError(
                    f"cached tile {key.file_stem} record {record_index} is invalid: "
                    f"{error}"
                ) from error
            summary.add(record)
        validated_tiles.append(
            (result, expected_records, expected_bytes, actual_hash)
        )

    summary_descriptor = summary.descriptor()
    if summary_descriptor["recordCount"] != grid.record_count:
        raise SpectralProductError("tiles do not provide complete frame coverage")

    tile_entries: list[dict[str, Any]] = []
    for result, expected_records, expected_bytes, expected_hash in validated_tiles:
        key = result.key
        uri = _tile_uri(key)
        tile_entries.append(
            {
                "payload": {
                    "byteLength": expected_bytes,
                    "sha256": expected_hash,
                    "uri": uri,
                },
                "recordCount": expected_records,
                "recordOrder": "row-major-local-y-then-x",
                "sampleIndex": key.sample_index,
                "tile": {
                    "height": key.height,
                    "width": key.width,
                    "x": key.x,
                    "y": key.y,
                },
            }
        )

    configuration = {
        **configuration_parameters,
        "jobKey": job_spec.job_key,
        "jobSpec": job_spec.as_dict(),
    }
    configuration_sha256 = _sha256_canonical(configuration)
    product_identity = {
        "configurationSha256": configuration_sha256,
        "schema": PRODUCT_SCHEMA,
        "summary": summary_descriptor,
        "tiles": tile_entries,
    }
    product_sha256 = _sha256_canonical(product_identity)
    product_id = f"scientific-spectral-frame-{product_sha256[:24]}"
    manifest = {
        "adaptivePixelOptions": adaptive_pixel_options_descriptor(options),
        "frame": grid.descriptor(),
        "id": product_id,
        "integrity": {
            "configurationSha256": configuration_sha256,
            "manifestSidecar": SIDECAR_NAME,
            "productSha256": product_sha256,
        },
        "observerFrequencyBinsHz": list(layout.observer_frequencies_hz),
        "pixelLayout": dict(layout.descriptor()),
        "producer": {
            "algorithmVersion": job_spec.algorithm_version,
            "id": job_spec.producer,
            "jobKey": job_spec.job_key,
            "jobSpec": job_spec.as_dict(),
        },
        "runtimeNumericBackend": {
            "descriptor": backend,
            "descriptorSha256": _sha256_canonical(backend),
        },
        "sampler": {
            "descriptor": sampler,
            "descriptorSha256": _sha256_canonical(sampler),
        },
        "schema": PRODUCT_SCHEMA,
        "scientificStatus": dict(SCIENTIFIC_STATUS),
        "summary": summary_descriptor,
        "tiles": tile_entries,
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    sidecar = f"{manifest_sha256}  {MANIFEST_NAME}\n".encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        (staging / "tiles").mkdir(mode=0o700)
        for result, expected_records, expected_bytes, expected_hash in validated_tiles:
            key = result.key
            payload = _read_stable_regular_file(
                result.payload_path,
                f"cached tile {key.file_stem} publication snapshot",
                maximum_bytes=expected_bytes,
            )
            actual_hash = hashlib.sha256(payload).hexdigest()
            if len(payload) != expected_bytes or actual_hash != expected_hash:
                raise SpectralProductError(
                    f"cached tile {key.file_stem} changed after validation"
                )
            _validate_task_receipt(
                job_spec,
                result,
                byte_length=expected_bytes,
                record_count=expected_records,
                sha256=expected_hash,
            )
            _atomic_write_no_replace(staging / _tile_uri(key), payload)
        _atomic_write_no_replace(staging / SIDECAR_NAME, sidecar)
        _atomic_write_no_replace(staging / MANIFEST_NAME, manifest_payload)
        _fsync_directory(staging)
        _promote_directory_no_replace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            _fsync_directory(output.parent)
        raise
    return SpectralProductPublication(
        output_directory=output,
        manifest_path=output / MANIFEST_NAME,
        manifest_sha256=manifest_sha256,
        product_id=product_id,
        product_sha256=product_sha256,
        tile_count=len(tile_entries),
        record_count=summary_descriptor["recordCount"],
    )


__all__ = (
    "ADAPTIVE_TILE_ALGORITHM_VERSION",
    "ADAPTIVE_TILE_PRODUCER_ID",
    "AdaptiveSpectralTileProducer",
    "KERR_SAMPLER_IMPLEMENTATION_ID",
    "MANIFEST_NAME",
    "PRODUCT_SCHEMA",
    "SCIENTIFIC_STATUS",
    "SIDECAR_NAME",
    "SpectralFrameGrid",
    "SpectralProductError",
    "SpectralProductPublication",
    "adaptive_pixel_options_descriptor",
    "build_spectral_job_spec",
    "default_numeric_backend_descriptor",
    "publish_spectral_product",
    "spectral_job_parameters",
)
