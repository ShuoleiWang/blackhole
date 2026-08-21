"""Deterministic numerical replay of exact-Kerr Novikov--Thorne products.

This verifier is deliberately narrower than an independent physics oracle.  It
reconstructs the closed production sampler from an authenticated
``offline-scientific-spectral-frame/v1`` manifest, reruns the same code family,
and requires every replayed public pixel record to be byte-identical.  That
detects payload/configuration inconsistency, but shared implementation errors
remain shared errors.  A fully resealed alternative configuration that happens
to produce the same public ABI needs an external manifest trust anchor to be
distinguished from a new valid product.  This is not NR, GRMHD, or an analytic
oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

from offline.adaptive_frame import AdaptivePixelOptions, integrate_spectral_pixel
from offline.disk_atmosphere import (
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
)
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import InputArtifact, canonical_json_bytes
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    KerrDiskRaySampler,
    PowerLawEscapedObserverSpectrum,
)
from offline.spectral_frame import (
    ScientificSpectralPixelRecord,
    SpectralPixelLayout,
    pack_adaptive_pixel,
    unpack_spectral_pixel,
)
from offline.spectral_product import (
    ADAPTIVE_TILE_ALGORITHM_VERSION,
    ADAPTIVE_TILE_PRODUCER_ID,
    KERR_SAMPLER_IMPLEMENTATION_ID,
    PRODUCT_SCHEMA,
    SpectralFrameGrid,
    adaptive_pixel_options_descriptor,
    default_numeric_backend_descriptor,
)
from scripts.render_offline_kerr_nt_frame import PRODUCER_SOURCE_FILES
from scripts.verify_offline_spectral_frame import (
    DEFAULT_SCHEMA,
    validate_scientific_spectral_frame,
)


ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS = 471


class KerrNtReplayError(RuntimeError):
    """A fail-closed exact-Kerr/NT replay verification failure."""


def _fail(path: str, message: str) -> NoReturn:
    raise KerrNtReplayError(f"{path}: {message}")


@dataclass(frozen=True, slots=True)
class ReplayResourceLimits:
    """Explicit finite work and artifact limits for one replay invocation."""

    maximum_manifest_bytes: int = 64 * 1024 * 1024
    maximum_source_file_bytes: int = 64 * 1024 * 1024
    maximum_product_bytes: int = 1024 * 1024 * 1024
    maximum_tile_bytes: int = 64 * 1024 * 1024
    maximum_tiles: int = 4096
    maximum_records: int = 4096
    maximum_frequency_bins: int = MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS
    maximum_total_ray_evaluations: int = 10_000_000
    maximum_adaptive_depth: int = 12
    maximum_ray_accepted_steps: int = 200_000
    maximum_ray_rejected_steps: int = 200_000
    maximum_ray_event_iterations: int = 256
    maximum_surface_iterations: int = 256
    maximum_surface_reintegrations: int = 200_000
    maximum_surface_subdivisions_per_segment: int = 64
    maximum_affine_length_m: float = 10_000.0

    def __post_init__(self) -> None:
        integer_names = (
            "maximum_manifest_bytes",
            "maximum_source_file_bytes",
            "maximum_product_bytes",
            "maximum_tile_bytes",
            "maximum_tiles",
            "maximum_records",
            "maximum_frequency_bins",
            "maximum_total_ray_evaluations",
            "maximum_ray_accepted_steps",
            "maximum_ray_rejected_steps",
            "maximum_ray_event_iterations",
            "maximum_surface_iterations",
            "maximum_surface_reintegrations",
            "maximum_surface_subdivisions_per_segment",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.maximum_adaptive_depth) is not int
            or self.maximum_adaptive_depth < 0
        ):
            raise ValueError("maximum_adaptive_depth must be a non-negative integer")
        if self.maximum_frequency_bins > MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS:
            raise ValueError(
                "maximum_frequency_bins may not exceed the official CIE "
                f"{MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS}-bin grid"
            )
        if (
            isinstance(self.maximum_affine_length_m, bool)
            or not isinstance(self.maximum_affine_length_m, (int, float))
            or not math.isfinite(float(self.maximum_affine_length_m))
            or self.maximum_affine_length_m <= 0.0
        ):
            raise ValueError("maximum_affine_length_m must be finite and positive")
        object.__setattr__(
            self,
            "maximum_affine_length_m",
            float(self.maximum_affine_length_m),
        )


DEFAULT_REPLAY_LIMITS = ReplayResourceLimits()


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(label, f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(label, f"non-finite JSON number {value!r} is forbidden")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"invalid UTF-8 JSON: {error}")
    if not isinstance(parsed, dict):
        _fail(label, "JSON root must be an object")
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        _fail(label, f"JSON is not finite canonical data: {error}")
    if canonical != payload:
        _fail(label, "manifest is not in canonical JSON encoding")
    return parsed


def _read_stable_regular(path: Path, maximum_bytes: int, label: str) -> bytes:
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
            _fail(label, f"file exceeds the {maximum_bytes}-byte replay limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
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
        _fail(path, "URI must be a non-empty relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        _fail(path, "URI must be normalized and traversal-free")
    return pure


def _read_relative_file(
    root: Path,
    uri: Any,
    maximum_bytes: int,
    label: str,
) -> bytes:
    pure = _normalized_uri(uri, label)
    if root.is_symlink():
        _fail(label, "symlinked artifact roots are forbidden")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directories: list[int] = []
    file_descriptor: int | None = None
    try:
        directories.append(os.open(root, directory_flags))
        for part in pure.parts[:-1]:
            directories.append(
                os.open(part, directory_flags, dir_fd=directories[-1])
            )
        file_descriptor = os.open(
            pure.parts[-1],
            file_flags,
            dir_fd=directories[-1],
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(label, "artifact must be a regular file")
        if before.st_size > maximum_bytes:
            _fail(label, f"artifact exceeds the {maximum_bytes}-byte replay limit")
        with os.fdopen(file_descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after = os.fstat(file_descriptor)
    except OSError as error:
        _fail(label, f"unable to open traversal-safe artifact: {error}")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directories):
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
        _fail(label, "artifact changed while it was being read")
    return payload


def _relative_regular_size(root: Path, uri: Any, label: str) -> int:
    """Return a traversal-safe regular artifact size without reading payload."""

    pure = _normalized_uri(uri, label)
    if root.is_symlink():
        _fail(label, "symlinked artifact roots are forbidden")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directories: list[int] = []
    file_descriptor: int | None = None
    try:
        directories.append(os.open(root, directory_flags))
        for part in pure.parts[:-1]:
            directories.append(
                os.open(part, directory_flags, dir_fd=directories[-1])
            )
        file_descriptor = os.open(
            pure.parts[-1],
            file_flags,
            dir_fd=directories[-1],
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "artifact must be a regular file")
        return metadata.st_size
    except OSError as error:
        _fail(label, f"unable to stat traversal-safe artifact: {error}")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)


def _preflight_product_tree(
    root: Path,
    tile_uris: Sequence[str],
    sidecar_uri: Any,
) -> None:
    """Bound directory enumeration before the structural verifier walks it."""

    sidecar = _normalized_uri(sidecar_uri, "$.integrity.manifestSidecar")
    if len(sidecar.parts) != 1:
        _fail("$.integrity.manifestSidecar", "sidecar must be at product root")
    expected_tiles: set[str] = set()
    for index, uri in enumerate(tile_uris):
        pure = _normalized_uri(uri, f"$.tiles[{index}].payload.uri")
        if len(pure.parts) != 2 or pure.parts[0] != "tiles":
            _fail(
                f"$.tiles[{index}].payload.uri",
                "replay tiles must be direct regular children of tiles/",
            )
        if pure.parts[1] in expected_tiles:
            _fail(f"$.tiles[{index}].payload.uri", "duplicate tile URI")
        expected_tiles.add(pure.parts[1])
    expected_root = {"manifest.json", sidecar.parts[0], "tiles"}
    if root.is_symlink():
        _fail("$files", "symlinked product roots are forbidden")
    try:
        observed_root: set[str] = set()
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name not in expected_root:
                    _fail("$files", f"undeclared root entry {entry.name!r}")
                if entry.is_symlink():
                    _fail("$files", f"symlinked root entry {entry.name!r}")
                if entry.name == "tiles":
                    if not entry.is_dir(follow_symlinks=False):
                        _fail("$files", "tiles must be a regular directory")
                elif not entry.is_file(follow_symlinks=False):
                    _fail("$files", f"root artifact {entry.name!r} is not regular")
                observed_root.add(entry.name)
        if observed_root != expected_root:
            _fail("$files", "product root is incomplete")

        observed_tiles: set[str] = set()
        with os.scandir(root / "tiles") as entries:
            for entry in entries:
                if entry.name not in expected_tiles:
                    _fail("$files", f"undeclared tile entry {entry.name!r}")
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    _fail("$files", f"tile {entry.name!r} is not a regular file")
                observed_tiles.add(entry.name)
        if observed_tiles != expected_tiles:
            _fail("$files", "declared tile set is incomplete")
    except OSError as error:
        _fail("$files", f"cannot enumerate product tree safely: {error}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(path, "expected a finite number")
    return float(value)


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "expected a boolean")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _ray_options(value: Any, path: str) -> RayTraceOptions:
    raw = _mapping(value, path)
    try:
        return RayTraceOptions(
            absolute_tolerance=_number(raw["absoluteTolerance"], path),
            relative_tolerance=_number(raw["relativeTolerance"], path),
            initial_step=_number(raw["initialStep"], path),
            minimum_step=_number(raw["minimumStep"], path),
            maximum_step=_number(raw["maximumStep"], path),
            maximum_affine_length=_number(raw["maximumAffineLength"], path),
            maximum_accepted_steps=_integer(raw["maximumAcceptedSteps"], path),
            maximum_rejected_steps=_integer(raw["maximumRejectedSteps"], path),
            null_residual_limit=_number(raw["nullResidualLimit"], path),
            metric_interpolation_error_limit=_number(
                raw["metricInterpolationErrorLimit"], path
            ),
            event_value_tolerance=_number(raw["eventValueTolerance"], path),
            event_affine_tolerance=_number(raw["eventAffineTolerance"], path),
            event_maximum_iterations=_integer(
                raw["eventMaximumIterations"], path
            ),
            record_path=_boolean(raw["recordPath"], path),
        )
    except KeyError as error:
        _fail(path, f"missing ray option {error.args[0]!r}")


def _surface_options(value: Any, path: str) -> SurfaceEventOptions:
    raw = _mapping(value, path)
    try:
        return SurfaceEventOptions(
            absolute_tolerance=_number(raw["absoluteTolerance"], path),
            relative_tolerance=_number(raw["relativeTolerance"], path),
            null_residual_limit=_number(raw["nullResidualLimit"], path),
            metric_interpolation_error_limit=_number(
                raw["metricInterpolationErrorLimit"], path
            ),
            surface_value_tolerance=_number(raw["surfaceValueTolerance"], path),
            affine_tolerance=_number(raw["affineTolerance"], path),
            maximum_iterations=_integer(raw["maximumIterations"], path),
            maximum_reintegrations=_integer(
                raw["maximumReintegrations"], path
            ),
            subdivisions_per_segment=_integer(
                raw["subdivisionsPerSegment"], path
            ),
        )
    except KeyError as error:
        _fail(path, f"missing surface option {error.args[0]!r}")


def _angular_law(value: Any):
    wrapper = _mapping(value, "$.sampler.descriptor.angularEmission")
    descriptor = _mapping(
        wrapper.get("descriptor"),
        "$.sampler.descriptor.angularEmission.descriptor",
    )
    implementation = _string(
        descriptor.get("implementationId"),
        "$.sampler.descriptor.angularEmission.descriptor.implementationId",
    )
    if implementation == "isotropic-angular-emission/v1":
        return IsotropicAngularEmission()
    if implementation == "flux-conserving-linear-limb-darkening/v1":
        try:
            coefficient = descriptor["coefficient"]
        except KeyError:
            _fail("$.sampler.descriptor.angularEmission.descriptor", "missing coefficient")
        return FluxConservingLinearLimbDarkening(
            _number(coefficient, "$.sampler.descriptor.angularEmission.descriptor.coefficient")
        )
    _fail(
        "$.sampler.descriptor.angularEmission.descriptor.implementationId",
        f"unsupported closed angular law {implementation!r}",
    )


def _escaped_spectrum(value: Any):
    wrapper = _mapping(value, "$.sampler.descriptor.escapedObserverSpectrum")
    descriptor = _mapping(
        wrapper.get("descriptor"),
        "$.sampler.descriptor.escapedObserverSpectrum.descriptor",
    )
    implementation = _string(
        descriptor.get("implementationId"),
        "$.sampler.descriptor.escapedObserverSpectrum.descriptor.implementationId",
    )
    if implementation == "dark-observer-frame-escape-spectrum/v1":
        return DarkEscapedObserverSpectrum()
    if implementation == "power-law-observer-frame-escape-spectrum/v1":
        try:
            return PowerLawEscapedObserverSpectrum(
                reference_specific_intensity_nu=_number(
                    descriptor["referenceSpecificIntensityNu"],
                    "$.sampler.descriptor.escapedObserverSpectrum.descriptor."
                    "referenceSpecificIntensityNu",
                ),
                reference_frequency_hz=_number(
                    descriptor["referenceFrequencyHz"],
                    "$.sampler.descriptor.escapedObserverSpectrum.descriptor.referenceFrequencyHz",
                ),
                spectral_index=_number(
                    descriptor["spectralIndex"],
                    "$.sampler.descriptor.escapedObserverSpectrum.descriptor.spectralIndex",
                ),
            )
        except KeyError as error:
            _fail(
                "$.sampler.descriptor.escapedObserverSpectrum.descriptor",
                f"missing power-law field {error.args[0]!r}",
            )
    _fail(
        "$.sampler.descriptor.escapedObserverSpectrum.descriptor.implementationId",
        f"unsupported closed escape spectrum {implementation!r}",
    )


def reconstruct_kerr_nt_sampler(descriptor: Mapping[str, Any]) -> KerrDiskRaySampler:
    """Reconstruct and content-round-trip the supported production sampler."""

    raw = _mapping(descriptor, "$.sampler.descriptor")
    if raw.get("implementationId") != KERR_SAMPLER_IMPLEMENTATION_ID:
        _fail(
            "$.sampler.descriptor.implementationId",
            f"only {KERR_SAMPLER_IMPLEMENTATION_ID!r} can be replayed",
        )
    if _integer(raw.get("version"), "$.sampler.descriptor.version") != 2:
        _fail("$.sampler.descriptor.version", "only sampler version 2 is supported")
    try:
        metric_raw = _mapping(raw["metric"], "$.sampler.descriptor.metric")
        source_id = _string(
            metric_raw["sourceId"], "$.sampler.descriptor.metric.sourceId"
        )
        time_dependent = _boolean(
            metric_raw["timeDependent"],
            "$.sampler.descriptor.metric.timeDependent",
        )
        if source_id != "analytic-kerr-kerr-schild" or time_dependent:
            _fail(
                "$.sampler.descriptor.metric",
                "replay requires the stationary analytic Kerr-Schild provider",
            )
        metric = KerrKerrSchildMetric(
            mass_m=_number(metric_raw["massM"], "$.sampler.descriptor.metric.massM"),
            spin_a_m=_number(metric_raw["spinAM"], "$.sampler.descriptor.metric.spinAM"),
            singularity_guard_m=_number(
                metric_raw["singularityGuardM"],
                "$.sampler.descriptor.metric.singularityGuardM",
            ),
        )

        termination_raw = _mapping(
            raw["termination"], "$.sampler.descriptor.termination"
        )
        capture_target = _string(
            termination_raw["captureTargetId"],
            "$.sampler.descriptor.termination.captureTargetId",
        )
        escape_target = _string(
            termination_raw["escapeTargetId"],
            "$.sampler.descriptor.termination.escapeTargetId",
        )
        if capture_target not in {
            "analytic-kerr-event-horizon",
            "analytic-kerr-stretched-horizon",
        } or escape_target != "analytic-kerr-escape-worldtube":
            _fail(
                "$.sampler.descriptor.termination",
                "unsupported terminal-surface identity",
            )
        termination = KerrOblateTermination(
            spin_a_m=_number(
                termination_raw["spinAM"],
                "$.sampler.descriptor.termination.spinAM",
            ),
            capture_radius_m=_number(
                termination_raw["captureRadiusM"],
                "$.sampler.descriptor.termination.captureRadiusM",
            ),
            escape_radius_m=_number(
                termination_raw["escapeRadiusM"],
                "$.sampler.descriptor.termination.escapeRadiusM",
            ),
            capture_target_id=capture_target,
            escape_target_id=escape_target,
        )

        disk_raw = _mapping(raw["disk"], "$.sampler.descriptor.disk")
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=_number(
                disk_raw["blackHoleMassKg"],
                "$.sampler.descriptor.disk.blackHoleMassKg",
            ),
            mass_accretion_rate_kg_s=_number(
                disk_raw["massAccretionRateKgS"],
                "$.sampler.descriptor.disk.massAccretionRateKgS",
            ),
            orientation=_string(
                disk_raw["orientation"],
                "$.sampler.descriptor.disk.orientation",
            ),
            colour_correction=_number(
                disk_raw["colourCorrection"],
                "$.sampler.descriptor.disk.colourCorrection",
            ),
        )
        ray_raw = _mapping(raw["rayOptions"], "$.sampler.descriptor.rayOptions")
        surface_raw = _mapping(
            raw["surfaceOptions"], "$.sampler.descriptor.surfaceOptions"
        )
        fine_options = _ray_options(
            ray_raw["fine"], "$.sampler.descriptor.rayOptions.fine"
        )
        surface_options = _surface_options(
            surface_raw["fine"], "$.sampler.descriptor.surfaceOptions.fine"
        )
        observer_raw = _mapping(raw["observer"], "$.sampler.descriptor.observer")
        convergence = _mapping(
            raw["convergence"], "$.sampler.descriptor.convergence"
        )
        frequency_shift = _mapping(
            raw["frequencyShift"], "$.sampler.descriptor.frequencyShift"
        )
        sampler = KerrDiskRaySampler(
            metric=metric,
            observer_radius_m=_number(
                observer_raw["radiusM"], "$.sampler.descriptor.observer.radiusM"
            ),
            termination=termination,
            disk=disk,
            outer_radius_m=_number(
                disk_raw["outerRadiusM"], "$.sampler.descriptor.disk.outerRadiusM"
            ),
            escaped_observer_spectrum=_escaped_spectrum(
                raw["escapedObserverSpectrum"]
            ),
            fine_options=fine_options,
            surface_options=surface_options,
            angular_emission_law=_angular_law(raw["angularEmission"]),
            observer_theta_rad=_number(
                observer_raw["thetaRad"], "$.sampler.descriptor.observer.thetaRad"
            ),
            observer_phi_ks_rad=_number(
                observer_raw["phiKsRad"], "$.sampler.descriptor.observer.phiKsRad"
            ),
            observer_coordinate_time_m=_number(
                observer_raw["coordinateTimeM"],
                "$.sampler.descriptor.observer.coordinateTimeM",
            ),
            coarse_tolerance_multiplier=_number(
                convergence["coarseToleranceMultiplier"],
                "$.sampler.descriptor.convergence.coarseToleranceMultiplier",
            ),
            terminal_event_tolerance_m=_number(
                convergence["terminalEventToleranceM"],
                "$.sampler.descriptor.convergence.terminalEventToleranceM",
            ),
            terminal_covector_tolerance=_number(
                convergence["terminalCovectorTolerance"],
                "$.sampler.descriptor.convergence.terminalCovectorTolerance",
            ),
            disk_radius_absolute_tolerance_m=_number(
                convergence["diskRadiusAbsoluteToleranceM"],
                "$.sampler.descriptor.convergence.diskRadiusAbsoluteToleranceM",
            ),
            disk_radius_relative_tolerance=_number(
                convergence["diskRadiusRelativeTolerance"],
                "$.sampler.descriptor.convergence.diskRadiusRelativeTolerance",
            ),
            frequency_shift_relative_tolerance=_number(
                convergence["frequencyShiftRelativeTolerance"],
                "$.sampler.descriptor.convergence.frequencyShiftRelativeTolerance",
            ),
            emission_angle_absolute_tolerance=_number(
                convergence["emissionAngleAbsoluteTolerance"],
                "$.sampler.descriptor.convergence.emissionAngleAbsoluteTolerance",
            ),
            specific_intensity_absolute_tolerance=_number(
                convergence["specificIntensityAbsoluteTolerance"],
                "$.sampler.descriptor.convergence.specificIntensityAbsoluteTolerance",
            ),
            specific_intensity_relative_tolerance=_number(
                convergence["specificIntensityRelativeTolerance"],
                "$.sampler.descriptor.convergence.specificIntensityRelativeTolerance",
            ),
            escape_direction_tolerance_rad=_number(
                convergence["escapeDirectionToleranceRad"],
                "$.sampler.descriptor.convergence.escapeDirectionToleranceRad",
            ),
            frequency_null_residual_limit=_number(
                frequency_shift["nullResidualLimit"],
                "$.sampler.descriptor.frequencyShift.nullResidualLimit",
            ),
            conserved_quantity_tolerance=_number(
                frequency_shift["conservedQuantityTolerance"],
                "$.sampler.descriptor.frequencyShift.conservedQuantityTolerance",
            ),
            emitter_event_tolerance_m=_number(
                frequency_shift["emitterEventToleranceM"],
                "$.sampler.descriptor.frequencyShift.emitterEventToleranceM",
            ),
        )
    except KeyError as error:
        _fail("$.sampler.descriptor", f"missing required field {error.args[0]!r}")
    except KerrNtReplayError:
        raise
    except (ArithmeticError, TypeError, ValueError) as error:
        _fail("$.sampler.descriptor", f"cannot reconstruct sampler: {error}")

    reconstructed = sampler.descriptor()
    if canonical_json_bytes(reconstructed) != canonical_json_bytes(raw):
        _fail(
            "$.sampler.descriptor",
            "descriptor is not the exact content-complete output of the supported sampler",
        )
    return sampler


def _adaptive_options(value: Any) -> AdaptivePixelOptions:
    raw = _mapping(value, "$.adaptivePixelOptions")
    try:
        options = AdaptivePixelOptions(
            minimum_depth=_integer(raw["minimumDepth"], "$.adaptivePixelOptions"),
            maximum_depth=_integer(raw["maximumDepth"], "$.adaptivePixelOptions"),
            maximum_ray_evaluations=_integer(
                raw["maximumRayEvaluations"], "$.adaptivePixelOptions"
            ),
            radiance_absolute_tolerances=tuple(
                _number(item, "$.adaptivePixelOptions.radianceAbsoluteTolerances")
                for item in raw["radianceAbsoluteTolerances"]
            ),
            radiance_relative_tolerance=_number(
                raw["radianceRelativeTolerance"], "$.adaptivePixelOptions"
            ),
            unresolved_solid_angle_fraction_tolerance=_number(
                raw["unresolvedSolidAngleFractionTolerance"],
                "$.adaptivePixelOptions",
            ),
            weighted_log_g_tolerance=_number(
                raw["weightedLogGTolerance"], "$.adaptivePixelOptions"
            ),
            weighted_direction_tolerance_rad=_number(
                raw["weightedDirectionToleranceRad"], "$.adaptivePixelOptions"
            ),
            radiance_guard_ceilings=tuple(
                _number(item, "$.adaptivePixelOptions.radianceGuardCeilings")
                for item in raw["radianceGuardCeilings"]
            ),
            stencil_version=_string(
                raw["stencilVersion"], "$.adaptivePixelOptions.stencilVersion"
            ),
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        _fail("$.adaptivePixelOptions", f"cannot reconstruct options: {error}")
    if adaptive_pixel_options_descriptor(options) != raw:
        _fail("$.adaptivePixelOptions", "options do not round-trip exactly")
    return options


def _grid(value: Any) -> SpectralFrameGrid:
    raw = _mapping(value, "$.frame")
    bounds = _mapping(raw.get("screenBounds"), "$.frame.screenBounds")
    try:
        grid = SpectralFrameGrid(
            width_pixels=_integer(raw["widthPixels"], "$.frame.widthPixels"),
            height_pixels=_integer(raw["heightPixels"], "$.frame.heightPixels"),
            screen_x_min=_number(bounds["xMin"], "$.frame.screenBounds.xMin"),
            screen_x_max=_number(bounds["xMax"], "$.frame.screenBounds.xMax"),
            screen_y_min=_number(bounds["yMin"], "$.frame.screenBounds.yMin"),
            screen_y_max=_number(bounds["yMax"], "$.frame.screenBounds.yMax"),
            sample_indices=tuple(
                _integer(item, "$.frame.sampleIndices")
                for item in raw["sampleIndices"]
            ),
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        _fail("$.frame", f"cannot reconstruct frame grid: {error}")
    if grid.descriptor() != raw:
        _fail("$.frame", "frame does not round-trip exactly")
    return grid


def _validate_source_snapshot(
    manifest: Mapping[str, Any],
    source_root: Path,
    maximum_source_file_bytes: int,
) -> tuple[InputArtifact, ...]:
    try:
        artifacts: list[InputArtifact] = []
        for relative in PRODUCER_SOURCE_FILES:
            path = source_root / relative
            payload = _read_stable_regular(
                path,
                maximum_source_file_bytes,
                f"source:{relative.as_posix()}",
            )
            artifacts.append(
                InputArtifact(
                    f"repo-source://{relative.as_posix()}",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        current = tuple(sorted(artifacts))
        job_spec = manifest["producer"]["jobSpec"]
        declared_inputs = job_spec["inputs"]
        declared_hashes = job_spec["producerSourceHashes"]
    except (KeyError, TypeError, OSError) as error:
        _fail("$.producer.jobSpec.inputs", f"cannot audit source snapshot: {error}")
    expected_inputs = [artifact.as_dict() for artifact in current]
    if declared_inputs != expected_inputs:
        _fail(
            "$.producer.jobSpec.inputs",
            "declared producer sources do not exactly match this replay checkout",
        )
    expected_hashes = sorted({artifact.sha256 for artifact in current})
    if declared_hashes != expected_hashes:
        _fail(
            "$.producer.jobSpec.producerSourceHashes",
            "producer source hashes do not exactly match declared source artifacts",
        )
    return current


def _preflight_manifest(
    path: Path,
    limits: ReplayResourceLimits,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_regular(path, limits.maximum_manifest_bytes, "$")
    manifest = _strict_json(payload, "$")
    try:
        tiles = manifest["tiles"]
        frequencies = manifest["observerFrequencyBinsHz"]
        options = manifest["adaptivePixelOptions"]
    except KeyError as error:
        _fail("$", f"manifest lacks preflight field {error.args[0]!r}")
    if not isinstance(tiles, list) or len(tiles) > limits.maximum_tiles:
        _fail("$.tiles", f"tile count exceeds replay limit {limits.maximum_tiles}")
    if (
        not isinstance(frequencies, list)
        or len(frequencies) > limits.maximum_frequency_bins
    ):
        _fail(
            "$.observerFrequencyBinsHz",
            f"frequency count exceeds replay limit {limits.maximum_frequency_bins}",
        )
    record_count = 0
    product_bytes = 0
    tile_uris: list[str] = []
    for index, entry in enumerate(tiles):
        try:
            records = entry["recordCount"]
            byte_length = entry["payload"]["byteLength"]
            uri = entry["payload"]["uri"]
        except (KeyError, TypeError) as error:
            _fail(f"$.tiles[{index}]", f"malformed resource declaration: {error}")
        if type(records) is not int or records < 1:
            _fail(f"$.tiles[{index}].recordCount", "must be a positive integer")
        if type(byte_length) is not int or byte_length < 1:
            _fail(f"$.tiles[{index}].payload.byteLength", "must be positive")
        actual_bytes = _relative_regular_size(
            path.parent,
            uri,
            f"$.tiles[{index}].payload.uri",
        )
        if actual_bytes != byte_length:
            _fail(
                f"$.tiles[{index}].payload.byteLength",
                "declared and actual bytes disagree during resource preflight",
            )
        if actual_bytes > limits.maximum_tile_bytes:
            _fail(
                f"$.tiles[{index}].payload.byteLength",
                f"tile exceeds replay limit {limits.maximum_tile_bytes}",
            )
        tile_uris.append(uri)
        record_count += records
        product_bytes += actual_bytes
    if record_count > limits.maximum_records:
        _fail("$.tiles", f"record count exceeds replay limit {limits.maximum_records}")
    if product_bytes > limits.maximum_product_bytes:
        _fail(
            "$.tiles",
            f"declared payload bytes exceed replay limit {limits.maximum_product_bytes}",
        )
    try:
        maximum_rays = options["maximumRayEvaluations"]
        maximum_depth = options["maximumDepth"]
    except (KeyError, TypeError) as error:
        _fail("$.adaptivePixelOptions", f"malformed resource declaration: {error}")
    if type(maximum_rays) is not int or maximum_rays < 1:
        _fail("$.adaptivePixelOptions.maximumRayEvaluations", "must be positive")
    if type(maximum_depth) is not int or maximum_depth < 0:
        _fail("$.adaptivePixelOptions.maximumDepth", "must be non-negative")
    if record_count * maximum_rays > limits.maximum_total_ray_evaluations:
        _fail(
            "$.adaptivePixelOptions.maximumRayEvaluations",
            "declared total ray budget exceeds replay resource limit",
        )
    if maximum_depth > limits.maximum_adaptive_depth:
        _fail(
            "$.adaptivePixelOptions.maximumDepth",
            f"adaptive depth exceeds replay limit {limits.maximum_adaptive_depth}",
        )
    try:
        sidecar_uri = manifest["integrity"]["manifestSidecar"]
    except (KeyError, TypeError) as error:
        _fail("$.integrity", f"malformed sidecar declaration: {error}")
    _preflight_product_tree(path.parent, tile_uris, sidecar_uri)
    return manifest, payload


def _enforce_sampler_limits(
    sampler: KerrDiskRaySampler,
    limits: ReplayResourceLimits,
) -> None:
    options = sampler.fine_options
    surface = sampler.surface_options
    checks = (
        (
            options.maximum_accepted_steps,
            limits.maximum_ray_accepted_steps,
            "$.sampler.descriptor.rayOptions.fine.maximumAcceptedSteps",
        ),
        (
            options.maximum_rejected_steps,
            limits.maximum_ray_rejected_steps,
            "$.sampler.descriptor.rayOptions.fine.maximumRejectedSteps",
        ),
        (
            options.event_maximum_iterations,
            limits.maximum_ray_event_iterations,
            "$.sampler.descriptor.rayOptions.fine.eventMaximumIterations",
        ),
        (
            surface.maximum_iterations,
            limits.maximum_surface_iterations,
            "$.sampler.descriptor.surfaceOptions.fine.maximumIterations",
        ),
        (
            surface.maximum_reintegrations,
            limits.maximum_surface_reintegrations,
            "$.sampler.descriptor.surfaceOptions.fine.maximumReintegrations",
        ),
        (
            surface.subdivisions_per_segment,
            limits.maximum_surface_subdivisions_per_segment,
            "$.sampler.descriptor.surfaceOptions.fine.subdivisionsPerSegment",
        ),
    )
    for actual, maximum, path in checks:
        if actual > maximum:
            _fail(path, f"value exceeds replay limit {maximum}")
    if options.maximum_affine_length > limits.maximum_affine_length_m:
        _fail(
            "$.sampler.descriptor.rayOptions.fine.maximumAffineLength",
            f"value exceeds replay limit {limits.maximum_affine_length_m}",
        )


def _record_differences(
    actual_payload: bytes,
    expected_payload: bytes,
    layout: SpectralPixelLayout,
) -> tuple[str, ...]:
    actual = unpack_spectral_pixel(layout, actual_payload)
    expected = unpack_spectral_pixel(layout, expected_payload)
    names = tuple(
        item.name
        for item in fields(ScientificSpectralPixelRecord)
        if getattr(actual, item.name) != getattr(expected, item.name)
    )
    return names or ("binary-encoding-including-signed-zero",)


def validate_kerr_nt_replay(
    manifest_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA,
    *,
    limits: ReplayResourceLimits = DEFAULT_REPLAY_LIMITS,
    source_root: Path | str = ROOT,
) -> dict[str, Any]:
    """Authenticate, reconstruct, and byte-exactly replay one Kerr/NT frame."""

    if not isinstance(limits, ReplayResourceLimits):
        raise TypeError("limits must be ReplayResourceLimits")
    path = Path(manifest_path).absolute()
    if path.name != "manifest.json":
        _fail("$", "v1 replay input must be named 'manifest.json'")
    schema = Path(schema_path).absolute()
    schema_payload = _read_stable_regular(
        schema,
        limits.maximum_manifest_bytes,
        "$schema",
    )
    default_schema_payload = _read_stable_regular(
        Path(DEFAULT_SCHEMA).absolute(),
        limits.maximum_manifest_bytes,
        "$defaultSchema",
    )
    if schema_payload != default_schema_payload:
        _fail(
            "$schema",
            "replay requires the repository's exact strict v1 schema",
        )
    manifest, manifest_payload = _preflight_manifest(path, limits)

    # Apart from the fail-fast resource envelope above, no physics field is
    # interpreted until the existing strict schema and structural verifier
    # authenticate the complete product.
    structural_report = validate_scientific_spectral_frame(path, schema)
    if _read_stable_regular(path, limits.maximum_manifest_bytes, "$") != manifest_payload:
        _fail("$", "manifest changed after structural verification")
    if manifest.get("schema") != PRODUCT_SCHEMA:
        _fail("$.schema", f"unsupported product schema {manifest.get('schema')!r}")
    producer = manifest["producer"]
    if (
        producer["id"] != ADAPTIVE_TILE_PRODUCER_ID
        or producer["algorithmVersion"] != ADAPTIVE_TILE_ALGORITHM_VERSION
    ):
        _fail("$.producer", "unsupported adaptive tile producer identity")

    source_root_path = Path(source_root).absolute()
    if source_root_path.resolve() != ROOT.resolve():
        _fail(
            "$.producer.jobSpec.inputs",
            "source replay is restricted to the checkout that loaded this verifier",
        )
    source_before = _validate_source_snapshot(
        manifest,
        source_root_path,
        limits.maximum_source_file_bytes,
    )

    layout = SpectralPixelLayout(tuple(manifest["observerFrequencyBinsHz"]))
    if dict(layout.descriptor()) != manifest["pixelLayout"]:
        _fail("$.pixelLayout", "layout does not round-trip exactly")
    grid = _grid(manifest["frame"])
    options = _adaptive_options(manifest["adaptivePixelOptions"])
    current_backend = default_numeric_backend_descriptor()
    if manifest["runtimeNumericBackend"]["descriptor"] != current_backend:
        _fail(
            "$.runtimeNumericBackend.descriptor",
            "byte-exact replay requires the declared CPython/binary64 backend",
        )
    sampler = reconstruct_kerr_nt_sampler(manifest["sampler"]["descriptor"])
    _enforce_sampler_limits(sampler, limits)

    record_count = 0
    total_ray_samples = 0
    maximum_accepted_steps = 0
    maximum_rejected_steps = 0
    root = path.parent
    for tile_index, entry in enumerate(manifest["tiles"]):
        tile = entry["tile"]
        tile_path = f"$.tiles[{tile_index}]"
        payload = _read_relative_file(
            root,
            entry["payload"]["uri"],
            entry["payload"]["byteLength"],
            f"{tile_path}.payload.uri",
        )
        if hashlib.sha256(payload).hexdigest() != entry["payload"]["sha256"]:
            _fail(f"{tile_path}.payload.sha256", "tile changed after authentication")
        for local_index in range(entry["recordCount"]):
            local_y, local_x = divmod(local_index, tile["width"])
            x = tile["x"] + local_x
            y = tile["y"] + local_y
            x_min, x_max, y_min, y_max = grid.pixel_bounds(x, y)
            try:
                result = integrate_spectral_pixel(
                    sampler,
                    layout.observer_frequencies_hz,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    options=options,
                )
                expected = pack_adaptive_pixel(layout, result, options)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
                _fail(
                    f"{tile_path}.records[{local_index}]",
                    f"replay computation failed closed: {error}",
                )
            offset = local_index * layout.record_bytes
            actual = payload[offset : offset + layout.record_bytes]
            if actual != expected:
                differences = ", ".join(
                    _record_differences(actual, expected, layout)
                )
                _fail(
                    f"{tile_path}.records[{local_index}]",
                    "deterministic numerical replay mismatch in " + differences,
                )
            record = unpack_spectral_pixel(layout, expected)
            record_count += 1
            total_ray_samples += record.sample_count
            maximum_accepted_steps = max(
                maximum_accepted_steps, record.maximum_accepted_steps
            )
            maximum_rejected_steps = max(
                maximum_rejected_steps, record.maximum_rejected_steps
            )
        if (
            _read_relative_file(
                root,
                entry["payload"]["uri"],
                entry["payload"]["byteLength"],
                f"{tile_path}.payload.uri",
            )
            != payload
        ):
            _fail(f"{tile_path}.payload.uri", "tile changed during replay")

    source_after = _validate_source_snapshot(
        manifest,
        source_root_path,
        limits.maximum_source_file_bytes,
    )
    if source_after != source_before:
        _fail("$.producer.jobSpec.inputs", "producer source changed during replay")
    if _read_stable_regular(path, limits.maximum_manifest_bytes, "$") != manifest_payload:
        _fail("$", "manifest changed during replay")
    if (
        _read_stable_regular(
            schema,
            limits.maximum_manifest_bytes,
            "$schema",
        )
        != schema_payload
    ):
        _fail("$schema", "schema changed during replay")
    final_structural_report = validate_scientific_spectral_frame(path, schema)
    if final_structural_report != structural_report:
        _fail("$", "structural evidence changed during replay")
    source_final = _validate_source_snapshot(
        manifest,
        source_root_path,
        limits.maximum_source_file_bytes,
    )
    if source_final != source_before:
        _fail("$.producer.jobSpec.inputs", "producer source changed during replay")
    if default_numeric_backend_descriptor() != current_backend:
        _fail(
            "$.runtimeNumericBackend.descriptor",
            "numeric backend changed during replay",
        )
    if record_count != structural_report["recordCount"]:
        _fail("$.tiles", "replayed record count disagrees with structural evidence")

    return {
        "id": manifest["id"],
        "independentPhysicsOracle": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isNumericalRelativitySolver": False,
        "maximumAcceptedSteps": maximum_accepted_steps,
        "maximumRejectedSteps": maximum_rejected_steps,
        "numericBackendCurrentMatch": True,
        "physicsReplayVerified": True,
        "productBoundSourceHashesCurrentMatch": True,
        "recordCount": record_count,
        "replayScope": (
            "same-code-family deterministic numerical replay; byte-exact public "
            "spectral ABI, aggregate source coverage/masks, and diagnostics"
        ),
        "sourceTopologyReplayScope": (
            "per-ray topology is recomputed and drives adaptive integration; "
            "the product persists only aggregate source evidence"
        ),
        "status": "exact-kerr-nt-deterministic-numerical-replay-conformant",
        "structuralContractVerified": True,
        "sourceArtifactCount": len(source_before),
        "sourceHashScope": (
            "exact current bytes for every JobSpec producer input; source-closure "
            "completeness remains a producer contract"
        ),
        "tamperDetectionScope": (
            "all replayed payload bytes and closed configuration/output "
            "consistency; a fully resealed byte-equivalent alternative manifest "
            "requires an external expected hash or signature"
        ),
        "tileCount": structural_report["tileCount"],
        "totalRaySamples": total_ray_samples,
    }
