"""Converged exact-Kerr thin-disk ray samples for adaptive spectral frames."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from offline.adaptive_frame import RayConvergenceAudit, SpectralRaySample
from offline.disk_atmosphere import (
    AngularEmissionLaw,
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
    equatorial_emission_angle_cosine,
)
from offline.geodesic import (
    HamiltonianState,
    RayRefinementResult,
    RayTraceOptions,
    SurfaceEventOptions,
    trace_refined_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    KerrZamoTetrad,
    kerr_bl_zamo_tetrad,
    kerr_ks_event_to_oblate,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import (
    StationaryNovikovThorneDisk,
    observer_to_emitter_frequency_shift_g,
)
from offline.kerr_disk_early_stop import (
    KERR_DISK_OPAQUE_HIT_OUTCOME,
    KerrDiskAnnulusSurface,
    KerrDiskVisibleSpectrumResult,
    transfer_early_stopped_kerr_disk_spectrum,
)
from offline.kerr_disk_transfer import (
    EscapedObserverSpecificIntensity,
    KERR_DISK_TRANSFER_SCIENTIFIC_STATUS,
    KerrDiskCrossingSignatureEntry,
    KerrDiskSpectrumResult,
    transfer_kerr_disk_spectrum,
)


class KerrDiskFrameError(RuntimeError):
    """Raised when a fine/coarse spectral ray cannot be certified."""


class DescribedEscapedObserverSpectrum(
    EscapedObserverSpecificIntensity,
    Protocol,
):
    """Observer-frame escape spectrum with deterministic identity metadata."""

    def descriptor(self) -> Mapping[str, Any]:
        """Return finite canonical-JSON configuration data."""

        ...


@dataclass(frozen=True, slots=True)
class DarkEscapedObserverSpectrum:
    """Exactly black observer-frame escape boundary."""

    implementation_id: str = field(
        default="dark-observer-frame-escape-spectrum/v1",
        init=False,
    )

    def __call__(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
        boundary_target_id: str,
    ) -> float:
        if not isinstance(terminal_state, HamiltonianState):
            raise TypeError("terminal_state must be a HamiltonianState")
        frequency = _finite_number(
            observer_frequency_hz,
            "observer_frequency_hz",
        )
        if frequency <= 0.0:
            raise ValueError("observer_frequency_hz must be positive")
        if not isinstance(boundary_target_id, str) or not boundary_target_id:
            raise ValueError("boundary_target_id must be non-empty")
        return 0.0

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "frequencyFrame": "observer",
            "implementationId": self.implementation_id,
            "kind": "dark-boundary",
            "quantity": "spectral-specific-intensity-I_nu",
            "units": "W m^-2 sr^-1 Hz^-1",
        }


@dataclass(frozen=True, slots=True)
class PowerLawEscapedObserverSpectrum:
    """Closed analytic observer-frame background with content-complete identity."""

    reference_specific_intensity_nu: float = 1.0
    reference_frequency_hz: float = 1.0e14
    spectral_index: float = 0.0
    implementation_id: str = field(
        default="power-law-observer-frame-escape-spectrum/v1",
        init=False,
    )

    def __post_init__(self) -> None:
        intensity = _finite_number(
            self.reference_specific_intensity_nu,
            "reference_specific_intensity_nu",
        )
        frequency = _finite_number(
            self.reference_frequency_hz,
            "reference_frequency_hz",
        )
        index = _finite_number(self.spectral_index, "spectral_index")
        if intensity < 0.0:
            raise ValueError("reference specific intensity must be non-negative")
        if frequency <= 0.0:
            raise ValueError("reference frequency must be positive")
        object.__setattr__(self, "reference_specific_intensity_nu", intensity)
        object.__setattr__(self, "reference_frequency_hz", frequency)
        object.__setattr__(self, "spectral_index", index)

    def __call__(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
        boundary_target_id: str,
    ) -> float:
        if not isinstance(terminal_state, HamiltonianState):
            raise TypeError("terminal_state must be a HamiltonianState")
        frequency = _finite_number(
            observer_frequency_hz,
            "observer_frequency_hz",
        )
        if frequency <= 0.0:
            raise ValueError("observer_frequency_hz must be positive")
        if not isinstance(boundary_target_id, str) or not boundary_target_id:
            raise ValueError("boundary_target_id must be non-empty")
        if self.reference_specific_intensity_nu == 0.0:
            return 0.0
        if self.spectral_index == 0.0:
            return self.reference_specific_intensity_nu
        log_frequency_ratio = (
            math.log(frequency) - math.log(self.reference_frequency_hz)
        )
        log_intensity = (
            math.log(self.reference_specific_intensity_nu)
            + self.spectral_index * log_frequency_ratio
        )
        if math.isnan(log_intensity):
            raise KerrDiskFrameError("power-law escape spectrum is indeterminate")
        if log_intensity > math.log(sys.float_info.max):
            raise KerrDiskFrameError("power-law escape spectrum overflowed")
        if log_intensity < math.log(math.ulp(0.0)):
            return 0.0
        result = math.exp(log_intensity)
        if not math.isfinite(result) or result < 0.0:
            raise KerrDiskFrameError("power-law escape spectrum is invalid")
        return result

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "frequencyFrame": "observer",
            "implementationId": self.implementation_id,
            "kind": "analytic-power-law-boundary",
            "quantity": "spectral-specific-intensity-I_nu",
            "referenceFrequencyHz": self.reference_frequency_hz,
            "referenceSpecificIntensityNu": (
                self.reference_specific_intensity_nu
            ),
            "spectralIndex": self.spectral_index,
            "units": "W m^-2 sr^-1 Hz^-1",
        }


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _positive_frequencies(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("observer_frequencies_hz must be a sequence")
    try:
        frequencies = tuple(
            _finite_number(value, f"observer_frequencies_hz[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise ValueError("observer_frequencies_hz must be a sequence") from error
    if not frequencies or any(frequency <= 0.0 for frequency in frequencies):
        raise ValueError("observer frequencies must be non-empty and positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return frequencies


def _canonical_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} object keys must be strings")
        return {
            key: _canonical_json_value(value[key], f"{label}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical_json_value(entry, f"{label}[{index}]")
            for index, entry in enumerate(value)
        ]
    raise ValueError(f"{label} contains a non-canonical value")


def _canonical_descriptor_json(value: Any, label: str) -> str:
    canonical = _canonical_json_value(value, label)
    if not isinstance(canonical, dict):
        raise ValueError(f"{label} must be an object")
    implementation_id = canonical.get("implementationId")
    if not isinstance(implementation_id, str) or not implementation_id:
        raise ValueError(f"{label} needs a non-empty implementationId")
    return json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ray_options_descriptor(options: RayTraceOptions) -> dict[str, Any]:
    return {
        "absoluteTolerance": options.absolute_tolerance,
        "eventAffineTolerance": options.event_affine_tolerance,
        "eventMaximumIterations": options.event_maximum_iterations,
        "eventValueTolerance": options.event_value_tolerance,
        "initialStep": options.initial_step,
        "maximumAcceptedSteps": options.maximum_accepted_steps,
        "maximumAffineLength": options.maximum_affine_length,
        "maximumRejectedSteps": options.maximum_rejected_steps,
        "maximumStep": options.maximum_step,
        "metricInterpolationErrorLimit": (
            options.metric_interpolation_error_limit
        ),
        "minimumStep": options.minimum_step,
        "nullResidualLimit": options.null_residual_limit,
        "recordPath": options.record_path,
        "relativeTolerance": options.relative_tolerance,
    }


def _surface_options_descriptor(options: SurfaceEventOptions) -> dict[str, Any]:
    return {
        "absoluteTolerance": options.absolute_tolerance,
        "affineTolerance": options.affine_tolerance,
        "maximumIterations": options.maximum_iterations,
        "maximumReintegrations": options.maximum_reintegrations,
        "metricInterpolationErrorLimit": (
            options.metric_interpolation_error_limit
        ),
        "nullResidualLimit": options.null_residual_limit,
        "relativeTolerance": options.relative_tolerance,
        "subdivisionsPerSegment": options.subdivisions_per_segment,
        "surfaceValueTolerance": options.surface_value_tolerance,
    }


def _within_tolerance(
    first: float,
    second: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    limit = absolute_tolerance + relative_tolerance * max(
        abs(first),
        abs(second),
    )
    return math.isfinite(limit) and abs(first - second) <= limit


def _angular_separation(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    return math.atan2(
        math.sqrt(math.fsum(value * value for value in cross)),
        math.fsum(first[index] * second[index] for index in range(3)),
    )


def _finite_worldtube_angular_direction(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
) -> tuple[float, float, float]:
    """Return the finite-worldtube KS angular continuation direction.

    This is a coordinate-angle convergence diagnostic, not local photon
    momentum, an observer-tetrad projection, or an asymptotic ICRS direction.
    """

    oblate = kerr_ks_event_to_oblate(metric, state.event)
    sine = math.sin(oblate.theta_rad)
    direction = (
        sine * math.cos(oblate.phi_ks_rad),
        sine * math.sin(oblate.phi_ks_rad),
        math.cos(oblate.theta_rad),
    )
    norm = math.sqrt(math.fsum(value * value for value in direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise KerrDiskFrameError("finite-worldtube direction is invalid")
    return tuple(value / norm for value in direction)  # type: ignore[return-value]


def _topology_token(
    result: KerrDiskSpectrumResult | KerrDiskVisibleSpectrumResult,
) -> str:
    signature = result.crossing_signature
    payload: dict[str, Any] = {
        "crossings": [
            {
                "orientation": entry.orientation,
                "radialRegion": entry.radial_region,
            }
            for entry in signature
        ],
        "visibleSource": result.source_kind,
    }
    if result.source_kind == "disk":
        opaque_index = result.first_opaque_crossing_index
        if opaque_index is None:
            raise KerrDiskFrameError("disk topology lacks an opaque crossing")
        payload["crossings"] = payload["crossings"][: opaque_index + 1]
        payload["firstOpaqueCrossingIndex"] = opaque_index
    else:
        payload["terminal"] = {
            "outcome": result.ray_boundary_outcome,
            "targetId": result.ray_boundary_target_id,
        }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _maximum_visible_surface_bracket_width(
    result: KerrDiskSpectrumResult | KerrDiskVisibleSpectrumResult,
) -> float:
    """Return the largest bracket that can affect the visible classification."""

    widths = result.crossing_bracket_affine_widths
    if result.source_kind == "disk":
        opaque_index = result.first_opaque_crossing_index
        if opaque_index is None:
            raise KerrDiskFrameError("disk topology lacks an opaque crossing")
        widths = widths[: opaque_index + 1]
    return max(widths, default=0.0)


@dataclass(frozen=True, slots=True)
class KerrDiskRaySampler:
    """Fine/coarse certified exact-Kerr scalar ray sampler."""

    metric: KerrKerrSchildMetric
    observer_radius_m: float
    termination: KerrOblateTermination
    disk: StationaryNovikovThorneDisk
    outer_radius_m: float
    escaped_observer_spectrum: DescribedEscapedObserverSpectrum
    fine_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    angular_emission_law: AngularEmissionLaw = field(
        default_factory=FluxConservingLinearLimbDarkening,
    )
    observer_theta_rad: float = math.pi / 3.0
    observer_phi_ks_rad: float | None = None
    observer_coordinate_time_m: float = 0.0
    coarse_tolerance_multiplier: float = 32.0
    terminal_event_tolerance_m: float = 2.0e-5
    terminal_covector_tolerance: float = 2.0e-5
    disk_radius_absolute_tolerance_m: float = 0.0
    disk_radius_relative_tolerance: float = 2.0e-5
    frequency_shift_relative_tolerance: float = 2.0e-5
    emission_angle_absolute_tolerance: float = 2.0e-5
    specific_intensity_absolute_tolerance: float = 0.0
    specific_intensity_relative_tolerance: float = 2.0e-4
    escape_direction_tolerance_rad: float = 2.0e-5
    frequency_null_residual_limit: float = 1.0e-7
    conserved_quantity_tolerance: float = 1.0e-7
    emitter_event_tolerance_m: float | None = None
    _observer_tetrad: KerrZamoTetrad = field(
        init=False,
        repr=False,
        compare=False,
    )
    _resolved_observer_phi_ks_rad: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _coarse_surface_options: SurfaceEventOptions = field(
        init=False,
        repr=False,
        compare=False,
    )
    _coarse_ray_options: RayTraceOptions = field(
        init=False,
        repr=False,
        compare=False,
    )
    _resolved_emitter_event_tolerance_m: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _escaped_descriptor_json: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _escaped_descriptor_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _angular_descriptor_json: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _angular_descriptor_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _early_surface: KerrDiskAnnulusSurface = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.metric, KerrKerrSchildMetric):
            raise TypeError("metric must be an exact KerrKerrSchildMetric")
        if not isinstance(self.termination, KerrOblateTermination):
            raise TypeError("termination must be a KerrOblateTermination")
        if not isinstance(self.disk, StationaryNovikovThorneDisk):
            raise TypeError("disk must be a StationaryNovikovThorneDisk")
        if self.disk.metric != self.metric:
            raise ValueError("disk and sampler must use the same exact Kerr metric")
        if self.termination.spin_a_m != self.metric.spin_a_m:
            raise ValueError("termination and metric Kerr spin must agree")
        if not isinstance(self.fine_options, RayTraceOptions):
            raise TypeError("fine_options must be a RayTraceOptions")
        if self.fine_options.record_path is not True:
            raise ValueError("fine_options must set record_path=True")
        if not isinstance(self.surface_options, SurfaceEventOptions):
            raise TypeError("surface_options must be a SurfaceEventOptions")
        if type(self.angular_emission_law) not in (
            IsotropicAngularEmission,
            FluxConservingLinearLimbDarkening,
        ):
            raise TypeError(
                "production sampler accepts only the closed built-in angular "
                "emission laws with implementation-owned descriptors"
            )
        angular_json = _canonical_descriptor_json(
            self.angular_emission_law.descriptor(),
            "angular emission law descriptor",
        )
        if type(self.escaped_observer_spectrum) not in (
            DarkEscapedObserverSpectrum,
            PowerLawEscapedObserverSpectrum,
        ):
            raise TypeError(
                "production sampler accepts only closed built-in escaped "
                "spectra with implementation-owned descriptors"
            )
        descriptor_method = getattr(
            self.escaped_observer_spectrum,
            "descriptor",
            None,
        )
        if not callable(descriptor_method):
            raise TypeError("escaped_observer_spectrum must provide descriptor()")
        escaped_json = _canonical_descriptor_json(
            descriptor_method(),
            "escaped observer spectrum descriptor",
        )
        escaped_descriptor = json.loads(escaped_json)
        if escaped_descriptor.get("frequencyFrame") != "observer":
            raise ValueError(
                "escaped observer spectrum descriptor must declare "
                "frequencyFrame='observer'"
            )

        observer_radius = _finite_number(
            self.observer_radius_m,
            "observer_radius_m",
        )
        observer_theta = _finite_number(
            self.observer_theta_rad,
            "observer_theta_rad",
        )
        if abs(math.cos(observer_theta)) <= 1.0e-10:
            raise ValueError(
                "an exactly edge-on observer is degenerate for a zero-thickness "
                "disk; use a non-equatorial inclination or a finite-thickness model"
            )
        observer_time = _finite_number(
            self.observer_coordinate_time_m,
            "observer_coordinate_time_m",
        )
        observer_phi = self.observer_phi_ks_rad
        if observer_phi is not None:
            observer_phi = _finite_number(observer_phi, "observer_phi_ks_rad")
        outer_radius = _finite_number(self.outer_radius_m, "outer_radius_m")
        if not (
            self.termination.capture_radius_m < observer_radius
            < self.termination.escape_radius_m
        ):
            raise ValueError("observer must lie between capture and escape worldtubes")
        if outer_radius < self.disk.isco_radius_m:
            raise ValueError("outer_radius_m must be at or outside the disk ISCO")
        if outer_radius >= self.termination.escape_radius_m:
            raise ValueError("disk outer radius must lie inside the escape worldtube")
        if observer_radius <= outer_radius:
            raise ValueError("ZAMO observer must lie outside the disk outer radius")
        if self.termination.capture_radius_m < self.metric.outer_horizon_radius_m:
            raise ValueError("capture worldtube may not lie inside the Kerr horizon")
        if self.termination.capture_radius_m >= self.disk.isco_radius_m:
            raise ValueError(
                "capture worldtube must lie strictly inside the disk ISCO"
            )

        positive_names = (
            "coarse_tolerance_multiplier",
            "terminal_event_tolerance_m",
            "terminal_covector_tolerance",
            "frequency_null_residual_limit",
            "conserved_quantity_tolerance",
        )
        normalized: dict[str, float] = {}
        for name in positive_names:
            value = _finite_number(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            normalized[name] = value
        if normalized["coarse_tolerance_multiplier"] <= 1.0:
            raise ValueError("coarse_tolerance_multiplier must exceed one")
        non_negative_names = (
            "disk_radius_absolute_tolerance_m",
            "disk_radius_relative_tolerance",
            "frequency_shift_relative_tolerance",
            "emission_angle_absolute_tolerance",
            "specific_intensity_absolute_tolerance",
            "specific_intensity_relative_tolerance",
            "escape_direction_tolerance_rad",
        )
        for name in non_negative_names:
            value = _finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            normalized[name] = value
        event_tolerance = (
            1.0e-8 * self.metric.mass_m
            if self.emitter_event_tolerance_m is None
            else _finite_number(
                self.emitter_event_tolerance_m,
                "emitter_event_tolerance_m",
            )
        )
        if event_tolerance < 0.0:
            raise ValueError("emitter_event_tolerance_m must be non-negative")

        observer_tetrad = kerr_bl_zamo_tetrad(
            self.metric,
            observer_radius_m=observer_radius,
            theta_rad=observer_theta,
            phi_ks_rad=observer_phi,
            coordinate_time_m=observer_time,
        )
        resolved_observer = kerr_ks_event_to_oblate(
            self.metric,
            observer_tetrad.event,
        )
        multiplier = normalized["coarse_tolerance_multiplier"]
        coarse_step_multiplier = min(8.0, math.sqrt(multiplier))
        coarse_ray_options = replace(
            self.fine_options,
            absolute_tolerance=self.fine_options.absolute_tolerance * multiplier,
            relative_tolerance=self.fine_options.relative_tolerance * multiplier,
            initial_step=(
                self.fine_options.initial_step * coarse_step_multiplier
            ),
            maximum_step=(
                self.fine_options.maximum_step * coarse_step_multiplier
            ),
            record_path=True,
        )
        coarse_surface_options = replace(
            self.surface_options,
            absolute_tolerance=self.surface_options.absolute_tolerance * multiplier,
            relative_tolerance=self.surface_options.relative_tolerance * multiplier,
        )
        object.__setattr__(self, "observer_radius_m", observer_radius)
        object.__setattr__(self, "observer_theta_rad", observer_theta)
        object.__setattr__(self, "observer_coordinate_time_m", observer_time)
        object.__setattr__(self, "outer_radius_m", outer_radius)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_observer_tetrad", observer_tetrad)
        object.__setattr__(
            self,
            "_resolved_observer_phi_ks_rad",
            resolved_observer.phi_ks_rad,
        )
        object.__setattr__(
            self,
            "_coarse_ray_options",
            coarse_ray_options,
        )
        object.__setattr__(
            self,
            "_coarse_surface_options",
            coarse_surface_options,
        )
        object.__setattr__(
            self,
            "_resolved_emitter_event_tolerance_m",
            event_tolerance,
        )
        object.__setattr__(self, "_escaped_descriptor_json", escaped_json)
        object.__setattr__(
            self,
            "_escaped_descriptor_sha256",
            hashlib.sha256(escaped_json.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(self, "_angular_descriptor_json", angular_json)
        object.__setattr__(
            self,
            "_angular_descriptor_sha256",
            hashlib.sha256(angular_json.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(
            self,
            "_early_surface",
            KerrDiskAnnulusSurface(self.disk, outer_radius),
        )

    def _assert_escaped_descriptor_stable(self) -> None:
        descriptor_method = getattr(self.escaped_observer_spectrum, "descriptor")
        current = _canonical_descriptor_json(
            descriptor_method(),
            "escaped observer spectrum descriptor",
        )
        if current != self._escaped_descriptor_json:
            raise KerrDiskFrameError(
                "escaped observer spectrum descriptor changed after construction"
            )

    def _assert_angular_descriptor_stable(self) -> None:
        current = _canonical_descriptor_json(
            self.angular_emission_law.descriptor(),
            "angular emission law descriptor",
        )
        if current != self._angular_descriptor_json:
            raise KerrDiskFrameError(
                "angular emission law descriptor changed after construction"
            )

    def descriptor(self) -> Mapping[str, Any]:
        """Return pure finite canonical data suitable for a future JobSpec."""

        self._assert_escaped_descriptor_stable()
        self._assert_angular_descriptor_stable()
        descriptor = {
            "angularEmission": {
                "acceptedImplementations": (
                    "closed-built-in-types-only"
                ),
                "descriptor": json.loads(self._angular_descriptor_json),
                "descriptorSha256": self._angular_descriptor_sha256,
                "isSolvedAtmosphere": False,
            },
            "convergence": {
                "coarseToleranceMultiplier": self.coarse_tolerance_multiplier,
                "diskRadiusAbsoluteToleranceM": (
                    self.disk_radius_absolute_tolerance_m
                ),
                "diskRadiusRelativeTolerance": self.disk_radius_relative_tolerance,
                "escapeDirectionToleranceRad": (
                    self.escape_direction_tolerance_rad
                ),
                "frequencyShiftRelativeTolerance": (
                    self.frequency_shift_relative_tolerance
                ),
                "emissionAngleAbsoluteTolerance": (
                    self.emission_angle_absolute_tolerance
                ),
                "specificIntensityAbsoluteTolerance": (
                    self.specific_intensity_absolute_tolerance
                ),
                "specificIntensityRelativeTolerance": (
                    self.specific_intensity_relative_tolerance
                ),
                "terminalCovectorTolerance": self.terminal_covector_tolerance,
                "terminalEventToleranceM": self.terminal_event_tolerance_m,
            },
            "disk": {
                "blackHoleMassKg": self.disk.black_hole_mass_kg,
                "colourCorrection": self.disk.colour_correction,
                "iscoRadiusM": self.disk.isco_radius_m,
                "massAccretionRateKgS": self.disk.mass_accretion_rate_kg_s,
                "orientation": self.disk.orientation,
                "outerRadiusM": self.outer_radius_m,
            },
            "escapeDirectionDiagnostic": {
                "frame": "finite-worldtube-KS-angular-continuation-direction",
                "isAsymptoticICRS": False,
                "isLocalPhotonMomentum": False,
                "purpose": "fine-coarse-and-subpixel-angular-comparison-only",
            },
            "escapedObserverSpectrum": {
                "acceptedImplementations": "closed-built-in-types-only",
                "descriptor": json.loads(self._escaped_descriptor_json),
                "descriptorSha256": self._escaped_descriptor_sha256,
                "frequencyFrame": "observer",
                "samplerAppliesAdditionalG3": False,
            },
            "frequencyShift": {
                "conservedQuantityTolerance": (
                    self.conserved_quantity_tolerance
                ),
                "emitterEventToleranceM": (
                    self._resolved_emitter_event_tolerance_m
                ),
                "nullResidualLimit": self.frequency_null_residual_limit,
            },
            "implementationId": "exact-kerr-nt-spectral-ray-sampler/v2",
            "metric": {
                "massM": self.metric.mass_m,
                "singularityGuardM": self.metric.singularity_guard_m,
                "sourceId": self.metric.source_id,
                "spinAM": self.metric.spin_a_m,
                "timeDependent": self.metric.time_dependent,
            },
            "observer": {
                "coordinateTimeM": self.observer_coordinate_time_m,
                "event": list(self._observer_tetrad.event),
                "fourVelocity": list(self._observer_tetrad.four_velocity),
                "phiKsRad": self._resolved_observer_phi_ks_rad,
                "radiusM": self.observer_radius_m,
                "thetaRad": self.observer_theta_rad,
                "type": "Boyer-Lindquist-ZAMO",
            },
            "observerFrequencyFrame": "observer-ZAMO",
            "rayOptions": {
                "coarseDerived": _ray_options_descriptor(
                    self._coarse_ray_options
                ),
                "fine": _ray_options_descriptor(self.fine_options),
                "recordCoarsePath": True,
            },
            "scientificStatus": dict(KERR_DISK_TRANSFER_SCIENTIFIC_STATUS),
            "screenConvention": {
                "projection": "pinhole",
                "screenX": "ZAMO-right-negative-azimuthal",
                "screenY": "ZAMO-up-negative-polar",
                "viewForward": "ZAMO-negative-radial",
            },
            "surfaceOptions": {
                "coarseDerived": _surface_options_descriptor(
                    self._coarse_surface_options
                ),
                "diskPlaneEndpointPolicy": (
                    "ignore-ray-endpoint-z0-contacts-outside-finite-annulus"
                ),
                "fine": _surface_options_descriptor(self.surface_options),
                "visiblePrefixProbeConvergence": {
                    "baseSubdivisionsPerAcceptedStep": (
                        self.surface_options.subdivisions_per_segment
                    ),
                    "claim": "declared-resolution-N-vs-2N-topology-agreement",
                    "mathematicallySurfaceComplete": False,
                    "verificationSubdivisionsPerAcceptedStep": (
                        2 * self.surface_options.subdivisions_per_segment
                    ),
                },
            },
            "visibleConvergencePolicy": {
                "boundarySource": "full-path-terminal-and-crossing-topology",
                "diskSource": "terminate-at-first-proven-opaque-crossing",
                "diskTerminalDiagnostics": "fine-coarse-first-opaque-event-and-covector",
                "hiddenBoundaryDiagnosticsForDisk": "not-integrated-not-invented",
                "opaqueDiskOutcome": KERR_DISK_OPAQUE_HIT_OUTCOME,
                "postOpaqueGeodesicIntegrated": False,
                "secondFullPathSurfaceLocalization": False,
            },
            "termination": {
                "captureRadiusM": self.termination.capture_radius_m,
                "captureTargetId": self.termination.capture_target_id,
                "escapeRadiusM": self.termination.escape_radius_m,
                "escapeTargetId": self.termination.escape_target_id,
                "spinAM": self.termination.spin_a_m,
            },
            "version": 2,
        }
        canonical = _canonical_json_value(descriptor, "KerrDiskRaySampler descriptor")
        if not isinstance(canonical, dict):
            raise AssertionError("sampler descriptor must remain an object")
        return canonical

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        """Trace, transfer, compare, and return one certified fine ray."""

        screen_x_value = _finite_number(screen_x, "screen_x")
        screen_y_value = _finite_number(screen_y, "screen_y")
        frequencies = _positive_frequencies(observer_frequencies_hz)
        self._assert_escaped_descriptor_stable()
        self._assert_angular_descriptor_stable()
        initial = kerr_zamo_camera_ray(
            self.metric,
            observer_radius_m=self.observer_radius_m,
            screen_x=screen_x_value,
            screen_y=screen_y_value,
            theta_rad=self.observer_theta_rad,
            phi_ks_rad=self.observer_phi_ks_rad,
            coordinate_time_m=self.observer_coordinate_time_m,
        )
        if initial.event != self._observer_tetrad.event:
            raise KerrDiskFrameError("camera ray and fixed ZAMO observer disagree")
        refinement = trace_refined_null_geodesic(
            self.metric,
            initial,
            termination=self.termination,
            interior_surface=self._early_surface,
            surface_options=self.surface_options,
            fine_options=self.fine_options,
            record_coarse_path=True,
            coarse_tolerance_multiplier=self.coarse_tolerance_multiplier,
            terminal_event_tolerance=self.terminal_event_tolerance_m,
            terminal_covector_tolerance=self.terminal_covector_tolerance,
        )
        if not isinstance(refinement, RayRefinementResult):
            raise KerrDiskFrameError("geodesic refinement returned an invalid result")
        if not refinement.discretizations_differ:
            raise KerrDiskFrameError(
                "fine/coarse traces did not use distinct discretizations"
            )
        fine_has_surface_trace = refinement.fine.interior_surface_trace is not None
        coarse_has_surface_trace = refinement.coarse.interior_surface_trace is not None
        if fine_has_surface_trace != coarse_has_surface_trace:
            raise KerrDiskFrameError(
                "fine/coarse traces disagree on accepted-step surface evidence"
            )
        early_stop_mode = fine_has_surface_trace
        for label, ray in (("fine", refinement.fine), ("coarse", refinement.coarse)):
            allowed_outcomes = (
                ("captured", "escaped", KERR_DISK_OPAQUE_HIT_OUTCOME)
                if early_stop_mode
                else ("captured", "escaped")
            )
            if ray.outcome not in allowed_outcomes:
                raise KerrDiskFrameError(f"{label} ray has no visible source")
            if ray.outcome == "captured":
                expected_target = self.termination.capture_target_id
            elif ray.outcome == "escaped":
                expected_target = self.termination.escape_target_id
            else:
                expected_target = self._early_surface.opaque_target_id
            if ray.terminal_target_id != expected_target:
                raise KerrDiskFrameError(
                    f"{label} ray terminal target disagrees with the worldtube/visible surface"
                )
            if not ray.segments or ray.segments[0].start != initial:
                raise KerrDiskFrameError(
                    f"{label} trace does not start from the shared camera state"
                )

        transfer_keywords = {
            "escaped_observer_specific_intensity_nu": (
                self.escaped_observer_spectrum
            ),
            "frequency_null_residual_limit": (
                self.frequency_null_residual_limit
            ),
            "conserved_quantity_tolerance": self.conserved_quantity_tolerance,
            "emitter_event_tolerance_m": (
                self._resolved_emitter_event_tolerance_m
            ),
            "angular_emission_law": self.angular_emission_law,
        }
        if early_stop_mode:
            fine_transfer = transfer_early_stopped_kerr_disk_spectrum(
                self._early_surface,
                refinement.fine,
                self._observer_tetrad.four_velocity,
                frequencies,
                surface_options=self.surface_options,
                **transfer_keywords,
            )
        else:
            fine_transfer = transfer_kerr_disk_spectrum(
                self.disk,
                refinement.fine,
                self._observer_tetrad.four_velocity,
                frequencies,
                outer_radius_m=self.outer_radius_m,
                surface_options=self.surface_options,
                **transfer_keywords,
            )
        self._assert_escaped_descriptor_stable()
        self._assert_angular_descriptor_stable()
        if early_stop_mode:
            coarse_transfer = transfer_early_stopped_kerr_disk_spectrum(
                self._early_surface,
                refinement.coarse,
                self._observer_tetrad.four_velocity,
                frequencies,
                surface_options=self._coarse_surface_options,
                **transfer_keywords,
            )
        else:
            coarse_transfer = transfer_kerr_disk_spectrum(
                self.disk,
                refinement.coarse,
                self._observer_tetrad.four_velocity,
                frequencies,
                outer_radius_m=self.outer_radius_m,
                surface_options=self._coarse_surface_options,
                **transfer_keywords,
            )
        self._assert_escaped_descriptor_stable()
        self._assert_angular_descriptor_stable()
        expected_transfer_type = (
            KerrDiskVisibleSpectrumResult
            if early_stop_mode
            else KerrDiskSpectrumResult
        )
        if not isinstance(fine_transfer, expected_transfer_type) or not isinstance(
            coarse_transfer,
            expected_transfer_type,
        ):
            raise KerrDiskFrameError("disk transfer returned an invalid result")
        if (
            fine_transfer.observer_frequencies_hz != frequencies
            or coarse_transfer.observer_frequencies_hz != frequencies
        ):
            raise KerrDiskFrameError("disk transfer returned the wrong frequencies")
        for label, transfer, ray in (
            ("fine", fine_transfer, refinement.fine),
            ("coarse", coarse_transfer, refinement.coarse),
        ):
            if early_stop_mode and ray.outcome == KERR_DISK_OPAQUE_HIT_OUTCOME:
                if (
                    not transfer.terminated_at_opaque_disk
                    or transfer.ray_boundary_outcome is not None
                    or transfer.ray_boundary_target_id is not None
                ):
                    raise KerrDiskFrameError(
                        f"{label} transfer invents a hidden boundary behind the disk"
                    )
            elif (
                transfer.ray_boundary_outcome != ray.outcome
                or transfer.ray_boundary_target_id != ray.terminal_target_id
            ):
                raise KerrDiskFrameError(
                    f"{label} transfer disagrees with its terminal ray semantics"
                )
            if early_stop_mode:
                surface_trace = ray.interior_surface_trace
                if surface_trace is None:
                    raise KerrDiskFrameError(
                        f"{label} ray lacks accepted-step surface evidence"
                    )
                try:
                    expected_signature = tuple(
                        KerrDiskCrossingSignatureEntry(
                            entry.crossing.orientation,
                            entry.decision.classification,  # type: ignore[arg-type]
                        )
                        for entry in surface_trace.crossings
                    )
                except (TypeError, ValueError) as error:
                    raise KerrDiskFrameError(
                        f"{label} ray surface classification is invalid"
                    ) from error
                expected_widths = tuple(
                    entry.crossing.bracket_affine_width
                    for entry in surface_trace.crossings
                )
                if (
                    transfer.crossing_signature != expected_signature
                    or transfer.crossing_bracket_affine_widths
                    != expected_widths
                ):
                    raise KerrDiskFrameError(
                        f"{label} transfer crossing evidence disagrees with its ray"
                    )
        if fine_transfer.source_kind != coarse_transfer.source_kind:
            raise KerrDiskFrameError("fine/coarse visible sources disagree")
        visible_source = fine_transfer.source_kind
        if _topology_token(fine_transfer) != _topology_token(coarse_transfer):
            raise KerrDiskFrameError(
                "fine/coarse visible crossing topologies disagree"
            )
        if early_stop_mode and visible_source == "disk":
            if (
                not refinement.outcome_agrees
                or not refinement.terminal_target_agrees
                or not refinement.converged
                or refinement.fine.outcome != KERR_DISK_OPAQUE_HIT_OUTCOME
                or refinement.coarse.outcome != KERR_DISK_OPAQUE_HIT_OUTCOME
            ):
                raise KerrDiskFrameError(
                    "fine/coarse visible opaque-disk events did not converge"
                )
        if visible_source != "disk":
            if not refinement.outcome_agrees:
                raise KerrDiskFrameError(
                    "fine/coarse boundary outcomes disagree near a separatrix"
                )
            if not refinement.terminal_target_agrees:
                raise KerrDiskFrameError(
                    "fine/coarse boundary terminal targets disagree"
                )
            if not refinement.converged:
                raise KerrDiskFrameError(
                    "fine/coarse boundary geodesic did not converge"
                )
            if (
                fine_transfer.ray_boundary_outcome
                != coarse_transfer.ray_boundary_outcome
                or fine_transfer.ray_boundary_target_id
                != coarse_transfer.ray_boundary_target_id
            ):
                raise KerrDiskFrameError(
                    "fine/coarse terminal boundary semantics disagree"
                )
            for label, transfer, ray in (
                ("fine", fine_transfer, refinement.fine),
                ("coarse", coarse_transfer, refinement.coarse),
            ):
                if visible_source == "captured-boundary":
                    if any(transfer.observed_specific_intensities_nu):
                        raise KerrDiskFrameError(
                            f"{label} captured boundary is not exactly black"
                        )
                    continue
                if not early_stop_mode:
                    continue
                if ray.terminal_target_id is None:
                    raise KerrDiskFrameError(
                        f"{label} escaped ray lacks a boundary target"
                    )
                expected_escape = tuple(
                    _finite_number(
                        self.escaped_observer_spectrum(
                            ray.terminal_state,
                            frequency,
                            ray.terminal_target_id,
                        ),
                        f"{label} escaped observer intensity bin {index}",
                    )
                    for index, frequency in enumerate(frequencies)
                )
                if any(value < 0.0 for value in expected_escape):
                    raise KerrDiskFrameError(
                        f"{label} escaped observer spectrum is negative"
                    )
                if transfer.observed_specific_intensities_nu != expected_escape:
                    raise KerrDiskFrameError(
                        f"{label} escaped intensity disagrees with the bound provider"
                    )
            self._assert_escaped_descriptor_stable()
        disk_radius_difference = 0.0
        relative_g_difference = 0.0
        if visible_source == "disk":
            fine_radius = fine_transfer.disk_radius_m
            coarse_radius = coarse_transfer.disk_radius_m
            if fine_radius is None or coarse_radius is None or not _within_tolerance(
                fine_radius,
                coarse_radius,
                absolute_tolerance=self.disk_radius_absolute_tolerance_m,
                relative_tolerance=self.disk_radius_relative_tolerance,
            ):
                raise KerrDiskFrameError("fine/coarse visible disk radii disagree")
            disk_radius_difference = abs(fine_radius - coarse_radius)
            fine_shift = fine_transfer.frequency_shift_g
            coarse_shift = coarse_transfer.frequency_shift_g
            if fine_shift is None or coarse_shift is None or not _within_tolerance(
                fine_shift,
                coarse_shift,
                absolute_tolerance=0.0,
                relative_tolerance=self.frequency_shift_relative_tolerance,
            ):
                raise KerrDiskFrameError("fine/coarse disk frequency shifts disagree")
            relative_g_difference = abs(fine_shift - coarse_shift) / max(
                abs(fine_shift),
                abs(coarse_shift),
                1.0e-300,
            )
            for label, transfer, ray in (
                ("fine", fine_transfer, refinement.fine),
                ("coarse", coarse_transfer, refinement.coarse),
            ):
                if transfer.emitter_event_tolerance_m is None or not math.isclose(
                    transfer.emitter_event_tolerance_m,
                    self._resolved_emitter_event_tolerance_m,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise KerrDiskFrameError(
                        f"{label} emitter-event tolerance provenance disagrees"
                    )
                if transfer.crossing is None or transfer.emitter is None or any(
                    abs(actual - expected)
                    > self._resolved_emitter_event_tolerance_m
                    for actual, expected in zip(
                        transfer.crossing.state.event,
                        transfer.emitter.event,
                    )
                ):
                    raise KerrDiskFrameError(
                        f"{label} disk crossing exceeds the emitter-event gate"
                    )
                if (
                    transfer.emitter.kerr_mass_m != self.metric.mass_m
                    or transfer.emitter.kerr_spin_a_m != self.metric.spin_a_m
                ):
                    raise KerrDiskFrameError(
                        f"{label} disk emitter is bound to a different Kerr metric"
                    )
                if early_stop_mode:
                    surface_trace = ray.interior_surface_trace
                    if (
                        surface_trace is None
                        or not surface_trace.crossings
                        or surface_trace.crossings[-1].crossing
                        is not transfer.crossing
                        or surface_trace.crossings[-1].decision.classification
                        != "opaque-annulus"
                    ):
                        raise KerrDiskFrameError(
                            f"{label} disk crossing is not the ray's terminal "
                            "opaque surface entry"
                        )
                if early_stop_mode:
                    oblate_event = kerr_ks_event_to_oblate(
                        self.metric,
                        transfer.crossing.state.event,
                    )
                    if not math.isclose(
                        transfer.disk_radius_m,
                        oblate_event.radius_m,
                        rel_tol=2.0e-13,
                        abs_tol=0.0,
                    ):
                        raise KerrDiskFrameError(
                            f"{label} disk radius is not bound to its crossing event"
                        )
                    expected_emitter = self.disk.emitter(
                        oblate_event.radius_m,
                        phi_ks_rad=oblate_event.phi_ks_rad,
                        coordinate_time_m=oblate_event.coordinate_time_m,
                    )
                    emitter_scalars_match = all(
                        math.isclose(
                            getattr(transfer.emitter, name),
                            getattr(expected_emitter, name),
                            rel_tol=8.0e-13,
                            abs_tol=8.0e-13,
                        )
                        for name in (
                            "kerr_mass_m",
                            "kerr_spin_a_m",
                            "radius_m",
                            "radius_over_mass",
                            "phi_ks_rad",
                            "angular_velocity_inverse_m",
                            "specific_energy",
                            "specific_angular_momentum_m",
                        )
                    )
                    emitter_vectors_match = all(
                        math.isclose(
                            actual,
                            expected,
                            rel_tol=8.0e-13,
                            abs_tol=8.0e-13,
                        )
                        for actual, expected in zip(
                            (*transfer.emitter.event, *transfer.emitter.four_velocity),
                            (*expected_emitter.event, *expected_emitter.four_velocity),
                        )
                    )
                    if (
                        transfer.emitter.orientation
                        != expected_emitter.orientation
                        or not emitter_scalars_match
                        or not emitter_vectors_match
                    ):
                        raise KerrDiskFrameError(
                            f"{label} disk emitter disagrees with the bound "
                            "Novikov-Thorne disk"
                        )
                    try:
                        expected_shift = observer_to_emitter_frequency_shift_g(
                            self.metric,
                            ray.segments[0].start,
                            self._observer_tetrad.four_velocity,
                            transfer.crossing.state,
                            expected_emitter,
                            null_residual_limit=self.frequency_null_residual_limit,
                            conserved_quantity_tolerance=(
                                self.conserved_quantity_tolerance
                            ),
                            emitter_event_tolerance_m=(
                                self._resolved_emitter_event_tolerance_m
                            ),
                        )
                        expected_angle = equatorial_emission_angle_cosine(
                            self.metric,
                            transfer.crossing.state,
                            expected_emitter,
                            null_residual_limit=self.frequency_null_residual_limit,
                            emitter_event_tolerance_m=(
                                self._resolved_emitter_event_tolerance_m
                            ),
                        )
                    except (
                        ArithmeticError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ) as error:
                        raise KerrDiskFrameError(
                            f"{label} disk physical oracle failed"
                        ) from error
                    if transfer.frequency_shift_g is None or not math.isclose(
                        transfer.frequency_shift_g,
                        expected_shift,
                        rel_tol=8.0e-13,
                        abs_tol=0.0,
                    ):
                        raise KerrDiskFrameError(
                            f"{label} disk frequency shift disagrees with the bound ray"
                        )
                    if transfer.emission_angle_cosine is None or not math.isclose(
                        transfer.emission_angle_cosine,
                        expected_angle,
                        rel_tol=0.0,
                        abs_tol=8.0e-13,
                    ):
                        raise KerrDiskFrameError(
                            f"{label} emission angle disagrees with the bound ray"
                        )
            fine_angle = fine_transfer.emission_angle_cosine
            coarse_angle = coarse_transfer.emission_angle_cosine
            fine_multiplier = fine_transfer.angular_emission_multiplier
            coarse_multiplier = coarse_transfer.angular_emission_multiplier
            if (
                fine_angle is None
                or coarse_angle is None
                or fine_multiplier is None
                or coarse_multiplier is None
                or abs(fine_angle - coarse_angle)
                > self.emission_angle_absolute_tolerance
            ):
                raise KerrDiskFrameError(
                    "fine/coarse disk emission angles disagree"
                )
            for label, angle, multiplier in (
                ("fine", fine_angle, fine_multiplier),
                ("coarse", coarse_angle, coarse_multiplier),
            ):
                expected_multiplier = _finite_number(
                    self.angular_emission_law.intensity_multiplier(angle),
                    f"{label} angular emission multiplier",
                )
                if not math.isclose(
                    multiplier,
                    expected_multiplier,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise KerrDiskFrameError(
                        f"{label} angular emission provenance disagrees"
                    )
            for label, transfer, radius in (
                ("fine", fine_transfer, fine_radius),
                ("coarse", coarse_transfer, coarse_radius),
            ):
                isotropic = transfer.isotropic_emitted_specific_intensities_nu
                emitted_frequencies = transfer.emitted_frequencies_hz
                if isotropic is None or emitted_frequencies is None:
                    raise KerrDiskFrameError(
                        f"{label} disk transfer lacks isotropic emission provenance"
                    )
                for index, (actual, frequency) in enumerate(
                    zip(isotropic, emitted_frequencies)
                ):
                    expected = self.disk.emitted_specific_intensity_nu(
                        radius,
                        frequency,
                    )
                    if not math.isclose(
                        actual,
                        expected,
                        rel_tol=8.0e-13,
                        abs_tol=0.0,
                    ):
                        raise KerrDiskFrameError(
                            f"{label} isotropic emission bin {index} disagrees "
                            "with the bound Novikov-Thorne disk"
                        )
            frequency_shift = fine_shift
        else:
            if (
                fine_transfer.disk_radius_m is not None
                or coarse_transfer.disk_radius_m is not None
                or fine_transfer.frequency_shift_g is not None
                or coarse_transfer.frequency_shift_g is not None
                or fine_transfer.isotropic_emitted_specific_intensities_nu
                is not None
                or coarse_transfer.isotropic_emitted_specific_intensities_nu
                is not None
                or fine_transfer.emission_angle_cosine is not None
                or coarse_transfer.emission_angle_cosine is not None
                or fine_transfer.angular_emission_multiplier is not None
                or coarse_transfer.angular_emission_multiplier is not None
                or fine_transfer.emitter_event_tolerance_m is not None
                or coarse_transfer.emitter_event_tolerance_m is not None
            ):
                raise KerrDiskFrameError("boundary source carries disk diagnostics")
            frequency_shift = None

        if len(fine_transfer.observed_specific_intensities_nu) != len(frequencies):
            raise KerrDiskFrameError("fine transfer returned the wrong bin count")
        if len(coarse_transfer.observed_specific_intensities_nu) != len(frequencies):
            raise KerrDiskFrameError("coarse transfer returned the wrong bin count")
        errors = tuple(
            abs(fine - coarse)
            for fine, coarse in zip(
                fine_transfer.observed_specific_intensities_nu,
                coarse_transfer.observed_specific_intensities_nu,
            )
        )
        for index, (fine, coarse, error) in enumerate(
            zip(
                fine_transfer.observed_specific_intensities_nu,
                coarse_transfer.observed_specific_intensities_nu,
                errors,
            )
        ):
            if not _within_tolerance(
                fine,
                coarse,
                absolute_tolerance=self.specific_intensity_absolute_tolerance,
                relative_tolerance=self.specific_intensity_relative_tolerance,
            ):
                raise KerrDiskFrameError(
                    f"fine/coarse specific intensity bin {index} disagrees"
                )

        escape_direction = None
        if visible_source == "escaped-boundary":
            fine_direction = _finite_worldtube_angular_direction(
                self.metric,
                refinement.fine.terminal_state,
            )
            coarse_direction = _finite_worldtube_angular_direction(
                self.metric,
                refinement.coarse.terminal_state,
            )
            separation = _angular_separation(fine_direction, coarse_direction)
            if (
                not math.isfinite(separation)
                or separation > self.escape_direction_tolerance_rad
            ):
                raise KerrDiskFrameError(
                    "fine/coarse finite-worldtube escape directions disagree"
                )
            escape_direction = fine_direction

        terminal_event_difference = (
            refinement.terminal_event_difference
            if early_stop_mode or visible_source != "disk"
            else 0.0
        )
        terminal_covector_difference = (
            refinement.terminal_covector_difference
            if early_stop_mode or visible_source != "disk"
            else 0.0
        )
        audit = RayConvergenceAudit(
            maximum_null_residual=max(
                refinement.fine.maximum_null_residual,
                refinement.coarse.maximum_null_residual,
            ),
            maximum_metric_interpolation_error=max(
                refinement.fine.maximum_metric_interpolation_error,
                refinement.coarse.maximum_metric_interpolation_error,
            ),
            terminal_event_difference_m=terminal_event_difference,
            terminal_covector_relative_difference=terminal_covector_difference,
            disk_radius_difference_m=disk_radius_difference,
            relative_g_difference=relative_g_difference,
            surface_bracket_affine_width=max(
                _maximum_visible_surface_bracket_width(fine_transfer),
                _maximum_visible_surface_bracket_width(coarse_transfer),
            ),
            accepted_steps=max(
                refinement.fine.accepted_steps,
                refinement.coarse.accepted_steps,
            ),
            rejected_steps=max(
                refinement.fine.rejected_steps,
                refinement.coarse.rejected_steps,
            ),
            ray_gate_passed=True,
            source_gate_passed=True,
            transfer_gate_passed=True,
        )

        return SpectralRaySample(
            specific_intensities_nu=(
                fine_transfer.observed_specific_intensities_nu
            ),
            absolute_errors_nu=errors,
            visible_source=visible_source,
            topology_signature=_topology_token(fine_transfer),
            frequency_shift_g=frequency_shift,
            escape_direction=escape_direction,
            ray_converged=True,
            convergence_audit=audit,
        )


__all__ = (
    "DarkEscapedObserverSpectrum",
    "DescribedEscapedObserverSpectrum",
    "KerrDiskFrameError",
    "KerrDiskRaySampler",
    "PowerLawEscapedObserverSpectrum",
)
