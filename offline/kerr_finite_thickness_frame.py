"""Fine/coarse certified scalar rays for the finite-thickness Kerr model.

The sampler in this module is the whole-ray convergence boundary missing from
``kerr_finite_thickness_transfer``.  Every screen sample launches two
independent exact-Kerr integrations, runs the accepted-step upper/lower
multi-surface detector on both paths, and transfers both certified terminal
prefixes.  A sample is returned only when the visible source, complete
crossing topology, terminal state, emitting face, pseudo-cylindrical radius,
frequency shift, signed emission cosine, and every spectral bin agree inside
declared tolerances.

The emitting surface remains the prescribed stationary Zhou et al. finite
height, while the thermal spectrum remains an equatorial Novikov--Thorne
proxy evaluated at matching pseudo-cylindrical radius.  This is not a solved
vertical structure, returning-radiation transport, an atmosphere, GRMHD, or
a complete GRRT image calculation.  The finite independent-ray stencil used
by a later frame integrator is also not a Sachs/Jacobi ray bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from offline.adaptive_frame import RayConvergenceAudit, SpectralRaySample
from offline.geodesic import (
    HamiltonianState,
    MultiInteriorSurfaceTrace,
    RayRefinementResult,
    RayTraceOptions,
    RayTraceResult,
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
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_finite_thickness import (
    EDGE_ON_COSINE_NUMERICAL_GUARD,
    StationaryKerrFiniteThicknessCalibration,
    UPPER,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    OPAQUE_OUTCOME,
    UPPER_SURFACE_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_finite_thickness_transfer import (
    BuiltInEscapedObserverSpectrum,
    KerrFiniteThicknessSpectrumResult,
    _transfer_kerr_finite_thickness_spectrum_certified,
)


IMPLEMENTATION_ID: Final = "kerr-finite-thickness-spectral-ray-sampler/v1"

_MAXIMUM_NULL_RESIDUAL_LIMIT: Final = 1.0e-6
_MAXIMUM_CONSERVED_QUANTITY_TOLERANCE: Final = 1.0e-6
_MAXIMUM_SURFACE_VALUE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE: Final = 1.0e-7
_MAXIMUM_BOUNDARY_VALUE_TOLERANCE_OVER_MASS: Final = 1.0e-7
_MAXIMUM_EMITTER_EVENT_TOLERANCE_OVER_MASS: Final = 1.0e-7
_MAXIMUM_COARSE_TOLERANCE_MULTIPLIER: Final = 64.0
_MAXIMUM_TERMINAL_EVENT_TOLERANCE_SCALE: Final = 2.0e-4
_MAXIMUM_TERMINAL_COVECTOR_TOLERANCE: Final = 2.0e-4
_MAXIMUM_DISK_RADIUS_ABSOLUTE_TOLERANCE_SCALE: Final = 1.0e-4
_MAXIMUM_DISK_RADIUS_RELATIVE_TOLERANCE: Final = 1.0e-4
_MAXIMUM_FREQUENCY_SHIFT_RELATIVE_TOLERANCE: Final = 1.0e-4
_MAXIMUM_EMISSION_COSINE_ABSOLUTE_TOLERANCE: Final = 1.0e-4
_MAXIMUM_SPECIFIC_INTENSITY_ABSOLUTE_TOLERANCE: Final = 0.0
_MAXIMUM_SPECIFIC_INTENSITY_RELATIVE_TOLERANCE: Final = 1.0e-3
_MAXIMUM_ESCAPE_DIRECTION_TOLERANCE_RAD: Final = 1.0e-4
_MAXIMUM_FINE_LOCAL_ABSOLUTE_TOLERANCE: Final = 1.0e-8
_MAXIMUM_FINE_LOCAL_RELATIVE_TOLERANCE: Final = 1.0e-8
_MAXIMUM_FINE_STEP_OVER_MASS: Final = 1.0
_MAXIMUM_FINE_EVENT_VALUE_TOLERANCE_OVER_MASS: Final = 1.0e-7
_MAXIMUM_FINE_EVENT_AFFINE_TOLERANCE_OVER_MASS: Final = 1.0e-8
_MAXIMUM_SURFACE_LOCAL_ABSOLUTE_TOLERANCE: Final = 1.0e-8
_MAXIMUM_SURFACE_LOCAL_RELATIVE_TOLERANCE: Final = 1.0e-8
_MAXIMUM_SURFACE_AFFINE_TOLERANCE_OVER_MASS: Final = 1.0e-8
_MAXIMUM_METRIC_INTERPOLATION_ERROR_LIMIT: Final = 1.0e-7

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "independently fine/coarse converged exact-Kerr scalar ray for a "
            "stationary phenomenological finite-height photosphere"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "spacetime": "exact stationary Kerr in Cartesian Kerr-Schild coordinates",
        "surface": "Zhou prescribed stationary finite-height photosphere",
        "thermalReference": (
            "equatorial Novikov-Thorne/Page-Thorne spectrum at matching "
            "pseudo-cylindrical radius"
        ),
        "heightFluxRateBinding": (
            "dimensionless height calibration rate and SI thermal disk rate "
            "are independently caller-supplied and are not silently equated"
        ),
        "fineCoarseWholeRayConvergence": True,
        "includesFineCoarseWholeRayConvergence": True,
        "multiSurfaceTopologyCompared": True,
        "signedFaceEmissionCosineCompared": True,
        "captureBoundary": "exactly black",
        "escapeBoundary": "closed built-in observer-frame spectrum",
        "observerMaterialPolicy": (
            "observer may lie over the physical radial annulus only when "
            "strictly outside both photosphere faces"
        ),
        "isHydrostaticVerticalStructureSolution": False,
        "isOffEquatorialGeodesicDisk": False,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isCompleteGeneralRelativisticRadiativeTransfer": False,
        "isSachsJacobiRayBundle": False,
        "prohibitedClaim": (
            "Do not describe this prescribed finite surface plus equatorial-NT "
            "proxy as hydrostatic structure, returning radiation, a solved "
            "atmosphere, GRMHD, complete GRRT, or a Sachs/Jacobi ray bundle."
        ),
    }
)


class KerrFiniteThicknessFrameError(RuntimeError):
    """Raised when a finite-thickness fine/coarse sample fails closed."""


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
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be non-empty and positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return frequencies


def _within_tolerance(
    first: float,
    second: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    limit = absolute_tolerance + relative_tolerance * max(abs(first), abs(second))
    return math.isfinite(limit) and abs(first - second) <= limit


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
        "metricInterpolationErrorLimit": options.metric_interpolation_error_limit,
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
        "metricInterpolationErrorLimit": options.metric_interpolation_error_limit,
        "nullResidualLimit": options.null_residual_limit,
        "relativeTolerance": options.relative_tolerance,
        "subdivisionsPerSegment": options.subdivisions_per_segment,
        "surfaceValueTolerance": options.surface_value_tolerance,
    }


def _canonical_descriptor(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite canonical JSON data") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object")
    implementation_id = decoded.get("implementationId")
    if not isinstance(implementation_id, str) or not implementation_id:
        raise ValueError(f"{label} needs a non-empty implementationId")
    return decoded


def _event_difference(first: HamiltonianState, second: HamiltonianState) -> float:
    return math.sqrt(
        math.fsum((left - right) ** 2 for left, right in zip(first.event, second.event))
    )


def _covector_relative_difference(
    first: HamiltonianState,
    second: HamiltonianState,
) -> float:
    scale = max(
        math.sqrt(math.fsum(value * value for value in first.covector)),
        math.sqrt(math.fsum(value * value for value in second.covector)),
        1.0e-300,
    )
    return math.sqrt(
        math.fsum(
            (left - right) ** 2
            for left, right in zip(first.covector, second.covector)
        )
    ) / scale


def _finite_worldtube_direction(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
) -> tuple[float, float, float]:
    oblate = kerr_ks_event_to_oblate(metric, state.event)
    sine = math.sin(oblate.theta_rad)
    direction = (
        sine * math.cos(oblate.phi_ks_rad),
        sine * math.sin(oblate.phi_ks_rad),
        math.cos(oblate.theta_rad),
    )
    norm = math.sqrt(math.fsum(value * value for value in direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise KerrFiniteThicknessFrameError(
            "finite-worldtube escape direction is invalid"
        )
    return tuple(value / norm for value in direction)  # type: ignore[return-value]


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


def _trace_topology_payload(ray: RayTraceResult) -> dict[str, Any]:
    trace = ray.multi_surface_trace
    if type(trace) is not MultiInteriorSurfaceTrace:
        raise KerrFiniteThicknessFrameError("ray lacks an exact multi-surface trace")
    return {
        "crossings": [
            {
                "classification": entry.decision.classification,
                "orientation": entry.crossing.orientation,
                "outcome": entry.decision.outcome,
                "surfaceId": entry.surface_id,
                "targetId": entry.decision.target_id,
                "terminates": entry.decision.terminates,
            }
            for entry in trace.crossings
        ],
        "surfaceIds": list(trace.surface_ids),
        "terminal": {
            "outcome": ray.outcome,
            "targetId": ray.terminal_target_id,
        },
    }


def _topology_token(ray: RayTraceResult) -> str:
    return json.dumps(
        _trace_topology_payload(ray),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _maximum_surface_bracket(ray: RayTraceResult) -> float:
    trace = ray.multi_surface_trace
    if type(trace) is not MultiInteriorSurfaceTrace:
        raise KerrFiniteThicknessFrameError("ray lacks an exact multi-surface trace")
    return max(
        (entry.crossing.bracket_affine_width for entry in trace.crossings),
        default=0.0,
    )


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessRaySampler:
    """Independently fine/coarse certified finite-thickness scalar sampler."""

    metric: KerrKerrSchildMetric
    observer_radius_m: float
    termination: KerrOblateTermination
    surface: KerrFiniteThicknessMultiSurface
    disk: StationaryNovikovThorneDisk
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum
    fine_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    observer_theta_rad: float = math.pi / 3.0
    observer_phi_ks_rad: float | None = None
    observer_coordinate_time_m: float = 0.0
    coarse_tolerance_multiplier: float = 32.0
    terminal_event_tolerance_m: float = 2.0e-5
    terminal_covector_tolerance: float = 2.0e-5
    disk_radius_absolute_tolerance_m: float = 0.0
    disk_radius_relative_tolerance: float = 2.0e-5
    frequency_shift_relative_tolerance: float = 2.0e-5
    emission_cosine_absolute_tolerance: float = 2.0e-5
    specific_intensity_absolute_tolerance: float = 0.0
    specific_intensity_relative_tolerance: float = 2.0e-4
    escape_direction_tolerance_rad: float = 2.0e-5
    frequency_null_residual_limit: float = 2.0e-7
    conserved_quantity_tolerance: float = 2.0e-7
    recorded_path_absolute_tolerance: float = 2.0e-10
    recorded_path_relative_tolerance: float = 2.0e-10
    boundary_value_tolerance_m: float | None = None
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
    _coarse_ray_options: RayTraceOptions = field(
        init=False,
        repr=False,
        compare=False,
    )
    _coarse_surface_options: SurfaceEventOptions = field(
        init=False,
        repr=False,
        compare=False,
    )
    _resolved_emitter_event_tolerance_m: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _resolved_boundary_value_tolerance_m: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _maximum_photosphere_oblate_radius_m: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _observer_pseudo_cylindrical_radius_over_mass: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _observer_upper_surface_value: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _observer_lower_surface_value: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _observer_within_physical_annulus: bool = field(
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

    def __post_init__(self) -> None:
        if type(self.metric) is not KerrKerrSchildMetric:
            raise TypeError("metric must be the exact KerrKerrSchildMetric")
        if type(self.termination) is not KerrOblateTermination:
            raise TypeError("termination must be the exact KerrOblateTermination")
        if type(self.surface) is not KerrFiniteThicknessMultiSurface:
            raise TypeError(
                "surface must be the exact KerrFiniteThicknessMultiSurface"
            )
        if type(self.surface.calibration) is not (
            StationaryKerrFiniteThicknessCalibration
        ):
            raise TypeError(
                "surface calibration must be the exact "
                "StationaryKerrFiniteThicknessCalibration"
            )
        if type(self.disk) is not StationaryNovikovThorneDisk:
            raise TypeError("disk must be the exact StationaryNovikovThorneDisk")
        if self.surface.metric is not self.metric or self.disk.metric is not self.metric:
            raise ValueError("metric, surface, and disk must share one exact metric")
        if self.surface.calibration.orientation != self.disk.orientation:
            raise ValueError("surface calibration and disk orientations disagree")
        if self.termination.spin_a_m != self.metric.spin_a_m:
            raise ValueError("termination and metric Kerr spins disagree")
        if type(self.fine_options) is not RayTraceOptions:
            raise TypeError("fine_options must be the exact RayTraceOptions")
        if self.fine_options.record_path is not True:
            raise ValueError("fine_options must set record_path=True")
        if type(self.surface_options) is not SurfaceEventOptions:
            raise TypeError("surface_options must be the exact SurfaceEventOptions")
        if type(self.escaped_observer_spectrum) not in (
            DarkEscapedObserverSpectrum,
            PowerLawEscapedObserverSpectrum,
        ):
            raise TypeError("escaped spectrum must be an exact closed built-in type")

        escape_descriptor = _canonical_descriptor(
            self.escaped_observer_spectrum.descriptor(),
            "escaped observer spectrum descriptor",
        )
        if (
            escape_descriptor.get("frequencyFrame") != "observer"
            or escape_descriptor.get("quantity")
            != "spectral-specific-intensity-I_nu"
        ):
            raise ValueError("escaped spectrum must be closed in observer-frame I_nu")
        escape_json = json.dumps(
            escape_descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

        observer_radius = _finite_number(self.observer_radius_m, "observer_radius_m")
        observer_theta = _finite_number(self.observer_theta_rad, "observer_theta_rad")
        observer_time = _finite_number(
            self.observer_coordinate_time_m,
            "observer_coordinate_time_m",
        )
        if observer_theta <= 0.0 or observer_theta >= math.pi:
            raise ValueError("observer_theta_rad must lie strictly inside (0, pi)")
        if abs(math.cos(observer_theta)) <= EDGE_ON_COSINE_NUMERICAL_GUARD:
            raise ValueError("edge-on finite-thickness viewing is not certified")
        observer_phi = self.observer_phi_ks_rad
        if observer_phi is not None:
            observer_phi = _finite_number(observer_phi, "observer_phi_ks_rad")

        maximum_photosphere_oblate_radius = (
            self.surface.calibration.photosphere_point(
                self.surface.calibration.outer_radius_over_mass,
                UPPER,
            ).radius_over_mass
            * self.metric.mass_m
        )
        inner_radius = (
            self.surface.calibration.isco_radius_over_mass * self.metric.mass_m
        )
        if not math.isclose(
            inner_radius,
            self.disk.isco_radius_m,
            rel_tol=2.0e-13,
            abs_tol=0.0,
        ):
            raise ValueError("surface and disk ISCO radii disagree")
        if not (
            self.termination.capture_radius_m
            < observer_radius
            < self.termination.escape_radius_m
        ):
            raise ValueError("observer must lie between capture and escape worldtubes")
        if maximum_photosphere_oblate_radius >= self.termination.escape_radius_m:
            raise ValueError(
                "maximum finite-photosphere oblate radius must lie strictly "
                "inside the escape worldtube"
            )
        horizon_radius = self.metric.outer_horizon_radius_m
        if self.termination.capture_radius_m < horizon_radius:
            raise ValueError("capture worldtube may not lie inside the Kerr horizon")
        if self.termination.capture_target_id == "analytic-kerr-event-horizon":
            if self.termination.capture_radius_m.hex() != horizon_radius.hex():
                raise ValueError(
                    "event-horizon capture target must use the exact Kerr "
                    "outer-horizon radius"
                )
        elif (
            self.termination.capture_target_id
            == "analytic-kerr-stretched-horizon"
        ):
            if self.termination.capture_radius_m <= horizon_radius:
                raise ValueError(
                    "stretched-horizon capture target must lie strictly "
                    "outside the Kerr horizon"
                )
        else:
            raise ValueError("unsupported finite-thickness capture target id")
        if (
            self.termination.escape_target_id
            != "analytic-kerr-escape-worldtube"
        ):
            raise ValueError("unsupported finite-thickness escape target id")
        if self.termination.capture_radius_m >= inner_radius:
            raise ValueError("capture worldtube must lie strictly inside the ISCO")

        normalized: dict[str, float] = {}
        for name in (
            "coarse_tolerance_multiplier",
            "terminal_event_tolerance_m",
            "terminal_covector_tolerance",
            "frequency_null_residual_limit",
            "conserved_quantity_tolerance",
            "recorded_path_absolute_tolerance",
            "recorded_path_relative_tolerance",
        ):
            value = _finite_number(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            normalized[name] = value
        if normalized["coarse_tolerance_multiplier"] <= 1.0:
            raise ValueError("coarse_tolerance_multiplier must exceed one")
        for name in (
            "disk_radius_absolute_tolerance_m",
            "disk_radius_relative_tolerance",
            "frequency_shift_relative_tolerance",
            "emission_cosine_absolute_tolerance",
            "specific_intensity_absolute_tolerance",
            "specific_intensity_relative_tolerance",
            "escape_direction_tolerance_rad",
        ):
            value = _finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            normalized[name] = value
        frame_convergence_actual = {
            "coarseToleranceMultiplier": normalized[
                "coarse_tolerance_multiplier"
            ],
            "diskRadiusAbsoluteToleranceM": normalized[
                "disk_radius_absolute_tolerance_m"
            ],
            "diskRadiusRelativeTolerance": normalized[
                "disk_radius_relative_tolerance"
            ],
            "emissionCosineAbsoluteTolerance": normalized[
                "emission_cosine_absolute_tolerance"
            ],
            "escapeDirectionToleranceRad": normalized[
                "escape_direction_tolerance_rad"
            ],
            "frequencyShiftRelativeTolerance": normalized[
                "frequency_shift_relative_tolerance"
            ],
            "specificIntensityAbsoluteTolerance": normalized[
                "specific_intensity_absolute_tolerance"
            ],
            "specificIntensityRelativeTolerance": normalized[
                "specific_intensity_relative_tolerance"
            ],
            "terminalCovectorTolerance": normalized[
                "terminal_covector_tolerance"
            ],
            "terminalEventToleranceM": normalized[
                "terminal_event_tolerance_m"
            ],
        }
        frame_convergence_maxima = {
            "coarseToleranceMultiplier": _MAXIMUM_COARSE_TOLERANCE_MULTIPLIER,
            "diskRadiusAbsoluteToleranceM": (
                _MAXIMUM_DISK_RADIUS_ABSOLUTE_TOLERANCE_SCALE
                * max(1.0, self.metric.mass_m)
            ),
            "diskRadiusRelativeTolerance": (
                _MAXIMUM_DISK_RADIUS_RELATIVE_TOLERANCE
            ),
            "emissionCosineAbsoluteTolerance": (
                _MAXIMUM_EMISSION_COSINE_ABSOLUTE_TOLERANCE
            ),
            "escapeDirectionToleranceRad": (
                _MAXIMUM_ESCAPE_DIRECTION_TOLERANCE_RAD
            ),
            "frequencyShiftRelativeTolerance": (
                _MAXIMUM_FREQUENCY_SHIFT_RELATIVE_TOLERANCE
            ),
            "specificIntensityAbsoluteTolerance": (
                _MAXIMUM_SPECIFIC_INTENSITY_ABSOLUTE_TOLERANCE
            ),
            "specificIntensityRelativeTolerance": (
                _MAXIMUM_SPECIFIC_INTENSITY_RELATIVE_TOLERANCE
            ),
            "terminalCovectorTolerance": (
                _MAXIMUM_TERMINAL_COVECTOR_TOLERANCE
            ),
            "terminalEventToleranceM": (
                _MAXIMUM_TERMINAL_EVENT_TOLERANCE_SCALE
                * max(1.0, self.metric.mass_m)
            ),
        }
        for name, maximum in frame_convergence_maxima.items():
            if frame_convergence_actual[name] > maximum:
                raise ValueError(
                    f"{name} exceeds the finite-thickness frame policy maximum"
                )

        trace_accuracy_actual = {
            "fineAbsoluteTolerance": self.fine_options.absolute_tolerance,
            "fineEventAffineTolerance": (
                self.fine_options.event_affine_tolerance
            ),
            "fineEventValueTolerance": self.fine_options.event_value_tolerance,
            "fineMaximumStep": self.fine_options.maximum_step,
            "fineMetricInterpolationErrorLimit": (
                self.fine_options.metric_interpolation_error_limit
            ),
            "fineNullResidualLimit": self.fine_options.null_residual_limit,
            "fineRelativeTolerance": self.fine_options.relative_tolerance,
            "surfaceAbsoluteTolerance": self.surface_options.absolute_tolerance,
            "surfaceAffineTolerance": self.surface_options.affine_tolerance,
            "surfaceMetricInterpolationErrorLimit": (
                self.surface_options.metric_interpolation_error_limit
            ),
            "surfaceNullResidualLimit": self.surface_options.null_residual_limit,
            "surfaceRelativeTolerance": self.surface_options.relative_tolerance,
        }
        trace_accuracy_maxima = {
            "fineAbsoluteTolerance": _MAXIMUM_FINE_LOCAL_ABSOLUTE_TOLERANCE,
            "fineEventAffineTolerance": (
                _MAXIMUM_FINE_EVENT_AFFINE_TOLERANCE_OVER_MASS
                * max(1.0, self.metric.mass_m)
            ),
            "fineEventValueTolerance": (
                _MAXIMUM_FINE_EVENT_VALUE_TOLERANCE_OVER_MASS
                * self.metric.mass_m
            ),
            "fineMaximumStep": (
                _MAXIMUM_FINE_STEP_OVER_MASS * self.metric.mass_m
            ),
            "fineMetricInterpolationErrorLimit": (
                _MAXIMUM_METRIC_INTERPOLATION_ERROR_LIMIT
            ),
            "fineNullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
            "fineRelativeTolerance": _MAXIMUM_FINE_LOCAL_RELATIVE_TOLERANCE,
            "surfaceAbsoluteTolerance": (
                _MAXIMUM_SURFACE_LOCAL_ABSOLUTE_TOLERANCE
            ),
            "surfaceAffineTolerance": (
                _MAXIMUM_SURFACE_AFFINE_TOLERANCE_OVER_MASS
                * max(1.0, self.metric.mass_m)
            ),
            "surfaceMetricInterpolationErrorLimit": (
                _MAXIMUM_METRIC_INTERPOLATION_ERROR_LIMIT
            ),
            "surfaceNullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
            "surfaceRelativeTolerance": (
                _MAXIMUM_SURFACE_LOCAL_RELATIVE_TOLERANCE
            ),
        }
        for name, maximum in trace_accuracy_maxima.items():
            if trace_accuracy_actual[name] > maximum:
                raise ValueError(
                    f"{name} exceeds the finite-thickness trace policy maximum"
                )
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
        boundary_tolerance = (
            self.fine_options.event_value_tolerance
            if self.boundary_value_tolerance_m is None
            else _finite_number(
                self.boundary_value_tolerance_m,
                "boundary_value_tolerance_m",
            )
        )
        if boundary_tolerance <= 0.0:
            raise ValueError("boundary_value_tolerance_m must be positive")
        tolerance_actual = {
            "boundaryValueToleranceM": boundary_tolerance,
            "conservedQuantityTolerance": normalized[
                "conserved_quantity_tolerance"
            ],
            "emitterEventToleranceM": event_tolerance,
            "nullResidualLimit": normalized["frequency_null_residual_limit"],
            "recordedPathAbsoluteTolerance": normalized[
                "recorded_path_absolute_tolerance"
            ],
            "recordedPathRelativeTolerance": normalized[
                "recorded_path_relative_tolerance"
            ],
            "surfaceValueTolerance": self.surface_options.surface_value_tolerance,
        }
        tolerance_maxima = {
            "boundaryValueToleranceM": (
                _MAXIMUM_BOUNDARY_VALUE_TOLERANCE_OVER_MASS
                * self.metric.mass_m
            ),
            "conservedQuantityTolerance": (
                _MAXIMUM_CONSERVED_QUANTITY_TOLERANCE
            ),
            "emitterEventToleranceM": (
                _MAXIMUM_EMITTER_EVENT_TOLERANCE_OVER_MASS * self.metric.mass_m
            ),
            "nullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
            "recordedPathAbsoluteTolerance": (
                _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
                * max(1.0, self.metric.mass_m)
            ),
            "recordedPathRelativeTolerance": (
                _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
            ),
            "surfaceValueTolerance": _MAXIMUM_SURFACE_VALUE_TOLERANCE,
        }
        for name, maximum in tolerance_maxima.items():
            if tolerance_actual[name] > maximum:
                raise ValueError(
                    f"{name} exceeds the finite-thickness transfer policy maximum"
                )

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
        observer_rho = (
            resolved_observer.radius_m
            * math.sin(resolved_observer.theta_rad)
            / self.metric.mass_m
        )
        observer_surface_state = HamiltonianState(
            observer_tetrad.event,
            (0.0, 0.0, 0.0, 0.0),
        )
        observer_upper_value = self.surface.value(
            UPPER_SURFACE_ID,
            observer_surface_state,
        )
        observer_lower_value = self.surface.value(
            LOWER_SURFACE_ID,
            observer_surface_state,
        )
        observer_within_annulus = (
            self.surface.calibration.contains_pseudo_cylindrical_radius(
                observer_rho
            )
        )
        if (
            observer_within_annulus
            and observer_upper_value
            <= self.surface_options.surface_value_tolerance
            and observer_lower_value
            <= self.surface_options.surface_value_tolerance
        ):
            raise ValueError(
                "observer lies on or inside the physical finite-thickness "
                "photosphere"
            )
        multiplier = normalized["coarse_tolerance_multiplier"]
        step_multiplier = min(8.0, math.sqrt(multiplier))
        coarse_ray_options = replace(
            self.fine_options,
            absolute_tolerance=self.fine_options.absolute_tolerance * multiplier,
            relative_tolerance=self.fine_options.relative_tolerance * multiplier,
            initial_step=self.fine_options.initial_step * step_multiplier,
            maximum_step=self.fine_options.maximum_step * step_multiplier,
            record_path=True,
        )
        coarse_surface_options = replace(
            self.surface_options,
            absolute_tolerance=self.surface_options.absolute_tolerance * multiplier,
            relative_tolerance=self.surface_options.relative_tolerance * multiplier,
        )
        resolved_path_absolute_tolerances = (
            max(
                normalized["recorded_path_absolute_tolerance"],
                self.fine_options.absolute_tolerance,
            ),
            max(
                normalized["recorded_path_absolute_tolerance"],
                coarse_ray_options.absolute_tolerance,
            ),
        )
        resolved_path_relative_tolerances = (
            max(
                normalized["recorded_path_relative_tolerance"],
                self.fine_options.relative_tolerance,
            ),
            max(
                normalized["recorded_path_relative_tolerance"],
                coarse_ray_options.relative_tolerance,
            ),
        )
        if max(resolved_path_absolute_tolerances) > (
            _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
            * max(1.0, self.metric.mass_m)
        ):
            raise ValueError(
                "resolved fine/coarse recorded-path absolute tolerance exceeds "
                "the transfer policy maximum"
            )
        if max(resolved_path_relative_tolerances) > (
            _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                "resolved fine/coarse recorded-path relative tolerance exceeds "
                "the transfer policy maximum"
            )

        object.__setattr__(self, "observer_radius_m", observer_radius)
        object.__setattr__(self, "observer_theta_rad", observer_theta)
        object.__setattr__(self, "observer_coordinate_time_m", observer_time)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_observer_tetrad", observer_tetrad)
        object.__setattr__(
            self,
            "_resolved_observer_phi_ks_rad",
            resolved_observer.phi_ks_rad,
        )
        object.__setattr__(self, "_coarse_ray_options", coarse_ray_options)
        object.__setattr__(self, "_coarse_surface_options", coarse_surface_options)
        object.__setattr__(
            self,
            "_resolved_emitter_event_tolerance_m",
            event_tolerance,
        )
        object.__setattr__(
            self,
            "_resolved_boundary_value_tolerance_m",
            boundary_tolerance,
        )
        object.__setattr__(
            self,
            "_maximum_photosphere_oblate_radius_m",
            maximum_photosphere_oblate_radius,
        )
        object.__setattr__(
            self,
            "_observer_pseudo_cylindrical_radius_over_mass",
            observer_rho,
        )
        object.__setattr__(
            self,
            "_observer_upper_surface_value",
            observer_upper_value,
        )
        object.__setattr__(
            self,
            "_observer_lower_surface_value",
            observer_lower_value,
        )
        object.__setattr__(
            self,
            "_observer_within_physical_annulus",
            observer_within_annulus,
        )
        object.__setattr__(self, "_escaped_descriptor_json", escape_json)
        object.__setattr__(
            self,
            "_escaped_descriptor_sha256",
            hashlib.sha256(escape_json.encode("utf-8")).hexdigest(),
        )

    def _assert_escape_descriptor_stable(self) -> None:
        current = _canonical_descriptor(
            self.escaped_observer_spectrum.descriptor(),
            "escaped observer spectrum descriptor",
        )
        current_json = json.dumps(
            current,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if current_json != self._escaped_descriptor_json:
            raise KerrFiniteThicknessFrameError(
                "escaped observer spectrum descriptor changed after construction"
            )

    def descriptor(self) -> Mapping[str, Any]:
        """Return content-complete finite configuration identity data."""

        self._assert_escape_descriptor_stable()
        fine_recorded_path_absolute = max(
            self.recorded_path_absolute_tolerance,
            self.fine_options.absolute_tolerance,
        )
        fine_recorded_path_relative = max(
            self.recorded_path_relative_tolerance,
            self.fine_options.relative_tolerance,
        )
        coarse_recorded_path_absolute = max(
            self.recorded_path_absolute_tolerance,
            self._coarse_ray_options.absolute_tolerance,
        )
        coarse_recorded_path_relative = max(
            self.recorded_path_relative_tolerance,
            self._coarse_ray_options.relative_tolerance,
        )
        descriptor = {
            "convergence": {
                "coarseToleranceMultiplier": self.coarse_tolerance_multiplier,
                "diskRadiusAbsoluteToleranceM": (
                    self.disk_radius_absolute_tolerance_m
                ),
                "diskRadiusRelativeTolerance": self.disk_radius_relative_tolerance,
                "emissionCosineAbsoluteTolerance": (
                    self.emission_cosine_absolute_tolerance
                ),
                "escapeDirectionToleranceRad": (
                    self.escape_direction_tolerance_rad
                ),
                "frequencyShiftRelativeTolerance": (
                    self.frequency_shift_relative_tolerance
                ),
                "specificIntensityAbsoluteTolerance": (
                    self.specific_intensity_absolute_tolerance
                ),
                "specificIntensityRelativeTolerance": (
                    self.specific_intensity_relative_tolerance
                ),
                "surfaceProbeCovectorRelativeTolerance": (
                    self.terminal_covector_tolerance
                ),
                "surfaceProbeEventToleranceM": self.terminal_event_tolerance_m,
                "terminalCovectorTolerance": self.terminal_covector_tolerance,
                "terminalEventToleranceM": self.terminal_event_tolerance_m,
            },
            "convergencePolicy": {
                "actual": {
                    "coarseToleranceMultiplier": (
                        self.coarse_tolerance_multiplier
                    ),
                    "diskRadiusAbsoluteToleranceM": (
                        self.disk_radius_absolute_tolerance_m
                    ),
                    "diskRadiusRelativeTolerance": (
                        self.disk_radius_relative_tolerance
                    ),
                    "emissionCosineAbsoluteTolerance": (
                        self.emission_cosine_absolute_tolerance
                    ),
                    "escapeDirectionToleranceRad": (
                        self.escape_direction_tolerance_rad
                    ),
                    "frequencyShiftRelativeTolerance": (
                        self.frequency_shift_relative_tolerance
                    ),
                    "specificIntensityAbsoluteTolerance": (
                        self.specific_intensity_absolute_tolerance
                    ),
                    "specificIntensityRelativeTolerance": (
                        self.specific_intensity_relative_tolerance
                    ),
                    "terminalCovectorTolerance": (
                        self.terminal_covector_tolerance
                    ),
                    "terminalEventToleranceM": self.terminal_event_tolerance_m,
                },
                "maxima": {
                    "coarseToleranceMultiplier": (
                        _MAXIMUM_COARSE_TOLERANCE_MULTIPLIER
                    ),
                    "diskRadiusAbsoluteToleranceM": (
                        _MAXIMUM_DISK_RADIUS_ABSOLUTE_TOLERANCE_SCALE
                        * max(1.0, self.metric.mass_m)
                    ),
                    "diskRadiusRelativeTolerance": (
                        _MAXIMUM_DISK_RADIUS_RELATIVE_TOLERANCE
                    ),
                    "emissionCosineAbsoluteTolerance": (
                        _MAXIMUM_EMISSION_COSINE_ABSOLUTE_TOLERANCE
                    ),
                    "escapeDirectionToleranceRad": (
                        _MAXIMUM_ESCAPE_DIRECTION_TOLERANCE_RAD
                    ),
                    "frequencyShiftRelativeTolerance": (
                        _MAXIMUM_FREQUENCY_SHIFT_RELATIVE_TOLERANCE
                    ),
                    "specificIntensityAbsoluteTolerance": (
                        _MAXIMUM_SPECIFIC_INTENSITY_ABSOLUTE_TOLERANCE
                    ),
                    "specificIntensityRelativeTolerance": (
                        _MAXIMUM_SPECIFIC_INTENSITY_RELATIVE_TOLERANCE
                    ),
                    "terminalCovectorTolerance": (
                        _MAXIMUM_TERMINAL_COVECTOR_TOLERANCE
                    ),
                    "terminalEventToleranceM": (
                        _MAXIMUM_TERMINAL_EVENT_TOLERANCE_SCALE
                        * max(1.0, self.metric.mass_m)
                    ),
                },
                "policy": (
                    "caller may tighten but may not exceed implementation maxima"
                ),
            },
            "diskThermalProxy": {
                "blackHoleMassKg": self.disk.black_hole_mass_kg,
                "colourCorrection": self.disk.colour_correction,
                "iscoRadiusM": self.disk.isco_radius_m,
                "massAccretionRateKgS": self.disk.mass_accretion_rate_kg_s,
                "orientation": self.disk.orientation,
                "radialReference": "equatorial-NT-at-matching-rho",
            },
            "escapeDirectionDiagnostic": {
                "frame": "finite-worldtube-KS-angular-continuation-direction",
                "isAsymptoticICRS": False,
                "isLocalPhotonMomentum": False,
            },
            "escapedObserverSpectrum": {
                "acceptedImplementations": "closed-built-in-types-only",
                "descriptor": json.loads(self._escaped_descriptor_json),
                "descriptorSha256": self._escaped_descriptor_sha256,
                "samplerAppliesAdditionalG3": False,
            },
            "finiteThicknessSurface": {
                "dimensionlessSpinMagnitude": (
                    self.surface.calibration.dimensionless_spin
                ),
                "eddingtonScaledMassAccretionRate": (
                    self.surface.calibration.eddington_scaled_mass_accretion_rate
                ),
                "heightRateIsIndependentOfThermalRate": True,
                "maximumPhotosphereOblateRadiusM": (
                    self._maximum_photosphere_oblate_radius_m
                ),
                "orientation": self.surface.calibration.orientation,
                "outerRadiusOverMass": (
                    self.surface.calibration.outer_radius_over_mass
                ),
                "thinnessGateMaximumHOverRho": (
                    self.surface.calibration.thinness_gate_maximum_h_over_rho
                ),
                "surfaceIds": list(self.surface.surface_ids),
                "type": "Zhou-prescribed-stationary-photosphere",
            },
            "frequencyTransfer": {
                "boundaryValueToleranceM": (
                    self._resolved_boundary_value_tolerance_m
                ),
                "conservedQuantityTolerance": self.conserved_quantity_tolerance,
                "emitterEventToleranceM": (
                    self._resolved_emitter_event_tolerance_m
                ),
                "nullResidualLimit": self.frequency_null_residual_limit,
                "recordedPathTolerancePolicy": (
                    "per trace max(configured minimum, producing ray local "
                    "tolerance)"
                ),
                "requestedRecordedPathAbsoluteTolerance": (
                    self.recorded_path_absolute_tolerance
                ),
                "requestedRecordedPathRelativeTolerance": (
                    self.recorded_path_relative_tolerance
                ),
                "fineRecordedPathAbsoluteTolerance": (
                    fine_recorded_path_absolute
                ),
                "fineRecordedPathRelativeTolerance": (
                    fine_recorded_path_relative
                ),
                "coarseRecordedPathAbsoluteTolerance": (
                    coarse_recorded_path_absolute
                ),
                "coarseRecordedPathRelativeTolerance": (
                    coarse_recorded_path_relative
                ),
            },
            "implementationId": IMPLEMENTATION_ID,
            "metric": {
                "massM": self.metric.mass_m,
                "signedSpinAM": self.metric.spin_a_m,
                "singularityGuardM": self.metric.singularity_guard_m,
                "sourceId": self.metric.source_id,
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
                "materialClearance": {
                    "lowerFaceSignedValue": self._observer_lower_surface_value,
                    "policy": (
                        "outside both faces whenever rho is in the physical "
                        "annulus"
                    ),
                    "pseudoCylindricalRadiusOverMass": (
                        self._observer_pseudo_cylindrical_radius_over_mass
                    ),
                    "status": "outside-certified",
                    "upperFaceSignedValue": self._observer_upper_surface_value,
                    "withinPhysicalAnnulus": (
                        self._observer_within_physical_annulus
                    ),
                },
            },
            "observerFrequencyFrame": "observer-ZAMO",
            "rayOptions": {
                "coarseDerived": _ray_options_descriptor(
                    self._coarse_ray_options
                ),
                "fine": _ray_options_descriptor(self.fine_options),
                "independentFineCoarseTraces": True,
                "recordCoarsePath": True,
            },
            "scientificStatus": dict(SCIENTIFIC_STATUS),
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
                "fine": _surface_options_descriptor(self.surface_options),
                "topologyAgreement": (
                    "surface ids, crossing order, face ids, orientations, "
                    "classifications, terminal outcome, and terminal target"
                ),
            },
            "traceAccuracyPolicy": {
                "actual": {
                    "fineAbsoluteTolerance": self.fine_options.absolute_tolerance,
                    "fineEventAffineTolerance": (
                        self.fine_options.event_affine_tolerance
                    ),
                    "fineEventValueTolerance": (
                        self.fine_options.event_value_tolerance
                    ),
                    "fineMaximumStep": self.fine_options.maximum_step,
                    "fineMetricInterpolationErrorLimit": (
                        self.fine_options.metric_interpolation_error_limit
                    ),
                    "fineNullResidualLimit": (
                        self.fine_options.null_residual_limit
                    ),
                    "fineRelativeTolerance": self.fine_options.relative_tolerance,
                    "surfaceAbsoluteTolerance": (
                        self.surface_options.absolute_tolerance
                    ),
                    "surfaceAffineTolerance": (
                        self.surface_options.affine_tolerance
                    ),
                    "surfaceMetricInterpolationErrorLimit": (
                        self.surface_options.metric_interpolation_error_limit
                    ),
                    "surfaceNullResidualLimit": (
                        self.surface_options.null_residual_limit
                    ),
                    "surfaceRelativeTolerance": (
                        self.surface_options.relative_tolerance
                    ),
                },
                "maxima": {
                    "fineAbsoluteTolerance": (
                        _MAXIMUM_FINE_LOCAL_ABSOLUTE_TOLERANCE
                    ),
                    "fineEventAffineTolerance": (
                        _MAXIMUM_FINE_EVENT_AFFINE_TOLERANCE_OVER_MASS
                        * max(1.0, self.metric.mass_m)
                    ),
                    "fineEventValueTolerance": (
                        _MAXIMUM_FINE_EVENT_VALUE_TOLERANCE_OVER_MASS
                        * self.metric.mass_m
                    ),
                    "fineMaximumStep": (
                        _MAXIMUM_FINE_STEP_OVER_MASS * self.metric.mass_m
                    ),
                    "fineMetricInterpolationErrorLimit": (
                        _MAXIMUM_METRIC_INTERPOLATION_ERROR_LIMIT
                    ),
                    "fineNullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
                    "fineRelativeTolerance": (
                        _MAXIMUM_FINE_LOCAL_RELATIVE_TOLERANCE
                    ),
                    "surfaceAbsoluteTolerance": (
                        _MAXIMUM_SURFACE_LOCAL_ABSOLUTE_TOLERANCE
                    ),
                    "surfaceAffineTolerance": (
                        _MAXIMUM_SURFACE_AFFINE_TOLERANCE_OVER_MASS
                        * max(1.0, self.metric.mass_m)
                    ),
                    "surfaceMetricInterpolationErrorLimit": (
                        _MAXIMUM_METRIC_INTERPOLATION_ERROR_LIMIT
                    ),
                    "surfaceNullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
                    "surfaceRelativeTolerance": (
                        _MAXIMUM_SURFACE_LOCAL_RELATIVE_TOLERANCE
                    ),
                },
                "policy": (
                    "fine trace and surface localization may tighten but may "
                    "not exceed implementation maxima"
                ),
            },
            "termination": {
                "captureRadiusM": self.termination.capture_radius_m,
                "captureTargetId": self.termination.capture_target_id,
                "escapeRadiusM": self.termination.escape_radius_m,
                "escapeTargetId": self.termination.escape_target_id,
                "spinAM": self.termination.spin_a_m,
                "visibilityConstraints": {
                    "captureStrictlyInsideDiskIsco": True,
                    "escapeStrictlyOutsideMaximumPhotosphereOblateRadius": True,
                    "maximumPhotosphereOblateRadiusM": (
                        self._maximum_photosphere_oblate_radius_m
                    ),
                },
            },
            "tolerancePolicy": {
                "actual": {
                    "boundaryValueToleranceM": (
                        self._resolved_boundary_value_tolerance_m
                    ),
                    "conservedQuantityTolerance": (
                        self.conserved_quantity_tolerance
                    ),
                    "emitterEventToleranceM": (
                        self._resolved_emitter_event_tolerance_m
                    ),
                    "nullResidualLimit": self.frequency_null_residual_limit,
                    "recordedPathAbsoluteTolerance": (
                        self.recorded_path_absolute_tolerance
                    ),
                    "recordedPathRelativeTolerance": (
                        self.recorded_path_relative_tolerance
                    ),
                    "fineResolvedRecordedPathAbsoluteTolerance": (
                        fine_recorded_path_absolute
                    ),
                    "fineResolvedRecordedPathRelativeTolerance": (
                        fine_recorded_path_relative
                    ),
                    "coarseResolvedRecordedPathAbsoluteTolerance": (
                        coarse_recorded_path_absolute
                    ),
                    "coarseResolvedRecordedPathRelativeTolerance": (
                        coarse_recorded_path_relative
                    ),
                    "surfaceValueTolerance": (
                        self.surface_options.surface_value_tolerance
                    ),
                },
                "maxima": {
                    "boundaryValueToleranceM": (
                        _MAXIMUM_BOUNDARY_VALUE_TOLERANCE_OVER_MASS
                        * self.metric.mass_m
                    ),
                    "conservedQuantityTolerance": (
                        _MAXIMUM_CONSERVED_QUANTITY_TOLERANCE
                    ),
                    "emitterEventToleranceM": (
                        _MAXIMUM_EMITTER_EVENT_TOLERANCE_OVER_MASS
                        * self.metric.mass_m
                    ),
                    "nullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
                    "recordedPathAbsoluteTolerance": (
                        _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
                        * max(1.0, self.metric.mass_m)
                    ),
                    "recordedPathRelativeTolerance": (
                        _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
                    ),
                    "fineResolvedRecordedPathAbsoluteTolerance": (
                        _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
                        * max(1.0, self.metric.mass_m)
                    ),
                    "fineResolvedRecordedPathRelativeTolerance": (
                        _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
                    ),
                    "coarseResolvedRecordedPathAbsoluteTolerance": (
                        _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
                        * max(1.0, self.metric.mass_m)
                    ),
                    "coarseResolvedRecordedPathRelativeTolerance": (
                        _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
                    ),
                    "surfaceValueTolerance": _MAXIMUM_SURFACE_VALUE_TOLERANCE,
                },
                "policy": (
                    "caller may tighten but may not exceed implementation maxima"
                ),
            },
            "version": 1,
        }
        return _canonical_descriptor(descriptor, "finite-thickness sampler descriptor")

    def _validate_ray_trace(
        self,
        ray: RayTraceResult,
        initial: HamiltonianState,
        *,
        label: str,
        ray_options: RayTraceOptions,
    ) -> MultiInteriorSurfaceTrace:
        if type(ray) is not RayTraceResult:
            raise KerrFiniteThicknessFrameError(f"{label} ray type is foreign")
        if ray.outcome not in ("captured", "escaped", OPAQUE_OUTCOME):
            raise KerrFiniteThicknessFrameError(f"{label} ray has no visible source")
        if ray.failure_reason is not None:
            raise KerrFiniteThicknessFrameError(f"{label} ray reports failure")
        if (
            type(ray.accepted_steps) is not int
            or ray.accepted_steps < 1
            or ray.accepted_steps > ray_options.maximum_accepted_steps
            or type(ray.rejected_steps) is not int
            or ray.rejected_steps < 0
            or ray.rejected_steps > ray_options.maximum_rejected_steps
            or not math.isfinite(ray.affine_length)
            or ray.affine_length <= 0.0
            or ray.affine_length > ray_options.maximum_affine_length
            or not math.isfinite(ray.maximum_null_residual)
            or ray.maximum_null_residual < 0.0
            or ray.maximum_null_residual
            > min(
                ray_options.null_residual_limit,
                self.frequency_null_residual_limit,
            )
            or not math.isfinite(ray.maximum_metric_interpolation_error)
            or ray.maximum_metric_interpolation_error < 0.0
            or ray.maximum_metric_interpolation_error
            > ray_options.metric_interpolation_error_limit
        ):
            raise KerrFiniteThicknessFrameError(
                f"{label} ray convergence diagnostics exceed their budgets"
            )
        if not ray.segments or ray.segments[0].start != initial:
            raise KerrFiniteThicknessFrameError(
                f"{label} ray does not start at the shared observer state"
            )
        if ray.segments[-1].end != ray.terminal_state:
            raise KerrFiniteThicknessFrameError(
                f"{label} ray path does not own its terminal state"
            )
        if len(ray.segments) != ray.accepted_steps:
            raise KerrFiniteThicknessFrameError(
                f"{label} ray path length disagrees with accepted steps"
            )
        if any(
            previous.end != current.start
            for previous, current in zip(ray.segments, ray.segments[1:])
        ):
            raise KerrFiniteThicknessFrameError(f"{label} ray path is discontinuous")
        trace = ray.multi_surface_trace
        if type(trace) is not MultiInteriorSurfaceTrace:
            raise KerrFiniteThicknessFrameError(
                f"{label} ray lacks exact multi-surface evidence"
            )
        if not trace.topology_converged:
            raise KerrFiniteThicknessFrameError(
                f"{label} multi-surface topology is unresolved"
            )
        if trace.surface_ids != tuple(sorted(self.surface.surface_ids)):
            raise KerrFiniteThicknessFrameError(
                f"{label} trace carries foreign surface ids"
            )
        if trace.base_subdivisions_per_step != self.surface_options.subdivisions_per_segment:
            raise KerrFiniteThicknessFrameError(
                f"{label} trace subdivisions disagree with sampler options"
            )
        if trace.verification_subdivisions_per_step != (
            2 * self.surface_options.subdivisions_per_segment
        ):
            raise KerrFiniteThicknessFrameError(
                f"{label} trace lacks the declared N-vs-2N topology check"
            )
        if (
            not math.isfinite(trace.maximum_probe_event_difference)
            or trace.maximum_probe_event_difference < 0.0
            or trace.maximum_probe_event_difference
            > self.terminal_event_tolerance_m
            or not math.isfinite(
                trace.maximum_probe_covector_relative_difference
            )
            or trace.maximum_probe_covector_relative_difference < 0.0
            or trace.maximum_probe_covector_relative_difference
            > self.terminal_covector_tolerance
            or type(trace.probe_reintegrations) is not int
            or trace.probe_reintegrations < 0
            or trace.probe_reintegrations
            > self.surface_options.maximum_reintegrations
            or type(trace.surface_value_evaluations) is not int
            or trace.surface_value_evaluations < 0
        ):
            raise KerrFiniteThicknessFrameError(
                f"{label} multi-surface convergence diagnostics are invalid"
            )

        prefix = 0.0
        segment_prefixes: list[float] = []
        for segment in ray.segments:
            segment_prefixes.append(prefix)
            prefix += segment.affine_length
        terminal_seen = False
        for entry in trace.crossings:
            crossing = entry.crossing
            if terminal_seen:
                raise KerrFiniteThicknessFrameError(
                    f"{label} trace continues behind an opaque face"
                )
            if entry.surface_id not in self.surface.surface_ids:
                raise KerrFiniteThicknessFrameError(
                    f"{label} crossing has a foreign surface id"
                )
            if (
                type(crossing.segment_index) is not int
                or crossing.segment_index < 0
                or crossing.segment_index >= len(ray.segments)
            ):
                raise KerrFiniteThicknessFrameError(
                    f"{label} crossing segment index is invalid"
                )
            segment = ray.segments[crossing.segment_index]
            if (
                crossing.segment_affine_length < 0.0
                or crossing.segment_affine_length > segment.affine_length
                or crossing.ray_affine_length < 0.0
                or not math.isfinite(crossing.bracket_affine_width)
                or crossing.bracket_affine_width < 0.0
                or crossing.bracket_affine_width
                > self.surface_options.affine_tolerance
                or type(crossing.iterations) is not int
                or crossing.iterations < 0
                or crossing.iterations > self.surface_options.maximum_iterations
            ):
                raise KerrFiniteThicknessFrameError(
                    f"{label} crossing affine position is invalid"
                )
            expected_affine = (
                segment_prefixes[crossing.segment_index]
                + crossing.segment_affine_length
            )
            if not math.isclose(
                crossing.ray_affine_length,
                expected_affine,
                rel_tol=2.0e-13,
                abs_tol=1.0e-12,
            ):
                raise KerrFiniteThicknessFrameError(
                    f"{label} crossing is not bound to its path segment"
                )
            if entry.decision != self.surface.classify(entry.surface_id, crossing):
                raise KerrFiniteThicknessFrameError(
                    f"{label} crossing classification is not surface-owned"
                )
            terminal_seen = entry.decision.terminates

        if ray.outcome == OPAQUE_OUTCOME:
            if not trace.crossings:
                raise KerrFiniteThicknessFrameError(
                    f"{label} opaque ray lacks a terminal face"
                )
            terminal = trace.crossings[-1]
            if (
                not terminal.decision.terminates
                or terminal.decision.outcome != ray.outcome
                or terminal.decision.target_id != ray.terminal_target_id
                or terminal.crossing.state != ray.terminal_state
                or terminal.surface_id not in (UPPER_SURFACE_ID, LOWER_SURFACE_ID)
            ):
                raise KerrFiniteThicknessFrameError(
                    f"{label} terminal face does not own the ray outcome"
                )
        elif terminal_seen:
            raise KerrFiniteThicknessFrameError(
                f"{label} boundary ray hides an opaque face"
            )
        else:
            expected_target = (
                self.termination.capture_target_id
                if ray.outcome == "captured"
                else self.termination.escape_target_id
            )
            expected_radius = (
                self.termination.capture_radius_m
                if ray.outcome == "captured"
                else self.termination.escape_radius_m
            )
            terminal_radius = self.termination.radius(ray.terminal_state)
            if (
                ray.terminal_target_id != expected_target
                or not math.isfinite(terminal_radius)
                or abs(terminal_radius - expected_radius)
                > min(
                    ray_options.event_value_tolerance,
                    self._resolved_boundary_value_tolerance_m,
                )
            ):
                raise KerrFiniteThicknessFrameError(
                    f"{label} boundary endpoint is not owned by the configured "
                    "Kerr worldtube"
                )
        return trace

    def _validate_refinement(
        self,
        refinement: RayRefinementResult,
        initial: HamiltonianState,
    ) -> tuple[RayTraceResult, RayTraceResult]:
        if type(refinement) is not RayRefinementResult:
            raise KerrFiniteThicknessFrameError("refinement result type is foreign")
        fine = refinement.fine
        coarse = refinement.coarse
        if fine is coarse or fine.segments is coarse.segments:
            raise KerrFiniteThicknessFrameError(
                "fine/coarse traces are not independent objects"
            )
        fine_trace = self._validate_ray_trace(
            fine,
            initial,
            label="fine",
            ray_options=self.fine_options,
        )
        coarse_trace = self._validate_ray_trace(
            coarse,
            initial,
            label="coarse",
            ray_options=self._coarse_ray_options,
        )
        outcome_agrees = fine.outcome == coarse.outcome
        target_agrees = fine.terminal_target_id == coarse.terminal_target_id
        discretizations_differ = (
            fine.accepted_steps,
            fine.rejected_steps,
        ) != (
            coarse.accepted_steps,
            coarse.rejected_steps,
        )
        event_difference = _event_difference(fine.terminal_state, coarse.terminal_state)
        covector_difference = _covector_relative_difference(
            fine.terminal_state,
            coarse.terminal_state,
        )
        topology_agrees = _topology_token(fine) == _topology_token(coarse)
        expected_converged = (
            outcome_agrees
            and target_agrees
            and discretizations_differ
            and topology_agrees
            and fine_trace.topology_converged
            and coarse_trace.topology_converged
            and event_difference <= self.terminal_event_tolerance_m
            and covector_difference <= self.terminal_covector_tolerance
        )
        if (
            refinement.outcome_agrees is not outcome_agrees
            or refinement.terminal_target_agrees is not target_agrees
            or refinement.discretizations_differ is not discretizations_differ
            or not math.isclose(
                refinement.terminal_event_difference,
                event_difference,
                rel_tol=2.0e-13,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                refinement.terminal_covector_difference,
                covector_difference,
                rel_tol=2.0e-13,
                abs_tol=1.0e-15,
            )
            or refinement.converged is not expected_converged
        ):
            raise KerrFiniteThicknessFrameError(
                "stored fine/coarse convergence diagnostics are stale"
            )
        if not expected_converged:
            raise KerrFiniteThicknessFrameError(
                "fine/coarse geodesic or multi-surface topology did not converge"
            )
        return fine, coarse

    def _transfer(
        self,
        ray: RayTraceResult,
        initial: HamiltonianState,
        frequencies: tuple[float, ...],
        *,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
    ) -> KerrFiniteThicknessSpectrumResult:
        # A recorded segment was produced only to the local tolerance of its
        # owning fine/coarse trace.  Reintegrating it with a *tighter* absolute
        # or relative tolerance can reject an otherwise valid producer path
        # merely because the two adaptive step sequences differ.  Treat the
        # configured recorded-path values as requested lower bounds and never
        # certify more tightly than the ray that actually produced this path.
        # The transfer layer still enforces its independent absolute maxima.
        path_absolute_tolerance = max(
            self.recorded_path_absolute_tolerance,
            ray_options.absolute_tolerance,
        )
        path_relative_tolerance = max(
            self.recorded_path_relative_tolerance,
            ray_options.relative_tolerance,
        )
        result, replay_certificate = (
            _transfer_kerr_finite_thickness_spectrum_certified(
                self.surface,
                self.disk,
                ray,
                initial,
                self._observer_tetrad.four_velocity,
                frequencies,
                escaped_observer_spectrum=self.escaped_observer_spectrum,
                termination=self.termination,
                ray_options=ray_options,
                surface_options=surface_options,
                null_residual_limit=self.frequency_null_residual_limit,
                conserved_quantity_tolerance=self.conserved_quantity_tolerance,
                surface_value_tolerance=surface_options.surface_value_tolerance,
                recorded_path_absolute_tolerance=path_absolute_tolerance,
                recorded_path_relative_tolerance=path_relative_tolerance,
                boundary_value_tolerance_m=(
                    self._resolved_boundary_value_tolerance_m
                ),
                emitter_event_tolerance_m=(
                    self._resolved_emitter_event_tolerance_m
                ),
            )
        )
        if type(result) is not KerrFiniteThicknessSpectrumResult:
            raise KerrFiniteThicknessFrameError("transfer result type is foreign")
        # Re-enter the immutable result's complete validator even if a wrapper
        # returned an object whose frozen fields were changed with low-level APIs.
        try:
            result = replace(
                result,
                _replay_certificate=replay_certificate,
            )
        except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
            raise KerrFiniteThicknessFrameError(
                "transfer result failed independent self-revalidation"
            ) from error
        if (
            result.surface is not self.surface
            or result.disk is not self.disk
            or result.termination is not self.termination
            or result.ray_options is not ray_options
            or result.surface_options is not surface_options
            or result.ray is not ray
            or result.observer_initial_state is not initial
            or result.escaped_observer_spectrum is not self.escaped_observer_spectrum
            or result.observer_four_velocity != self._observer_tetrad.four_velocity
            or result.observer_frequencies_hz != frequencies
            or result.null_residual_limit != self.frequency_null_residual_limit
            or result.conserved_quantity_tolerance
            != self.conserved_quantity_tolerance
            or result.surface_value_tolerance
            != surface_options.surface_value_tolerance
            or result.recorded_path_absolute_tolerance
            != path_absolute_tolerance
            or result.recorded_path_relative_tolerance
            != path_relative_tolerance
            or result.boundary_value_tolerance_m
            != self._resolved_boundary_value_tolerance_m
            or result.emitter_event_tolerance_m
            != self._resolved_emitter_event_tolerance_m
        ):
            raise KerrFiniteThicknessFrameError(
                "transfer result is not bound to its ray and sampler model"
            )
        if ray.outcome == OPAQUE_OUTCOME:
            trace = ray.multi_surface_trace
            if (
                type(trace) is not MultiInteriorSurfaceTrace
                or result.terminal_surface_entry is not trace.crossings[-1]
                or result.terminal_surface_entry.crossing.state != ray.terminal_state
            ):
                raise KerrFiniteThicknessFrameError(
                    "transfer terminal face is not the ray's proven terminal entry"
                )
        return result

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        """Trace, transfer, compare, and return one certified fine sample."""

        x = _finite_number(screen_x, "screen_x")
        y = _finite_number(screen_y, "screen_y")
        frequencies = _positive_frequencies(observer_frequencies_hz)
        self._assert_escape_descriptor_stable()
        initial = kerr_zamo_camera_ray(
            self.metric,
            observer_radius_m=self.observer_radius_m,
            screen_x=x,
            screen_y=y,
            theta_rad=self.observer_theta_rad,
            phi_ks_rad=self.observer_phi_ks_rad,
            coordinate_time_m=self.observer_coordinate_time_m,
        )
        if initial.event != self._observer_tetrad.event:
            raise KerrFiniteThicknessFrameError(
                "camera ray and fixed ZAMO observer disagree"
            )
        refinement = trace_refined_null_geodesic(
            self.metric,
            initial,
            termination=self.termination,
            multi_interior_surface=self.surface,
            surface_options=self.surface_options,
            fine_options=self.fine_options,
            record_coarse_path=True,
            coarse_tolerance_multiplier=self.coarse_tolerance_multiplier,
            terminal_event_tolerance=self.terminal_event_tolerance_m,
            terminal_covector_tolerance=self.terminal_covector_tolerance,
        )
        fine_ray, coarse_ray = self._validate_refinement(refinement, initial)
        fine = self._transfer(
            fine_ray,
            initial,
            frequencies,
            ray_options=self.fine_options,
            surface_options=self.surface_options,
        )
        self._assert_escape_descriptor_stable()
        coarse = self._transfer(
            coarse_ray,
            initial,
            frequencies,
            ray_options=self._coarse_ray_options,
            surface_options=self._coarse_surface_options,
        )
        self._assert_escape_descriptor_stable()

        if fine.transfer_configuration_sha256 == coarse.transfer_configuration_sha256:
            raise KerrFiniteThicknessFrameError(
                "fine/coarse transfer identities do not bind distinct trace options"
            )
        if (
            fine.escape_spectrum_descriptor_sha256
            != coarse.escape_spectrum_descriptor_sha256
        ):
            raise KerrFiniteThicknessFrameError(
                "fine/coarse escape spectrum identities disagree"
            )
        if fine.source_kind != coarse.source_kind:
            raise KerrFiniteThicknessFrameError("fine/coarse visible sources disagree")
        fine_topology = _topology_token(fine_ray)
        if fine_topology != _topology_token(coarse_ray):
            raise KerrFiniteThicknessFrameError(
                "fine/coarse multi-surface topologies disagree"
            )

        disk_radius_difference = 0.0
        relative_g_difference = 0.0
        frequency_shift: float | None = None
        escape_direction: tuple[float, float, float] | None = None
        if fine.source_kind == "finite-thickness-disk":
            required = (
                fine.equatorial_reference_radius_m,
                coarse.equatorial_reference_radius_m,
                fine.pseudo_cylindrical_radius_over_mass,
                coarse.pseudo_cylindrical_radius_over_mass,
                fine.frequency_shift_g,
                coarse.frequency_shift_g,
                fine.photon_projection,
                coarse.photon_projection,
            )
            if any(value is None for value in required):
                raise KerrFiniteThicknessFrameError(
                    "finite-thickness source lacks physical diagnostics"
                )
            fine_radius = fine.equatorial_reference_radius_m  # type: ignore[assignment]
            coarse_radius = coarse.equatorial_reference_radius_m  # type: ignore[assignment]
            if not _within_tolerance(
                fine_radius,
                coarse_radius,
                absolute_tolerance=self.disk_radius_absolute_tolerance_m,
                relative_tolerance=self.disk_radius_relative_tolerance,
            ):
                raise KerrFiniteThicknessFrameError(
                    "fine/coarse pseudo-cylindrical radii disagree"
                )
            disk_radius_difference = abs(fine_radius - coarse_radius)
            if fine.face != coarse.face:
                raise KerrFiniteThicknessFrameError(
                    "fine/coarse visible photosphere faces disagree"
                )
            fine_shift = fine.frequency_shift_g  # type: ignore[assignment]
            coarse_shift = coarse.frequency_shift_g  # type: ignore[assignment]
            if not _within_tolerance(
                fine_shift,
                coarse_shift,
                absolute_tolerance=0.0,
                relative_tolerance=self.frequency_shift_relative_tolerance,
            ):
                raise KerrFiniteThicknessFrameError(
                    "fine/coarse frequency shifts disagree"
                )
            relative_g_difference = abs(fine_shift - coarse_shift) / max(
                abs(fine_shift),
                abs(coarse_shift),
                1.0e-300,
            )
            fine_mu = fine.photon_projection.outgoing_cosine  # type: ignore[union-attr]
            coarse_mu = coarse.photon_projection.outgoing_cosine  # type: ignore[union-attr]
            if abs(fine_mu - coarse_mu) > self.emission_cosine_absolute_tolerance:
                raise KerrFiniteThicknessFrameError(
                    "fine/coarse signed face emission cosines disagree"
                )
            if fine_mu <= 0.0 or coarse_mu <= 0.0:
                raise KerrFiniteThicknessFrameError(
                    "visible photosphere needs strictly outgoing signed mu"
                )
            frequency_shift = fine_shift
            visible_source = "disk"
        elif fine.source_kind == "captured-boundary":
            if any(
                value != 0.0 or math.copysign(1.0, value) < 0.0
                for value in (
                    *fine.observed_specific_intensities_nu,
                    *coarse.observed_specific_intensities_nu,
                )
            ):
                raise KerrFiniteThicknessFrameError(
                    "captured boundary is not exactly positive-zero black"
                )
            visible_source = "captured-boundary"
        elif fine.source_kind == "escaped-boundary":
            for label, transfer, ray in (
                ("fine", fine, fine_ray),
                ("coarse", coarse, coarse_ray),
            ):
                if ray.terminal_target_id is None:
                    raise KerrFiniteThicknessFrameError(
                        f"{label} escaped ray lacks a terminal target"
                    )
                expected = tuple(
                    self.escaped_observer_spectrum(
                        ray.terminal_state,
                        frequency,
                        ray.terminal_target_id,
                    )
                    for frequency in frequencies
                )
                if transfer.observed_specific_intensities_nu != expected:
                    raise KerrFiniteThicknessFrameError(
                        f"{label} escape spectrum disagrees with its closed provider"
                    )
            fine_direction = _finite_worldtube_direction(
                self.metric,
                fine_ray.terminal_state,
            )
            coarse_direction = _finite_worldtube_direction(
                self.metric,
                coarse_ray.terminal_state,
            )
            separation = _angular_separation(fine_direction, coarse_direction)
            if separation > self.escape_direction_tolerance_rad:
                raise KerrFiniteThicknessFrameError(
                    "fine/coarse finite-worldtube escape directions disagree"
                )
            escape_direction = fine_direction
            visible_source = "escaped-boundary"
        else:
            raise KerrFiniteThicknessFrameError("transfer returned unknown source kind")

        if (
            len(fine.observed_specific_intensities_nu) != len(frequencies)
            or len(coarse.observed_specific_intensities_nu) != len(frequencies)
        ):
            raise KerrFiniteThicknessFrameError("transfer returned wrong bin count")
        errors = tuple(
            abs(left - right)
            for left, right in zip(
                fine.observed_specific_intensities_nu,
                coarse.observed_specific_intensities_nu,
            )
        )
        for index, (left, right) in enumerate(
            zip(
                fine.observed_specific_intensities_nu,
                coarse.observed_specific_intensities_nu,
            )
        ):
            if not _within_tolerance(
                left,
                right,
                absolute_tolerance=self.specific_intensity_absolute_tolerance,
                relative_tolerance=self.specific_intensity_relative_tolerance,
            ):
                raise KerrFiniteThicknessFrameError(
                    f"fine/coarse specific intensity bin {index} disagrees"
                )

        audit = RayConvergenceAudit(
            maximum_null_residual=max(
                fine_ray.maximum_null_residual,
                coarse_ray.maximum_null_residual,
            ),
            maximum_metric_interpolation_error=max(
                fine_ray.maximum_metric_interpolation_error,
                coarse_ray.maximum_metric_interpolation_error,
            ),
            terminal_event_difference_m=refinement.terminal_event_difference,
            terminal_covector_relative_difference=(
                refinement.terminal_covector_difference
            ),
            disk_radius_difference_m=disk_radius_difference,
            relative_g_difference=relative_g_difference,
            surface_bracket_affine_width=max(
                _maximum_surface_bracket(fine_ray),
                _maximum_surface_bracket(coarse_ray),
            ),
            accepted_steps=max(fine_ray.accepted_steps, coarse_ray.accepted_steps),
            rejected_steps=max(fine_ray.rejected_steps, coarse_ray.rejected_steps),
            ray_gate_passed=True,
            source_gate_passed=True,
            transfer_gate_passed=True,
        )
        return SpectralRaySample(
            specific_intensities_nu=fine.observed_specific_intensities_nu,
            absolute_errors_nu=errors,
            visible_source=visible_source,
            topology_signature=fine_topology,
            frequency_shift_g=frequency_shift,
            escape_direction=escape_direction,
            ray_converged=True,
            convergence_audit=audit,
        )


__all__ = (
    "IMPLEMENTATION_ID",
    "KerrFiniteThicknessFrameError",
    "KerrFiniteThicknessRaySampler",
    "SCIENTIFIC_STATUS",
)
