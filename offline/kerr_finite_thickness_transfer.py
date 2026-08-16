"""First-visible scalar spectrum from a stationary finite-thickness Kerr face.

This module consumes a *completed and topology-converged* multi-surface ray,
then deterministically re-traces it with its exact termination and trace
options to authenticate the complete first-visible history.  Geometry belongs
to ``KerrFiniteThicknessMultiSurface``; the local matter frame and signed face
projection belong to ``KerrFiniteThicknessFaceEmitter``; the radial thermal
reference belongs to ``StationaryNovikovThorneDisk``.  The only radiative
composition performed here is

``I_nu,obs = g**3 f_D20(mu) I_nu,NT(rho, nu_obs / g)``.

The photosphere is the stationary phenomenological Zhou calibration.  Its
local thermal flux is the *equatorial* Page--Thorne value at the same
pseudo-cylindrical radius ``rho``.  This is not a hydrostatic or
off-equatorial geodesic disk, returning-radiation transport, a solved
atmosphere, GRMHD, or a fine/coarse whole-ray convergence product.
The calibration's dimensionless accretion rate and the thermal disk's SI
accretion rate remain independently caller-supplied: this module does not
invent an Eddington-rate convention to convert or equate them.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, Sequence, TypeAlias

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    CertifiedRecordedPathSampler,
    ClassifiedMultiInteriorSurfaceCrossing,
    HamiltonianState,
    MultiInteriorSurfaceTrace,
    RayPathSegment,
    RayTraceOptions,
    RayTraceResult,
    RecordedPathSamplingError,
    RecordedPathSamplingOptions,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
    hamiltonian_null_residual,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateEvent,
    KerrOblateTermination,
    kerr_constants_of_motion,
    kerr_ks_event_to_oblate,
    kerr_ks_event_to_oblate_meridional,
    stationary_axisymmetric_constants,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_finite_thickness import (
    EDDINGTON_SCALING_DEFINITION,
    HEIGHT_SOURCE_URL,
    LOWER,
    MODEL_IMPLEMENTATION_ID as CALIBRATION_IMPLEMENTATION_ID,
    PRIMARY_SOURCE_URL as CALIBRATION_PRIMARY_SOURCE_URL,
    UPPER,
    PhotosphereFace,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
    KerrFiniteThicknessPhotonProjection,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    OPAQUE_OUTCOME,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_finite_thickness_replay_certificate import (
    _issue_replay_certificate,
    _require_replay_certificate,
)
from offline.spacetime import Vector4, bilinear


IMPLEMENTATION_ID: Final = "kerr-finite-thickness-first-visible-transfer/v1"

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "first-visible scalar transfer from a stationary phenomenological "
            "finite-height Kerr photosphere"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "observable": "observer-frame scalar spectral specific intensity I_nu",
        "transferInvariant": "I_nu / nu^3",
        "surface": "Zhou stationary phenomenological finite-height calibration",
        "thermalReference": (
            "equatorial Novikov-Thorne/Page-Thorne flux at matching "
            "pseudo-cylindrical rho"
        ),
        "heightFluxRateBinding": (
            "dimensionless height-calibration rate and SI thermal-disk rate "
            "are independently caller-supplied; no Eddington convention is "
            "invented here"
        ),
        "angularEmission": "built-in flux-conserving KERRBB D20 law",
        "captureBoundary": "exactly black",
        "escapeBoundary": "closed built-in observer-frame spectrum",
        "boundaryAuthentication": (
            "exact Kerr oblate termination provider, target identity, and "
            "terminal worldtube residual"
        ),
        "requiresTopologyConvergedMultiSurfaceTrace": True,
        "crossingProvenance": (
            "strict observer-to-source ordering plus certified Hamiltonian "
            "reintegration from each claimed recorded segment"
        ),
        "firstVisibleAuthentication": (
            "exact deterministic full ray and N/2N multi-surface topology "
            "replay with caller-supplied canonical trace options"
        ),
        "includesFineCoarseWholeRayConvergence": False,
        "isHydrostaticVerticalStructureSolution": False,
        "isOffEquatorialGeodesicDisk": False,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this stationary finite-height plus equatorial-NT-"
            "at-rho reference as hydrostatic structure, an off-equatorial "
            "geodesic disk, returning radiation, a solved atmosphere, GRMHD, "
            "or an independently fine/coarse converged ray product."
        ),
    }
)

FiniteThicknessSourceKind: TypeAlias = Literal[
    "finite-thickness-disk",
    "captured-boundary",
    "escaped-boundary",
]
BuiltInEscapedObserverSpectrum: TypeAlias = (
    DarkEscapedObserverSpectrum | PowerLawEscapedObserverSpectrum
)

_ANGULAR_LAW: Final = FluxConservingLinearLimbDarkening()
_DEFAULT_NULL_RESIDUAL_LIMIT: Final = 2.0e-7
_DEFAULT_CONSERVED_QUANTITY_TOLERANCE: Final = 2.0e-7
_DEFAULT_SURFACE_VALUE_TOLERANCE: Final = 2.0e-8
_DEFAULT_RECORDED_PATH_ABSOLUTE_TOLERANCE: Final = 2.0e-10
_DEFAULT_RECORDED_PATH_RELATIVE_TOLERANCE: Final = 2.0e-10
_MAXIMUM_NULL_RESIDUAL_LIMIT: Final = 1.0e-6
_MAXIMUM_CONSERVED_QUANTITY_TOLERANCE: Final = 1.0e-6
_MAXIMUM_SURFACE_VALUE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE: Final = 1.0e-7
_MAXIMUM_BOUNDARY_VALUE_TOLERANCE_OVER_MASS: Final = 1.0e-7
_MAXIMUM_EMITTER_EVENT_TOLERANCE_OVER_MASS: Final = 1.0e-7
_MAXIMUM_TRACE_ABSOLUTE_TOLERANCE_SCALE: Final = 1.0e-7
_MAXIMUM_TRACE_RELATIVE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_TRACE_NULL_RESIDUAL_LIMIT: Final = 1.0e-6
_MAXIMUM_TRACE_METRIC_INTERPOLATION_ERROR_LIMIT: Final = 1.0e-6
_MAXIMUM_RAY_STEP_SCALE: Final = 2.0
_MAXIMUM_TRACE_EVENT_TOLERANCE_SCALE: Final = 1.0e-7
_MAXIMUM_SURFACE_EVENT_VALUE_TOLERANCE: Final = 1.0e-7
_MAXIMUM_RAY_AFFINE_LENGTH_SCALE: Final = 1.0e6
_MAXIMUM_TRACE_INTEGER_BUDGET: Final = 1_000_000
_MAXIMUM_TRACE_ROOT_ITERATIONS: Final = 1_024
_MAXIMUM_SURFACE_SUBDIVISIONS: Final = 128
_FOUR_VELOCITY_TOLERANCE: Final = 5.0e-10
_RECOMPUTATION_RELATIVE_TOLERANCE: Final = 8.0e-13


class KerrFiniteThicknessTransferError(RuntimeError):
    """Raised when a nominal first-visible transfer is not self-consistent."""


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
        result = tuple(
            _finite_number(value, f"observer_frequencies_hz[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise ValueError("observer_frequencies_hz must be a sequence") from error
    if not result or any(value <= 0.0 for value in result):
        raise ValueError("observer frequencies must be non-empty and positive")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return result


def _finite_four_velocity(values: Sequence[float]) -> Vector4:
    if isinstance(values, (str, bytes)):
        raise ValueError("observer_four_velocity must contain four numbers")
    try:
        entries = tuple(values)
    except TypeError as error:
        raise ValueError(
            "observer_four_velocity must contain four numbers"
        ) from error
    if len(entries) != 4:
        raise ValueError("observer_four_velocity must contain four numbers")
    return tuple(  # type: ignore[return-value]
        _finite_number(value, f"observer_four_velocity[{index}]")
        for index, value in enumerate(entries)
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("descriptor must be finite canonical JSON data") from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _relative_close(
    actual: float,
    expected: float,
    tolerance: float = _RECOMPUTATION_RELATIVE_TOLERANCE,
    absolute_scale: float = 1.0,
) -> bool:
    return abs(actual - expected) <= tolerance * max(
        abs(actual),
        abs(expected),
        absolute_scale,
    )


def _surface_face_and_target(surface_id: str) -> tuple[PhotosphereFace, str]:
    if surface_id == UPPER_SURFACE_ID:
        return UPPER, UPPER_TARGET_ID
    if surface_id == LOWER_SURFACE_ID:
        return LOWER, LOWER_TARGET_ID
    raise ValueError("terminal surface id is not a finite-thickness face")


def _validate_builtin_escape_spectrum(
    spectrum: BuiltInEscapedObserverSpectrum,
) -> str:
    if type(spectrum) not in (
        DarkEscapedObserverSpectrum,
        PowerLawEscapedObserverSpectrum,
    ):
        raise TypeError(
            "escaped spectrum must be an exact built-in closed observer-frame "
            "spectrum"
        )
    descriptor = spectrum.descriptor()
    if not isinstance(descriptor, Mapping):
        raise ValueError("escaped spectrum descriptor must be a mapping")
    descriptor_json = _canonical_json(descriptor)
    decoded = json.loads(descriptor_json)
    if (
        decoded.get("frequencyFrame") != "observer"
        or decoded.get("quantity") != "spectral-specific-intensity-I_nu"
        or decoded.get("units") != "W m^-2 sr^-1 Hz^-1"
    ):
        raise ValueError("escaped spectrum is not closed in the observer I_nu frame")
    return descriptor_json


def _validated_trace_options_descriptor(
    metric: KerrKerrSchildMetric,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> Mapping[str, Any]:
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("ray_options must be the exact built-in RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError(
            "surface_options must be the exact built-in SurfaceEventOptions"
        )
    ray_values = {
        name: getattr(ray_options, name)
        for name in RayTraceOptions.__dataclass_fields__
    }
    surface_values = {
        name: getattr(surface_options, name)
        for name in SurfaceEventOptions.__dataclass_fields__
    }
    try:
        canonical_ray = RayTraceOptions(**ray_values)
        canonical_surface = SurfaceEventOptions(**surface_values)
    except (TypeError, ValueError) as error:
        raise ValueError("ray/surface trace options failed canonical replay") from error
    if canonical_ray != ray_options or canonical_surface != surface_options:
        raise ValueError("ray/surface trace options are not canonical")
    if type(ray_options.record_path) is not bool or not ray_options.record_path:
        raise ValueError("finite-thickness transfer requires record_path=True")

    ray_float_names = (
        "absolute_tolerance",
        "relative_tolerance",
        "initial_step",
        "minimum_step",
        "maximum_step",
        "maximum_affine_length",
        "null_residual_limit",
        "metric_interpolation_error_limit",
        "event_value_tolerance",
        "event_affine_tolerance",
    )
    surface_float_names = (
        "absolute_tolerance",
        "relative_tolerance",
        "null_residual_limit",
        "metric_interpolation_error_limit",
        "surface_value_tolerance",
        "affine_tolerance",
    )
    for owner, values, names in (
        ("ray_options", ray_values, ray_float_names),
        ("surface_options", surface_values, surface_float_names),
    ):
        for name in names:
            _finite_number(values[name], f"{owner}.{name}")

    scale = max(1.0, metric.mass_m)
    ray_maxima = {
        "absolute_tolerance": (
            _MAXIMUM_TRACE_ABSOLUTE_TOLERANCE_SCALE * scale
        ),
        "relative_tolerance": _MAXIMUM_TRACE_RELATIVE_TOLERANCE,
        "initial_step": _MAXIMUM_RAY_STEP_SCALE * scale,
        "minimum_step": _MAXIMUM_RAY_STEP_SCALE * scale,
        "maximum_step": _MAXIMUM_RAY_STEP_SCALE * scale,
        "maximum_affine_length": _MAXIMUM_RAY_AFFINE_LENGTH_SCALE * scale,
        "maximum_accepted_steps": _MAXIMUM_TRACE_INTEGER_BUDGET,
        "maximum_rejected_steps": _MAXIMUM_TRACE_INTEGER_BUDGET,
        "null_residual_limit": _MAXIMUM_TRACE_NULL_RESIDUAL_LIMIT,
        "metric_interpolation_error_limit": (
            _MAXIMUM_TRACE_METRIC_INTERPOLATION_ERROR_LIMIT
        ),
        "event_value_tolerance": (
            _MAXIMUM_TRACE_EVENT_TOLERANCE_SCALE * scale
        ),
        "event_affine_tolerance": (
            _MAXIMUM_TRACE_EVENT_TOLERANCE_SCALE * scale
        ),
        "event_maximum_iterations": _MAXIMUM_TRACE_ROOT_ITERATIONS,
    }
    surface_maxima = {
        "absolute_tolerance": (
            _MAXIMUM_TRACE_ABSOLUTE_TOLERANCE_SCALE * scale
        ),
        "relative_tolerance": _MAXIMUM_TRACE_RELATIVE_TOLERANCE,
        "null_residual_limit": _MAXIMUM_TRACE_NULL_RESIDUAL_LIMIT,
        "metric_interpolation_error_limit": (
            _MAXIMUM_TRACE_METRIC_INTERPOLATION_ERROR_LIMIT
        ),
        "surface_value_tolerance": _MAXIMUM_SURFACE_EVENT_VALUE_TOLERANCE,
        "affine_tolerance": _MAXIMUM_TRACE_EVENT_TOLERANCE_SCALE * scale,
        "maximum_iterations": _MAXIMUM_TRACE_ROOT_ITERATIONS,
        "maximum_reintegrations": _MAXIMUM_TRACE_INTEGER_BUDGET,
        "subdivisions_per_segment": _MAXIMUM_SURFACE_SUBDIVISIONS,
    }
    for owner, actual, maxima in (
        ("ray_options", ray_values, ray_maxima),
        ("surface_options", surface_values, surface_maxima),
    ):
        for name, maximum in maxima.items():
            if actual[name] > maximum:
                raise ValueError(
                    f"{owner}.{name} exceeds the trace replay policy maximum"
                )
    return {
        "policy": (
            "caller may tighten trace settings but may not exceed exact "
            "deterministic replay maxima"
        ),
        "rayTraceOptions": {
            "actual": ray_values,
            "maxima": ray_maxima,
            "required": {"record_path": True},
        },
        "scaleM": scale,
        "surfaceEventOptions": {
            "actual": surface_values,
            "maxima": surface_maxima,
            "requirements": {
                "subdivisionsEven": True,
                "subdivisionsMinimum": 2,
            },
        },
    }


def _configuration_descriptor_json(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    escaped_spectrum: BuiltInEscapedObserverSpectrum,
    termination: KerrOblateTermination,
    tolerance_policy: Mapping[str, Any],
    trace_options_descriptor: Mapping[str, Any],
) -> str:
    escape_descriptor = json.loads(_validate_builtin_escape_spectrum(escaped_spectrum))
    metric = surface.metric
    calibration = surface.calibration
    descriptor: dict[str, Any] = {
        "angularEmission": dict(_ANGULAR_LAW.descriptor()),
        "diskFluxReference": {
            "blackHoleMassKg": disk.black_hole_mass_kg,
            "colourCorrection": disk.colour_correction,
            "massAccretionRateKgS": disk.mass_accretion_rate_kg_s,
            "orientation": disk.orientation,
            "radiusCoordinate": "equatorial Page-Thorne at matching rho",
        },
        "escapeSpectrum": escape_descriptor,
        "finiteThicknessCalibration": {
            "dimensionlessSpinMagnitude": calibration.dimensionless_spin,
            "eddingtonScaling": dict(EDDINGTON_SCALING_DEFINITION),
            "eddingtonScaledMassAccretionRate": (
                calibration.eddington_scaled_mass_accretion_rate
            ),
            "heightSource": HEIGHT_SOURCE_URL,
            "implementationId": CALIBRATION_IMPLEMENTATION_ID,
            "orientation": calibration.orientation,
            "outerRadiusOverMass": calibration.outer_radius_over_mass,
            "primarySource": CALIBRATION_PRIMARY_SOURCE_URL,
            "thinnessGateMaximumHOverRho": (
                calibration.thinness_gate_maximum_h_over_rho
            ),
        },
        "implementationId": IMPLEMENTATION_ID,
        "metric": {
            "massM": metric.mass_m,
            "signedSpinAM": metric.spin_a_m,
            "singularityGuardM": metric.singularity_guard_m,
            "sourceId": metric.source_id,
            "timeDependent": metric.time_dependent,
        },
        "surfaceAdapter": {
            "opaqueOutcome": OPAQUE_OUTCOME,
            "surfaceIds": list(surface.surface_ids),
            "targets": {
                LOWER_SURFACE_ID: LOWER_TARGET_ID,
                UPPER_SURFACE_ID: UPPER_TARGET_ID,
            },
        },
        "termination": {
            "captureRadiusM": termination.capture_radius_m,
            "captureTargetId": termination.capture_target_id,
            "escapeRadiusM": termination.escape_radius_m,
            "escapeTargetId": termination.escape_target_id,
            "kind": "exact-kerr-oblate-worldtubes",
            "signedSpinAM": termination.spin_a_m,
            "visibilityConstraints": {
                "captureStrictlyInsideDiskIsco": True,
                "diskIscoRadiusM": (
                    calibration.isco_radius_over_mass * metric.mass_m
                ),
                "diskOuterRadiusM": (
                    calibration.outer_radius_over_mass * metric.mass_m
                ),
                "maximumPhotosphereOblateRadiusM": (
                    calibration.photosphere_point(
                        calibration.outer_radius_over_mass,
                        UPPER,
                    ).radius_over_mass
                    * metric.mass_m
                ),
                "escapeStrictlyOutsideMaximumPhotosphereOblateRadius": True,
            },
        },
        "tolerancePolicy": dict(tolerance_policy),
        "traceReplay": dict(trace_options_descriptor),
    }
    return _canonical_json(descriptor)


def _validate_model_ownership(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    escaped_spectrum: BuiltInEscapedObserverSpectrum,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> None:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be the exact finite-thickness adapter")
    if type(surface.metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must be the exact built-in Kerr metric")
    _validate_builtin_escape_spectrum(escaped_spectrum)
    _validated_trace_options_descriptor(
        surface.metric,
        ray_options,
        surface_options,
    )
    if type(surface.calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "surface calibration must be the exact built-in stationary "
            "finite-thickness calibration"
        )
    if type(disk) is not StationaryNovikovThorneDisk:
        raise TypeError("disk must be the exact StationaryNovikovThorneDisk")
    if disk.metric is not surface.metric:
        raise ValueError("disk and finite-thickness surface must own the same metric")
    if disk.orientation != surface.calibration.orientation:
        raise ValueError("disk and finite-thickness calibration orientations disagree")
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be the exact built-in Kerr oblate provider")
    if not _relative_close(
        termination.spin_a_m,
        surface.metric.spin_a_m,
        8.0 * math.ulp(1.0),
        surface.metric.mass_m,
    ):
        raise ValueError("termination and finite-thickness metric spins disagree")
    horizon_radius_m = surface.metric.outer_horizon_radius_m
    if termination.capture_radius_m < horizon_radius_m:
        raise ValueError("capture worldtube may not lie inside the Kerr horizon")
    if termination.capture_target_id == "analytic-kerr-event-horizon":
        if termination.capture_radius_m.hex() != horizon_radius_m.hex():
            raise ValueError(
                "event-horizon capture target must use the exact Kerr "
                "outer-horizon radius"
            )
    elif termination.capture_target_id == "analytic-kerr-stretched-horizon":
        if termination.capture_radius_m <= horizon_radius_m:
            raise ValueError(
                "stretched-horizon capture target must lie strictly outside "
                "the Kerr horizon"
            )
    else:
        raise ValueError("unsupported finite-thickness capture target id")
    if termination.escape_target_id != "analytic-kerr-escape-worldtube":
        raise ValueError("unsupported finite-thickness escape target id")
    disk_isco_radius_m = (
        surface.calibration.isco_radius_over_mass * surface.metric.mass_m
    )
    maximum_photosphere_oblate_radius_m = (
        surface.calibration.photosphere_point(
            surface.calibration.outer_radius_over_mass,
            UPPER,
        ).radius_over_mass
        * surface.metric.mass_m
    )
    if termination.capture_radius_m >= disk_isco_radius_m:
        raise ValueError(
            "capture worldtube must lie strictly inside the finite-thickness "
            "disk ISCO"
        )
    if termination.escape_radius_m <= maximum_photosphere_oblate_radius_m:
        raise ValueError(
            "escape worldtube must lie strictly outside the maximum "
            "finite-thickness photosphere oblate radius"
        )


def _validated_tolerance_policy(
    metric: KerrKerrSchildMetric,
    *,
    null_residual_limit: float,
    conserved_quantity_tolerance: float,
    surface_value_tolerance: float,
    recorded_path_absolute_tolerance: float,
    recorded_path_relative_tolerance: float,
    boundary_value_tolerance_m: float,
    emitter_event_tolerance_m: float,
) -> Mapping[str, Any]:
    maximum_path_absolute = (
        _MAXIMUM_RECORDED_PATH_ABSOLUTE_TOLERANCE_SCALE
        * max(1.0, metric.mass_m)
    )
    maximum_boundary = (
        _MAXIMUM_BOUNDARY_VALUE_TOLERANCE_OVER_MASS * metric.mass_m
    )
    maximum_emitter_event = (
        _MAXIMUM_EMITTER_EVENT_TOLERANCE_OVER_MASS * metric.mass_m
    )
    actual = {
        "boundaryValueToleranceM": boundary_value_tolerance_m,
        "conservedQuantityTolerance": conserved_quantity_tolerance,
        "emitterEventToleranceM": emitter_event_tolerance_m,
        "nullResidualLimit": null_residual_limit,
        "recordedPathAbsoluteTolerance": recorded_path_absolute_tolerance,
        "recordedPathRelativeTolerance": recorded_path_relative_tolerance,
        "surfaceValueTolerance": surface_value_tolerance,
    }
    maxima = {
        "boundaryValueToleranceM": maximum_boundary,
        "conservedQuantityTolerance": (
            _MAXIMUM_CONSERVED_QUANTITY_TOLERANCE
        ),
        "emitterEventToleranceM": maximum_emitter_event,
        "nullResidualLimit": _MAXIMUM_NULL_RESIDUAL_LIMIT,
        "recordedPathAbsoluteTolerance": maximum_path_absolute,
        "recordedPathRelativeTolerance": (
            _MAXIMUM_RECORDED_PATH_RELATIVE_TOLERANCE
        ),
        "surfaceValueTolerance": _MAXIMUM_SURFACE_VALUE_TOLERANCE,
    }
    positive_names = tuple(name for name in actual if name != "emitterEventToleranceM")
    if any(not math.isfinite(value) for value in actual.values()):
        raise ValueError("transfer tolerance policy contains a non-finite value")
    if any(actual[name] <= 0.0 for name in positive_names):
        raise ValueError("transfer tolerance policy requires positive limits")
    if emitter_event_tolerance_m < 0.0:
        raise ValueError("emitter event tolerance must be non-negative")
    for name, maximum in maxima.items():
        if actual[name] > maximum:
            raise ValueError(
                f"{name} exceeds the finite-thickness transfer policy maximum"
            )
    return {
        "actual": actual,
        "maxima": maxima,
        "policy": "caller may tighten but may not exceed implementation maxima",
    }


def _validate_model_context(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    escaped_spectrum: BuiltInEscapedObserverSpectrum,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    *,
    null_residual_limit: float,
    conserved_quantity_tolerance: float,
    surface_value_tolerance: float,
    recorded_path_absolute_tolerance: float,
    recorded_path_relative_tolerance: float,
    boundary_value_tolerance_m: float,
    emitter_event_tolerance_m: float,
) -> str:
    _validate_model_ownership(
        surface,
        disk,
        escaped_spectrum,
        termination,
        ray_options,
        surface_options,
    )
    tolerance_policy = _validated_tolerance_policy(
        surface.metric,
        null_residual_limit=null_residual_limit,
        conserved_quantity_tolerance=conserved_quantity_tolerance,
        surface_value_tolerance=surface_value_tolerance,
        recorded_path_absolute_tolerance=recorded_path_absolute_tolerance,
        recorded_path_relative_tolerance=recorded_path_relative_tolerance,
        boundary_value_tolerance_m=boundary_value_tolerance_m,
        emitter_event_tolerance_m=emitter_event_tolerance_m,
    )
    return _configuration_descriptor_json(
        surface,
        disk,
        escaped_spectrum,
        termination,
        tolerance_policy,
        _validated_trace_options_descriptor(
            surface.metric,
            ray_options,
            surface_options,
        ),
    )


def _validate_future_observer_velocity(
    metric: KerrKerrSchildMetric,
    observer_state: HamiltonianState,
    observer_four_velocity: Sequence[float],
) -> Vector4:
    velocity = _finite_four_velocity(observer_four_velocity)
    norm = bilinear(velocity, metric.sample(observer_state.event).covariant, velocity)
    if (
        velocity[0] <= 0.0
        or not math.isfinite(norm)
        or not _relative_close(norm, -1.0, _FOUR_VELOCITY_TOLERANCE)
    ):
        raise ValueError("observer_four_velocity must be future unit-timelike")
    return velocity


def _validate_observer_outside_photosphere(
    surface: KerrFiniteThicknessMultiSurface,
    observer_state: HamiltonianState,
    surface_value_tolerance: float,
) -> None:
    oblate = kerr_ks_event_to_oblate_meridional(
        surface.metric,
        observer_state.event,
    )
    rho = (
        oblate.radius_m
        * math.sin(oblate.theta_rad)
        / surface.metric.mass_m
    )
    calibration = surface.calibration
    if not calibration.contains_pseudo_cylindrical_radius(rho):
        return
    upper_value = surface.value(UPPER_SURFACE_ID, observer_state)
    lower_value = surface.value(LOWER_SURFACE_ID, observer_state)
    if (
        upper_value <= surface_value_tolerance
        and lower_value <= surface_value_tolerance
    ):
        raise ValueError(
            "observer lies on or inside the physical finite-thickness "
            "photosphere"
        )


def _validate_conserved_pair(
    metric: KerrKerrSchildMetric,
    first: HamiltonianState,
    second: HamiltonianState,
    tolerance: float,
) -> None:
    momentum_scale = max(
        *(abs(value) for value in first.covector),
        *(abs(value) for value in second.covector),
    )
    if not math.isfinite(momentum_scale) or momentum_scale <= 0.0:
        raise ValueError("photon covectors need a finite common affine scale")
    normalized_first = HamiltonianState(
        first.event,
        tuple(value / momentum_scale for value in first.covector),
    )
    normalized_second = HamiltonianState(
        second.event,
        tuple(value / momentum_scale for value in second.covector),
    )
    first_energy, first_lz = stationary_axisymmetric_constants(normalized_first)
    second_energy, second_lz = stationary_axisymmetric_constants(normalized_second)
    if not _relative_close(first_energy, second_energy, tolerance):
        raise ValueError("photon Killing energy is not conserved")
    if not _relative_close(first_lz, second_lz, tolerance, metric.mass_m):
        raise ValueError("photon axial angular momentum is not conserved")
    first_k = kerr_constants_of_motion(metric, normalized_first).carter_k
    second_k = kerr_constants_of_motion(metric, normalized_second).carter_k
    if not _relative_close(
        first_k,
        second_k,
        tolerance,
        metric.mass_m * metric.mass_m,
    ):
        raise ValueError("photon Carter constant is not conserved")


def _validate_recorded_ray(
    metric: KerrKerrSchildMetric,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    null_residual_limit: float,
    recorded_path_absolute_tolerance: float,
    recorded_path_relative_tolerance: float,
) -> tuple[RayPathSegment, ...]:
    if type(ray) is not RayTraceResult:
        raise TypeError("ray must be the exact RayTraceResult")
    if type(observer_initial_state) is not HamiltonianState:
        raise TypeError("observer_initial_state must be the exact HamiltonianState")
    if ray.outcome not in (OPAQUE_OUTCOME, "captured", "escaped"):
        raise ValueError("finite-thickness transfer requires a successful ray outcome")
    if ray.failure_reason is not None:
        raise ValueError("failed ray may not enter finite-thickness transfer")
    if not isinstance(ray.terminal_target_id, str) or not ray.terminal_target_id:
        raise ValueError("successful ray needs a terminal target id")
    if (
        type(ray.accepted_steps) is not int
        or ray.accepted_steps < 1
        or type(ray.rejected_steps) is not int
        or ray.rejected_steps < 0
    ):
        raise ValueError("ray step diagnostics are invalid")
    affine_length = _finite_number(ray.affine_length, "ray.affine_length")
    maximum_null = _finite_number(
        ray.maximum_null_residual,
        "ray.maximum_null_residual",
    )
    maximum_metric = _finite_number(
        ray.maximum_metric_interpolation_error,
        "ray.maximum_metric_interpolation_error",
    )
    if affine_length <= 0.0 or maximum_null < 0.0 or maximum_metric != 0.0:
        raise ValueError("exact-Kerr ray diagnostics are invalid")
    if maximum_null > null_residual_limit:
        raise ValueError("ray null residual exceeds the transfer limit")
    if not isinstance(ray.segments, tuple) or not ray.segments:
        raise ValueError("finite-thickness transfer requires a recorded ray")
    segments = ray.segments
    if len(segments) != ray.accepted_steps:
        raise ValueError("recorded segment count disagrees with accepted steps")
    if any(type(segment) is not RayPathSegment for segment in segments):
        raise TypeError("recorded ray contains a foreign segment type")
    if segments[0].start != observer_initial_state:
        raise ValueError("observer initial state does not own the recorded ray")
    if segments[-1].end != ray.terminal_state:
        raise ValueError("recorded ray does not end at its terminal state")
    if any(
        previous.end != current.start
        for previous, current in zip(segments, segments[1:])
    ):
        raise ValueError("recorded ray segments are not contiguous")
    if any(
        not math.isfinite(segment.affine_length) or segment.affine_length <= 0.0
        for segment in segments
    ):
        raise ValueError("recorded ray contains a non-positive segment")
    recorded_length = math.fsum(segment.affine_length for segment in segments)
    if not _relative_close(recorded_length, affine_length, 2.0e-13):
        raise ValueError("recorded ray length disagrees with its segments")

    actual_maximum_null = 0.0
    for segment in segments:
        for state in (segment.start, segment.midpoint, segment.end):
            if type(state) is not HamiltonianState:
                raise TypeError("recorded ray contains a foreign state type")
            residual = hamiltonian_null_residual(metric, state)
            if not math.isfinite(residual) or residual > null_residual_limit:
                raise ValueError("recorded ray state exceeds the null residual limit")
            actual_maximum_null = max(actual_maximum_null, residual)
        midpoint_residual = _finite_number(
            segment.midpoint_null_residual,
            "segment.midpoint_null_residual",
        )
        if midpoint_residual < 0.0 or midpoint_residual > null_residual_limit:
            raise ValueError("recorded midpoint null diagnostic is invalid")
    if actual_maximum_null > maximum_null * (1.0 + 8.0e-13) + 1.0e-15:
        raise ValueError("ray maximum null diagnostic understates recorded states")
    terminal_sampler = CertifiedRecordedPathSampler(
        metric,
        RecordedPathSamplingOptions(
            absolute_tolerance=recorded_path_absolute_tolerance,
            relative_tolerance=recorded_path_relative_tolerance,
            null_residual_limit=null_residual_limit,
            metric_interpolation_error_limit=1.0e-15,
            maximum_reintegrations=1,
        ),
    )
    try:
        terminal_sampler.sample(
            segments[-1],
            1.0,
            expected=ray.terminal_state,
            label="terminal recorded segment endpoint",
        )
    except RecordedPathSamplingError as error:
        raise ValueError(
            "terminal state is not bound to Hamiltonian reintegration of "
            f"its recorded segment: {error}"
        ) from error
    return segments


def _validate_terminal_worldtube(
    termination: KerrOblateTermination,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    boundary_value_tolerance_m: float,
) -> None:
    if termination.classify_initial(observer_initial_state) is not None:
        raise ValueError("observer starts on or beyond a terminal worldtube")
    terminal_radius = termination.radius(ray.terminal_state)
    if not math.isfinite(terminal_radius):
        raise ValueError("terminal Kerr oblate radius is not finite")
    if ray.outcome in ("captured", "escaped"):
        if ray.outcome == "captured":
            expected_target = termination.capture_target_id
            expected_radius = termination.capture_radius_m
        else:
            expected_target = termination.escape_target_id
            expected_radius = termination.escape_radius_m
        if ray.terminal_target_id != expected_target:
            raise ValueError("boundary ray target is not owned by its termination")
        if abs(terminal_radius - expected_radius) > boundary_value_tolerance_m:
            raise ValueError(
                "boundary terminal state is not on its authenticated worldtube"
            )
        previous_radius = termination.radius(ray.segments[-1].start)
        if ray.outcome == "captured" and previous_radius <= expected_radius:
            raise ValueError("capture segment does not approach from outside")
        if ray.outcome == "escaped" and previous_radius >= expected_radius:
            raise ValueError("escape segment does not approach from inside")
        return
    if not (
        termination.capture_radius_m + boundary_value_tolerance_m
        < terminal_radius
        < termination.escape_radius_m - boundary_value_tolerance_m
    ):
        raise ValueError(
            "disk-hit terminal state is not strictly between its worldtubes"
        )


def _validate_trace(
    surface: KerrFiniteThicknessMultiSurface,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    null_residual_limit: float,
    conserved_quantity_tolerance: float,
    surface_value_tolerance: float,
    recorded_path_absolute_tolerance: float,
    recorded_path_relative_tolerance: float,
) -> ClassifiedMultiInteriorSurfaceCrossing | None:
    trace = ray.multi_surface_trace
    if type(trace) is not MultiInteriorSurfaceTrace:
        raise ValueError("ray must carry the exact MultiInteriorSurfaceTrace")
    if (
        type(trace.base_subdivisions_per_step) is not int
        or trace.base_subdivisions_per_step < 2
        or trace.base_subdivisions_per_step % 2 != 0
    ):
        raise ValueError("multi-surface base subdivisions must be even N >= 2")
    if (
        type(trace.verification_subdivisions_per_step) is not int
        or trace.verification_subdivisions_per_step
        != 2 * trace.base_subdivisions_per_step
    ):
        raise ValueError("multi-surface verification subdivisions must equal 2N")
    if type(trace.topology_converged) is not bool:
        raise TypeError("multi-surface topology convergence flag must be exact bool")
    if not trace.topology_converged:
        raise ValueError("multi-surface topology is not converged")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            trace.maximum_probe_event_difference,
            trace.maximum_probe_covector_relative_difference,
        )
    ):
        raise ValueError("multi-surface probe convergence diagnostics are invalid")
    if (
        type(trace.probe_reintegrations) is not int
        or trace.probe_reintegrations < 0
        or type(trace.surface_value_evaluations) is not int
        or trace.surface_value_evaluations < 0
    ):
        raise ValueError("multi-surface work diagnostics are invalid")
    expected_ids = tuple(sorted(surface.surface_ids))
    if (
        type(trace.surface_ids) is not tuple
        or any(
            type(surface_id) is not str or not surface_id
            for surface_id in trace.surface_ids
        )
        or trace.surface_ids != tuple(sorted(trace.surface_ids))
        or len(set(trace.surface_ids)) != len(trace.surface_ids)
        or trace.surface_ids != expected_ids
    ):
        raise ValueError("multi-surface trace ids do not match the supplied adapter")

    path_sampler = CertifiedRecordedPathSampler(
        surface.metric,
        RecordedPathSamplingOptions(
            absolute_tolerance=recorded_path_absolute_tolerance,
            relative_tolerance=recorded_path_relative_tolerance,
            null_residual_limit=null_residual_limit,
            metric_interpolation_error_limit=1.0e-15,
            maximum_reintegrations=max(1, len(trace.crossings)),
        ),
    )

    segment_prefixes: list[float] = []
    prefix = 0.0
    for segment in ray.segments:
        segment_prefixes.append(prefix)
        prefix += segment.affine_length

    # Preflight the complete observer-to-source topology before trusting any
    # member enough to index a segment or invoke product classification.
    previous_affine = -math.inf
    terminal_seen = False
    for entry_index, entry in enumerate(trace.crossings):
        if type(entry) is not ClassifiedMultiInteriorSurfaceCrossing:
            raise TypeError("multi-surface trace contains a foreign crossing entry")
        if terminal_seen:
            raise ValueError("multi-surface trace continues after a terminal crossing")
        crossing = entry.crossing
        # Repeat the complete crossing ABI at this trust boundary.  In
        # particular, never allow Python's negative indexing semantics to
        # reinterpret a forged ``segment_index=-1`` as the final ray segment.
        if type(crossing) is not RecordedSurfaceCrossing:
            raise TypeError("multi-surface trace contains a foreign crossing type")
        if type(crossing.state) is not HamiltonianState:
            raise TypeError("surface crossing contains a foreign state type")
        if (
            type(crossing.segment_index) is not int
            or crossing.segment_index < 0
            or crossing.segment_index >= len(ray.segments)
        ):
            raise ValueError("surface crossing segment index is outside the ray")
        if (
            not math.isfinite(crossing.segment_affine_length)
            or crossing.segment_affine_length < 0.0
            or not math.isfinite(crossing.ray_affine_length)
            or crossing.ray_affine_length < 0.0
        ):
            raise ValueError(
                "surface crossing affine diagnostics are negative or invalid"
            )
        if (
            type(crossing.orientation) is not int
            or crossing.orientation not in (-1, 1)
        ):
            raise ValueError("surface crossing orientation must be integer -1 or +1")
        if (
            not math.isfinite(crossing.surface_value)
            or not math.isfinite(crossing.bracket_affine_width)
            or crossing.bracket_affine_width < 0.0
            or type(crossing.iterations) is not int
            or crossing.iterations < 0
        ):
            raise ValueError("surface crossing root diagnostics are invalid")
        if crossing.ray_affine_length <= previous_affine:
            raise ValueError(
                "surface crossings must be strictly observer-to-source ordered"
            )
        if entry.decision.terminates and entry_index != len(trace.crossings) - 1:
            raise ValueError("only the final multi-surface crossing may terminate")
        terminal_seen = entry.decision.terminates
        previous_affine = crossing.ray_affine_length

    terminal: ClassifiedMultiInteriorSurfaceCrossing | None = None
    for entry_index, entry in enumerate(trace.crossings):
        crossing = entry.crossing
        segment = ray.segments[crossing.segment_index]
        if crossing.segment_affine_length > segment.affine_length:
            raise ValueError("surface crossing lies beyond its recorded segment")
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
            raise ValueError("surface crossing affine diagnostics are inconsistent")

        endpoint_tolerance = max(
            128.0 * math.ulp(max(1.0, segment.affine_length)),
            1.0e-14 * max(1.0, segment.affine_length),
        )
        sample_fraction: float | None = None
        if crossing.segment_affine_length <= endpoint_tolerance:
            if crossing.state != segment.start:
                raise ValueError(
                    "near-start surface crossing state is not the segment start"
                )
        elif (
            segment.affine_length - crossing.segment_affine_length
            <= endpoint_tolerance
        ):
            if crossing.state != segment.end:
                raise ValueError(
                    "near-end surface crossing state is not the segment end"
                )
            # Equality to a recorded endpoint is not provenance by itself:
            # independently reconstruct the accepted step from its start.
            sample_fraction = 1.0
        else:
            sample_fraction = (
                crossing.segment_affine_length / segment.affine_length
            )
        if sample_fraction is not None:
            try:
                path_sampler.sample(
                    segment,
                    sample_fraction,
                    expected=crossing.state,
                    label=(
                        f"multi-surface crossing {entry_index} "
                        f"({entry.surface_id})"
                    ),
                )
            except RecordedPathSamplingError as error:
                raise ValueError(
                    "surface crossing state is not bound to its claimed "
                    f"recorded segment: {error}"
                ) from error
        recomputed_value = surface.value(entry.surface_id, crossing.state)
        if (
            abs(recomputed_value) > surface_value_tolerance
            or not math.isclose(
                crossing.surface_value,
                recomputed_value,
                rel_tol=2.0e-13,
                abs_tol=surface_value_tolerance * 1.0e-6,
            )
        ):
            raise ValueError("surface crossing does not lie on its declared face")
        if entry.decision != surface.classify(entry.surface_id, crossing):
            raise ValueError("surface crossing classification is not adapter-owned")
        _validate_conserved_pair(
            surface.metric,
            observer_initial_state,
            crossing.state,
            conserved_quantity_tolerance,
        )
        if entry.decision.terminates:
            terminal = entry

    if ray.outcome == OPAQUE_OUTCOME:
        if terminal is None or terminal is not trace.crossings[-1]:
            raise ValueError("disk-hit ray lacks its final terminal face crossing")
        if (
            terminal.decision.outcome != ray.outcome
            or terminal.decision.target_id != ray.terminal_target_id
            or terminal.crossing.state != ray.terminal_state
            or not math.isclose(
                terminal.crossing.ray_affine_length,
                ray.affine_length,
                rel_tol=2.0e-13,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("terminal face crossing does not own the ray outcome")
    elif terminal is not None:
        raise ValueError("boundary ray may not hide a terminal disk crossing")
    return terminal


def _frequency_shift_and_projection(
    metric: KerrKerrSchildMetric,
    observer_state: HamiltonianState,
    observer_four_velocity: Vector4,
    emitter_state: HamiltonianState,
    emitter: KerrFiniteThicknessFaceEmitter,
    *,
    null_residual_limit: float,
    conserved_quantity_tolerance: float,
    event_tolerance_m: float,
) -> tuple[float, float, float, KerrFiniteThicknessPhotonProjection]:
    _validate_conserved_pair(
        metric,
        observer_state,
        emitter_state,
        conserved_quantity_tolerance,
    )
    common_scale = max(
        *(abs(value) for value in observer_state.covector),
        *(abs(value) for value in emitter_state.covector),
    )
    if not math.isfinite(common_scale) or common_scale <= 0.0:
        raise ValueError("photon covectors need a finite common affine scale")
    observer_covector = tuple(value / common_scale for value in observer_state.covector)
    emitter_covector = tuple(value / common_scale for value in emitter_state.covector)
    observer_frequency = math.fsum(
        observer_four_velocity[index] * observer_covector[index]
        for index in range(4)
    )
    emitter_frequency = math.fsum(
        emitter.four_velocity[index] * emitter_covector[index]
        for index in range(4)
    )
    if not math.isfinite(observer_frequency) or observer_frequency <= 0.0:
        raise ValueError("past-directed observer photon frequency must be positive")
    if not math.isfinite(emitter_frequency) or emitter_frequency <= 0.0:
        raise ValueError("past-directed emitter photon frequency must be positive")
    shift = observer_frequency / emitter_frequency
    if not math.isfinite(shift) or shift <= 0.0:
        raise KerrFiniteThicknessTransferError("frequency shift g is invalid")
    projection = emitter.project_past_directed_photon(
        emitter_state,
        null_residual_limit=null_residual_limit,
        event_tolerance_m=event_tolerance_m,
        backside_policy="reject",
    )
    if (
        projection.face_classification != "outgoing"
        or projection.outgoing_cosine <= 0.0
    ):
        raise KerrFiniteThicknessTransferError(
            "finite-thickness emission requires strictly positive front-face mu"
        )
    raw_emitter_frequency = math.fsum(
        emitter.four_velocity[index] * emitter_state.covector[index]
        for index in range(4)
    )
    if not _relative_close(
        projection.local_frequency,
        raw_emitter_frequency,
        2.0e-13,
    ):
        raise KerrFiniteThicknessTransferError(
            "face projection and invariant emitter frequency disagree"
        )
    return shift, observer_frequency, emitter_frequency, projection


def _crossing_oblate_and_rho(
    metric: KerrKerrSchildMetric,
    entry: ClassifiedMultiInteriorSurfaceCrossing,
) -> tuple[KerrOblateEvent, float, PhotosphereFace, str]:
    face, target = _surface_face_and_target(entry.surface_id)
    if entry.decision.target_id != target:
        raise ValueError("terminal face id and target id disagree")
    oblate = kerr_ks_event_to_oblate(metric, entry.crossing.state.event)
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / metric.mass_m
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("terminal pseudo-cylindrical radius is invalid")
    return oblate, rho, face, target


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessSpectrumResult:
    """Immutable, self-revalidating scalar result for one certified ray."""

    surface: KerrFiniteThicknessMultiSurface
    disk: StationaryNovikovThorneDisk
    ray: RayTraceResult
    termination: KerrOblateTermination
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    observer_initial_state: HamiltonianState
    observer_four_velocity: Vector4
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum
    observer_frequencies_hz: tuple[float, ...]
    observed_specific_intensities_nu: tuple[float, ...]
    source_kind: FiniteThicknessSourceKind
    transfer_configuration_sha256: str
    escape_spectrum_descriptor_sha256: str
    null_residual_limit: float
    conserved_quantity_tolerance: float
    surface_value_tolerance: float
    recorded_path_absolute_tolerance: float
    recorded_path_relative_tolerance: float
    boundary_value_tolerance_m: float
    emitter_event_tolerance_m: float
    terminal_surface_entry: ClassifiedMultiInteriorSurfaceCrossing | None = None
    face: PhotosphereFace | None = None
    pseudo_cylindrical_radius_over_mass: float | None = None
    equatorial_reference_radius_m: float | None = None
    emitter: KerrFiniteThicknessFaceEmitter | None = None
    emitter_descriptor_sha256: str | None = None
    frequency_shift_g: float | None = None
    normalized_observer_frequency: float | None = None
    normalized_emitter_frequency: float | None = None
    photon_projection: KerrFiniteThicknessPhotonProjection | None = None
    emitted_frequencies_hz: tuple[float, ...] | None = None
    isotropic_emitted_specific_intensities_nu: tuple[float, ...] | None = None
    angular_emission_multiplier: float | None = None
    emitted_specific_intensities_nu: tuple[float, ...] | None = None
    _replay_certificate: InitVar[object | None] = None

    def __post_init__(self, _replay_certificate: object | None) -> None:
        _validate_model_ownership(
            self.surface,
            self.disk,
            self.escaped_observer_spectrum,
            self.termination,
            self.ray_options,
            self.surface_options,
        )
        configuration_sha = _validate_sha256(
            self.transfer_configuration_sha256,
            "transfer_configuration_sha256",
        )
        escape_json = _validate_builtin_escape_spectrum(
            self.escaped_observer_spectrum
        )
        expected_escape_sha = _sha256_text(escape_json)
        escape_sha = _validate_sha256(
            self.escape_spectrum_descriptor_sha256,
            "escape_spectrum_descriptor_sha256",
        )
        if escape_sha != expected_escape_sha:
            raise ValueError("escaped spectrum provenance hash is stale")

        null_limit = _finite_number(
            self.null_residual_limit,
            "null_residual_limit",
        )
        constant_tolerance = _finite_number(
            self.conserved_quantity_tolerance,
            "conserved_quantity_tolerance",
        )
        surface_tolerance = _finite_number(
            self.surface_value_tolerance,
            "surface_value_tolerance",
        )
        path_absolute_tolerance = _finite_number(
            self.recorded_path_absolute_tolerance,
            "recorded_path_absolute_tolerance",
        )
        path_relative_tolerance = _finite_number(
            self.recorded_path_relative_tolerance,
            "recorded_path_relative_tolerance",
        )
        boundary_tolerance = _finite_number(
            self.boundary_value_tolerance_m,
            "boundary_value_tolerance_m",
        )
        event_tolerance = _finite_number(
            self.emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
        if (
            null_limit <= 0.0
            or constant_tolerance <= 0.0
            or surface_tolerance <= 0.0
            or path_absolute_tolerance <= 0.0
            or path_relative_tolerance <= 0.0
            or boundary_tolerance <= 0.0
            or event_tolerance < 0.0
        ):
            raise ValueError("transfer tolerances are invalid")
        configuration_json = _validate_model_context(
            self.surface,
            self.disk,
            self.escaped_observer_spectrum,
            self.termination,
            self.ray_options,
            self.surface_options,
            null_residual_limit=null_limit,
            conserved_quantity_tolerance=constant_tolerance,
            surface_value_tolerance=surface_tolerance,
            recorded_path_absolute_tolerance=path_absolute_tolerance,
            recorded_path_relative_tolerance=path_relative_tolerance,
            boundary_value_tolerance_m=boundary_tolerance,
            emitter_event_tolerance_m=event_tolerance,
        )
        expected_configuration_sha = _sha256_text(configuration_json)
        if configuration_sha != expected_configuration_sha:
            raise ValueError("transfer configuration provenance hash is stale")
        frequencies = _positive_frequencies(self.observer_frequencies_hz)
        intensities = tuple(
            _finite_number(value, f"observed_specific_intensities_nu[{index}]")
            for index, value in enumerate(self.observed_specific_intensities_nu)
        )
        if len(intensities) != len(frequencies) or any(
            value < 0.0 for value in intensities
        ):
            raise ValueError("observer-frame spectrum is malformed")
        observer_velocity = _validate_future_observer_velocity(
            self.surface.metric,
            self.observer_initial_state,
            self.observer_four_velocity,
        )
        _validate_observer_outside_photosphere(
            self.surface,
            self.observer_initial_state,
            surface_tolerance,
        )
        _validate_recorded_ray(
            self.surface.metric,
            self.ray,
            self.observer_initial_state,
            null_limit,
            path_absolute_tolerance,
            path_relative_tolerance,
        )
        _validate_conserved_pair(
            self.surface.metric,
            self.observer_initial_state,
            self.ray.terminal_state,
            constant_tolerance,
        )
        _validate_terminal_worldtube(
            self.termination,
            self.ray,
            self.observer_initial_state,
            boundary_tolerance,
        )
        terminal = _validate_trace(
            self.surface,
            self.ray,
            self.observer_initial_state,
            null_limit,
            constant_tolerance,
            surface_tolerance,
            path_absolute_tolerance,
            path_relative_tolerance,
        )
        if _replay_certificate is None:
            replay_certificate = _issue_replay_certificate(
                self.surface,
                self.termination,
                self.observer_initial_state,
                self.ray,
                self.ray_options,
                self.surface_options,
            )
        else:
            replay_certificate = _replay_certificate
            _require_replay_certificate(
                replay_certificate,
                self.surface,
                self.termination,
                self.observer_initial_state,
                self.ray,
                self.ray_options,
                self.surface_options,
            )

        disk_fields = (
            self.terminal_surface_entry,
            self.face,
            self.pseudo_cylindrical_radius_over_mass,
            self.equatorial_reference_radius_m,
            self.emitter,
            self.emitter_descriptor_sha256,
            self.frequency_shift_g,
            self.normalized_observer_frequency,
            self.normalized_emitter_frequency,
            self.photon_projection,
            self.emitted_frequencies_hz,
            self.isotropic_emitted_specific_intensities_nu,
            self.angular_emission_multiplier,
            self.emitted_specific_intensities_nu,
        )
        if self.ray.outcome != OPAQUE_OUTCOME:
            expected_kind = f"{self.ray.outcome}-boundary"
            if self.source_kind != expected_kind:
                raise ValueError("boundary source kind disagrees with the ray outcome")
            if any(value is not None for value in disk_fields):
                raise ValueError("boundary result may not carry disk-hit diagnostics")
            if self.ray.outcome == "captured":
                expected_intensities = tuple(0.0 for _ in frequencies)
            else:
                expected_intensities = tuple(
                    _finite_number(
                        self.escaped_observer_spectrum(
                            self.ray.terminal_state,
                            frequency,
                            self.ray.terminal_target_id,  # type: ignore[arg-type]
                        ),
                        f"escaped observer intensity at bin {index}",
                    )
                    for index, frequency in enumerate(frequencies)
                )
                if any(value < 0.0 for value in expected_intensities):
                    raise ValueError("escaped observer spectrum returned negative I_nu")
            if intensities != expected_intensities:
                raise ValueError(
                    "boundary spectrum is not the configured closed result"
                )
        else:
            if self.source_kind != "finite-thickness-disk":
                raise ValueError("opaque finite-thickness ray requires a disk source")
            if terminal is None or self.terminal_surface_entry is not terminal:
                raise ValueError("result does not retain its terminal trace entry")
            oblate, rho, face, _target = _crossing_oblate_and_rho(
                self.surface.metric,
                terminal,
            )
            reference_radius = rho * self.surface.metric.mass_m
            if self.face != face:
                raise ValueError(
                    "result face disagrees with terminal stable surface id"
                )
            stored_rho = _finite_number(
                self.pseudo_cylindrical_radius_over_mass,
                "pseudo_cylindrical_radius_over_mass",
            )
            stored_radius = _finite_number(
                self.equatorial_reference_radius_m,
                "equatorial_reference_radius_m",
            )
            if not _relative_close(stored_rho, rho, 2.0e-13):
                raise ValueError("stored pseudo-cylindrical radius is stale")
            if not _relative_close(stored_radius, reference_radius, 2.0e-13):
                raise ValueError("stored equatorial NT reference radius is stale")
            if type(self.emitter) is not KerrFiniteThicknessFaceEmitter:
                raise TypeError("disk result needs the exact finite-thickness emitter")
            expected_emitter = KerrFiniteThicknessFaceEmitter(
                metric=self.surface.metric,
                calibration=self.surface.calibration,
                pseudo_cylindrical_radius_over_mass=rho,
                face=face,
                phi_ks_rad=oblate.phi_ks_rad,
                coordinate_time_m=oblate.coordinate_time_m,
            )
            if self.emitter != expected_emitter:
                raise ValueError("finite-thickness emitter is foreign or stale")
            emitter_sha = _validate_sha256(
                self.emitter_descriptor_sha256,
                "emitter_descriptor_sha256",
            )
            if emitter_sha != expected_emitter.model_descriptor_sha256:
                raise ValueError("finite-thickness emitter provenance hash is stale")
            shift, observer_frequency, emitter_frequency, projection = (
                _frequency_shift_and_projection(
                    self.surface.metric,
                    self.observer_initial_state,
                    observer_velocity,
                    terminal.crossing.state,
                    expected_emitter,
                    null_residual_limit=null_limit,
                    conserved_quantity_tolerance=constant_tolerance,
                    event_tolerance_m=event_tolerance,
                )
            )
            stored_shift = _finite_number(self.frequency_shift_g, "frequency_shift_g")
            stored_observer_frequency = _finite_number(
                self.normalized_observer_frequency,
                "normalized_observer_frequency",
            )
            stored_emitter_frequency = _finite_number(
                self.normalized_emitter_frequency,
                "normalized_emitter_frequency",
            )
            if (
                stored_shift <= 0.0
                or stored_observer_frequency <= 0.0
                or stored_emitter_frequency <= 0.0
                or not _relative_close(stored_shift, shift, 2.0e-13)
                or not _relative_close(
                    stored_observer_frequency,
                    observer_frequency,
                    2.0e-13,
                )
                or not _relative_close(
                    stored_emitter_frequency,
                    emitter_frequency,
                    2.0e-13,
                )
            ):
                raise ValueError("stored invariant frequency diagnostics are stale")
            if self.photon_projection != projection:
                raise ValueError("stored face photon projection is stale")

            emitted_frequencies = tuple(
                _finite_number(value, f"emitted_frequencies_hz[{index}]")
                for index, value in enumerate(self.emitted_frequencies_hz or ())
            )
            expected_emitted_frequencies = tuple(
                frequency / shift for frequency in frequencies
            )
            if len(emitted_frequencies) != len(frequencies) or any(
                not _relative_close(actual, expected, 2.0e-13)
                for actual, expected in zip(
                    emitted_frequencies,
                    expected_emitted_frequencies,
                )
            ):
                raise ValueError("emitter-frame frequencies are inconsistent with g")
            isotropic = tuple(
                _finite_number(
                    value,
                    f"isotropic_emitted_specific_intensities_nu[{index}]",
                )
                for index, value in enumerate(
                    self.isotropic_emitted_specific_intensities_nu or ()
                )
            )
            expected_isotropic = tuple(
                self.disk.emitted_specific_intensity_nu(reference_radius, frequency)
                for frequency in expected_emitted_frequencies
            )
            if len(isotropic) != len(frequencies) or any(
                not _relative_close(actual, expected, 8.0e-13, 1.0e-300)
                for actual, expected in zip(isotropic, expected_isotropic)
            ):
                raise ValueError("isotropic equatorial-NT-at-rho spectrum is stale")
            multiplier = _finite_number(
                self.angular_emission_multiplier,
                "angular_emission_multiplier",
            )
            expected_multiplier = _ANGULAR_LAW.intensity_multiplier(
                projection.outgoing_cosine
            )
            if multiplier <= 0.0 or not _relative_close(
                multiplier,
                expected_multiplier,
                2.0e-13,
            ):
                raise ValueError("KERRBB D20 angular multiplier is stale")
            emitted = tuple(
                _finite_number(value, f"emitted_specific_intensities_nu[{index}]")
                for index, value in enumerate(
                    self.emitted_specific_intensities_nu or ()
                )
            )
            expected_emitted = tuple(value * multiplier for value in isotropic)
            shift_cubed = shift * shift * shift
            expected_observed = tuple(
                shift_cubed * value for value in expected_emitted
            )
            if (
                not math.isfinite(shift_cubed)
                or shift_cubed <= 0.0
                or len(emitted) != len(frequencies)
                or any(
                    not _relative_close(actual, expected, 8.0e-13, 1.0e-300)
                    for actual, expected in zip(emitted, expected_emitted)
                )
                or any(
                    not _relative_close(actual, expected, 8.0e-13, 1.0e-300)
                    for actual, expected in zip(intensities, expected_observed)
                )
            ):
                raise ValueError("finite-thickness spectrum violates I_nu/nu^3")

        object.__setattr__(self, "observer_four_velocity", observer_velocity)
        object.__setattr__(self, "observer_frequencies_hz", frequencies)
        object.__setattr__(self, "observed_specific_intensities_nu", intensities)
        object.__setattr__(self, "transfer_configuration_sha256", configuration_sha)
        object.__setattr__(self, "escape_spectrum_descriptor_sha256", escape_sha)
        object.__setattr__(self, "null_residual_limit", null_limit)
        object.__setattr__(
            self,
            "conserved_quantity_tolerance",
            constant_tolerance,
        )
        object.__setattr__(self, "surface_value_tolerance", surface_tolerance)
        object.__setattr__(
            self,
            "recorded_path_absolute_tolerance",
            path_absolute_tolerance,
        )
        object.__setattr__(
            self,
            "recorded_path_relative_tolerance",
            path_relative_tolerance,
        )
        object.__setattr__(
            self,
            "boundary_value_tolerance_m",
            boundary_tolerance,
        )
        object.__setattr__(self, "emitter_event_tolerance_m", event_tolerance)
        # Close the time-of-check/time-of-use window across all scalar and
        # provenance recomputation.  This repeats only immutable comparisons,
        # never the geodesic integration authenticated by the certificate.
        _require_replay_certificate(
            replay_certificate,
            self.surface,
            self.termination,
            self.observer_initial_state,
            self.ray,
            self.ray_options,
            self.surface_options,
        )


def _transfer_kerr_finite_thickness_spectrum_certified(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    observer_four_velocity: Sequence[float],
    observer_frequencies_hz: Sequence[float],
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum,
    null_residual_limit: float = _DEFAULT_NULL_RESIDUAL_LIMIT,
    conserved_quantity_tolerance: float = _DEFAULT_CONSERVED_QUANTITY_TOLERANCE,
    surface_value_tolerance: float = _DEFAULT_SURFACE_VALUE_TOLERANCE,
    recorded_path_absolute_tolerance: float = (
        _DEFAULT_RECORDED_PATH_ABSOLUTE_TOLERANCE
    ),
    recorded_path_relative_tolerance: float = (
        _DEFAULT_RECORDED_PATH_RELATIVE_TOLERANCE
    ),
    boundary_value_tolerance_m: float | None = None,
    emitter_event_tolerance_m: float | None = None,
) -> tuple[KerrFiniteThicknessSpectrumResult, object]:
    """Compose one ray and return its process-local replay capability.

    The ray must already contain a converged ``MultiInteriorSurfaceTrace``.
    A deterministic full trace replay authenticates that no earlier crossing
    was omitted or altered.  The terminal stable surface id selects the
    upper/lower face, and the same stored past-directed photon is used for both
    invariant frequency shift and the signed face projection.
    """

    _validate_model_ownership(
        surface,
        disk,
        escaped_observer_spectrum,
        termination,
        ray_options,
        surface_options,
    )
    frequencies = _positive_frequencies(observer_frequencies_hz)
    null_limit = _finite_number(null_residual_limit, "null_residual_limit")
    constant_tolerance = _finite_number(
        conserved_quantity_tolerance,
        "conserved_quantity_tolerance",
    )
    surface_tolerance = _finite_number(
        surface_value_tolerance,
        "surface_value_tolerance",
    )
    path_absolute_tolerance = _finite_number(
        recorded_path_absolute_tolerance,
        "recorded_path_absolute_tolerance",
    )
    path_relative_tolerance = _finite_number(
        recorded_path_relative_tolerance,
        "recorded_path_relative_tolerance",
    )
    boundary_tolerance = (
        2.0e-8 * surface.metric.mass_m
        if boundary_value_tolerance_m is None
        else _finite_number(
            boundary_value_tolerance_m,
            "boundary_value_tolerance_m",
        )
    )
    event_tolerance = (
        1.0e-8 * surface.metric.mass_m
        if emitter_event_tolerance_m is None
        else _finite_number(
            emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
    )
    if (
        null_limit <= 0.0
        or constant_tolerance <= 0.0
        or surface_tolerance <= 0.0
        or path_absolute_tolerance <= 0.0
        or path_relative_tolerance <= 0.0
        or boundary_tolerance <= 0.0
        or event_tolerance < 0.0
    ):
        raise ValueError("transfer tolerances are invalid")
    configuration_json = _validate_model_context(
        surface,
        disk,
        escaped_observer_spectrum,
        termination,
        ray_options,
        surface_options,
        null_residual_limit=null_limit,
        conserved_quantity_tolerance=constant_tolerance,
        surface_value_tolerance=surface_tolerance,
        recorded_path_absolute_tolerance=path_absolute_tolerance,
        recorded_path_relative_tolerance=path_relative_tolerance,
        boundary_value_tolerance_m=boundary_tolerance,
        emitter_event_tolerance_m=event_tolerance,
    )
    observer_velocity = _validate_future_observer_velocity(
        surface.metric,
        observer_initial_state,
        observer_four_velocity,
    )
    _validate_observer_outside_photosphere(
        surface,
        observer_initial_state,
        surface_tolerance,
    )
    _validate_recorded_ray(
        surface.metric,
        ray,
        observer_initial_state,
        null_limit,
        path_absolute_tolerance,
        path_relative_tolerance,
    )
    _validate_conserved_pair(
        surface.metric,
        observer_initial_state,
        ray.terminal_state,
        constant_tolerance,
    )
    _validate_terminal_worldtube(
        termination,
        ray,
        observer_initial_state,
        boundary_tolerance,
    )
    terminal = _validate_trace(
        surface,
        ray,
        observer_initial_state,
        null_limit,
        constant_tolerance,
        surface_tolerance,
        path_absolute_tolerance,
        path_relative_tolerance,
    )
    replay_certificate = _issue_replay_certificate(
        surface,
        termination,
        observer_initial_state,
        ray,
        ray_options,
        surface_options,
    )
    configuration_sha = _sha256_text(configuration_json)
    escape_sha = _sha256_text(
        _validate_builtin_escape_spectrum(escaped_observer_spectrum)
    )

    common = {
        "surface": surface,
        "disk": disk,
        "ray": ray,
        "termination": termination,
        "ray_options": ray_options,
        "surface_options": surface_options,
        "observer_initial_state": observer_initial_state,
        "observer_four_velocity": observer_velocity,
        "escaped_observer_spectrum": escaped_observer_spectrum,
        "observer_frequencies_hz": frequencies,
        "transfer_configuration_sha256": configuration_sha,
        "escape_spectrum_descriptor_sha256": escape_sha,
        "null_residual_limit": null_limit,
        "conserved_quantity_tolerance": constant_tolerance,
        "surface_value_tolerance": surface_tolerance,
        "recorded_path_absolute_tolerance": path_absolute_tolerance,
        "recorded_path_relative_tolerance": path_relative_tolerance,
        "boundary_value_tolerance_m": boundary_tolerance,
        "emitter_event_tolerance_m": event_tolerance,
        "_replay_certificate": replay_certificate,
    }
    if ray.outcome == "captured":
        result = KerrFiniteThicknessSpectrumResult(
            **common,
            observed_specific_intensities_nu=tuple(0.0 for _ in frequencies),
            source_kind="captured-boundary",
        )
        return result, replay_certificate
    if ray.outcome == "escaped":
        intensities = tuple(
            _finite_number(
                escaped_observer_spectrum(
                    ray.terminal_state,
                    frequency,
                    ray.terminal_target_id,  # type: ignore[arg-type]
                ),
                f"escaped observer intensity at bin {index}",
            )
            for index, frequency in enumerate(frequencies)
        )
        if any(value < 0.0 for value in intensities):
            raise ValueError("escaped observer spectrum returned negative I_nu")
        result = KerrFiniteThicknessSpectrumResult(
            **common,
            observed_specific_intensities_nu=intensities,
            source_kind="escaped-boundary",
        )
        return result, replay_certificate
    if terminal is None:
        raise KerrFiniteThicknessTransferError("opaque disk ray lacks a terminal face")

    oblate, rho, face, _target = _crossing_oblate_and_rho(surface.metric, terminal)
    emitter = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=rho,
        face=face,
        phi_ks_rad=oblate.phi_ks_rad,
        coordinate_time_m=oblate.coordinate_time_m,
    )
    shift, observer_frequency, emitter_frequency, projection = (
        _frequency_shift_and_projection(
            surface.metric,
            observer_initial_state,
            observer_velocity,
            terminal.crossing.state,
            emitter,
            null_residual_limit=null_limit,
            conserved_quantity_tolerance=constant_tolerance,
            event_tolerance_m=event_tolerance,
        )
    )
    emitted_frequencies = tuple(frequency / shift for frequency in frequencies)
    if any(not math.isfinite(value) or value <= 0.0 for value in emitted_frequencies):
        raise KerrFiniteThicknessTransferError("emitter-frame frequency is invalid")
    reference_radius = rho * surface.metric.mass_m
    isotropic = tuple(
        disk.emitted_specific_intensity_nu(reference_radius, frequency)
        for frequency in emitted_frequencies
    )
    if any(not math.isfinite(value) or value < 0.0 for value in isotropic):
        raise KerrFiniteThicknessTransferError("equatorial NT reference is invalid")
    multiplier = _ANGULAR_LAW.intensity_multiplier(projection.outgoing_cosine)
    emitted = tuple(value * multiplier for value in isotropic)
    shift_cubed = shift * shift * shift
    observed = tuple(shift_cubed * value for value in emitted)
    if (
        not math.isfinite(shift_cubed)
        or shift_cubed <= 0.0
        or any(
            not math.isfinite(value) or value < 0.0
            for value in (*emitted, *observed)
        )
    ):
        raise KerrFiniteThicknessTransferError("finite-thickness I_nu overflowed")
    result = KerrFiniteThicknessSpectrumResult(
        **common,
        observed_specific_intensities_nu=observed,
        source_kind="finite-thickness-disk",
        terminal_surface_entry=terminal,
        face=face,
        pseudo_cylindrical_radius_over_mass=rho,
        equatorial_reference_radius_m=reference_radius,
        emitter=emitter,
        emitter_descriptor_sha256=emitter.model_descriptor_sha256,
        frequency_shift_g=shift,
        normalized_observer_frequency=observer_frequency,
        normalized_emitter_frequency=emitter_frequency,
        photon_projection=projection,
        emitted_frequencies_hz=emitted_frequencies,
        isotropic_emitted_specific_intensities_nu=isotropic,
        angular_emission_multiplier=multiplier,
        emitted_specific_intensities_nu=emitted,
    )
    return result, replay_certificate


def transfer_kerr_finite_thickness_spectrum(
    surface: KerrFiniteThicknessMultiSurface,
    disk: StationaryNovikovThorneDisk,
    ray: RayTraceResult,
    observer_initial_state: HamiltonianState,
    observer_four_velocity: Sequence[float],
    observer_frequencies_hz: Sequence[float],
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    escaped_observer_spectrum: BuiltInEscapedObserverSpectrum,
    null_residual_limit: float = _DEFAULT_NULL_RESIDUAL_LIMIT,
    conserved_quantity_tolerance: float = _DEFAULT_CONSERVED_QUANTITY_TOLERANCE,
    surface_value_tolerance: float = _DEFAULT_SURFACE_VALUE_TOLERANCE,
    recorded_path_absolute_tolerance: float = (
        _DEFAULT_RECORDED_PATH_ABSOLUTE_TOLERANCE
    ),
    recorded_path_relative_tolerance: float = (
        _DEFAULT_RECORDED_PATH_RELATIVE_TOLERANCE
    ),
    boundary_value_tolerance_m: float | None = None,
    emitter_event_tolerance_m: float | None = None,
) -> KerrFiniteThicknessSpectrumResult:
    """Compose one untrusted ray after exactly one deterministic full replay."""

    result, _certificate = _transfer_kerr_finite_thickness_spectrum_certified(
        surface,
        disk,
        ray,
        observer_initial_state,
        observer_four_velocity,
        observer_frequencies_hz,
        termination=termination,
        ray_options=ray_options,
        surface_options=surface_options,
        escaped_observer_spectrum=escaped_observer_spectrum,
        null_residual_limit=null_residual_limit,
        conserved_quantity_tolerance=conserved_quantity_tolerance,
        surface_value_tolerance=surface_value_tolerance,
        recorded_path_absolute_tolerance=recorded_path_absolute_tolerance,
        recorded_path_relative_tolerance=recorded_path_relative_tolerance,
        boundary_value_tolerance_m=boundary_value_tolerance_m,
        emitter_event_tolerance_m=emitter_event_tolerance_m,
    )
    return result


__all__ = (
    "BuiltInEscapedObserverSpectrum",
    "FiniteThicknessSourceKind",
    "IMPLEMENTATION_ID",
    "KerrFiniteThicknessSpectrumResult",
    "KerrFiniteThicknessTransferError",
    "SCIENTIFIC_STATUS",
    "transfer_kerr_finite_thickness_spectrum",
)
