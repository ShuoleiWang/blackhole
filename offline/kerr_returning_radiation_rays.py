"""Deterministic one-direction Kerr returning-radiation ray primitive.

This module connects the authenticated finite-height local launch frame to the
exact four-dimensional Kerr Hamiltonian tracer.  It deliberately stops at one
local emission direction: it classifies that photon's first subsequent fate
and, for a front-face return, computes the local emitter-to-receiver frequency
ratio and the corresponding bolometric ``g^4`` factor.

The initial event lies exactly on the emitting photosphere.  Moving it by an
epsilon would change both the geodesic and its provenance, while feeding the
unmodified zero directly to the generic accepted-step surface scanner would
correctly fail as an unbracketed endpoint contact.  The tracer's public,
authenticated initial-contact declaration therefore assigns only affine zero
to the outward-positive side of the emitting face.  Every positive-affine
probe uses the real signed surface value.  A very small subsequent root remains
rejected as unresolved initial contact; roundoff is never promoted to flux.

This is not a returning-radiation matrix, a quadrature rule, a stress-work
term, a KERRBB implementation, a solved atmosphere, or GRMHD.  In particular,
``receiver_incidence_weighted_g4`` still lacks an emission angular law, solid
angle/area Jacobian, annulus quadrature weight, and spectral redistribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import hmac
import json
import math
import os
import threading
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping
import weakref

from offline.geodesic import (
    HamiltonianState,
    InitialMultiSurfaceContact,
    InteriorSurfaceDecision,
    RayTraceOptions,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_ks_event_to_oblate,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    PhotosphereFace,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)


IMPLEMENTATION_ID: Final = "finite-thickness-kerr-returning-ray/v1"

RETURNED_OUTCOME: Final = "returned-to-finite-thickness-photosphere"
PLUNGE_OUTCOME: Final = "entered-unmodelled-plunge-sink"
PLUNGE_UPPER_TARGET_ID: Final = "finite-thickness-upper-plunge-entry"
PLUNGE_LOWER_TARGET_ID: Final = "finite-thickness-lower-plunge-entry"

ReturningRayFate = Literal[
    "return-upper",
    "return-lower",
    "captured",
    "escaped",
    "plunge-sink",
]
_FATES: Final = (
    "return-upper",
    "return-lower",
    "captured",
    "escaped",
    "plunge-sink",
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "deterministic single-local-direction finite-thickness Kerr "
            "return/capture/escape/plunge fate and receiver transport primitive"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "initialSurfaceContact": (
            "exact authenticated launch state is assigned the declared "
            "outward-positive topology side; all positive-affine probes use "
            "the physical signed photosphere and near-zero later roots fail closed"
        ),
        "surfaceCompleteness": (
            "independent fine/coarse whole rays, each with accepted-step N/2N "
            "shared-probe topology at finite declared resolution"
        ),
        "receiverCoefficientSemantics": (
            "g^4 and mu_receiver*g^4 for one returned direction only; no "
            "emission law, solid-angle/area Jacobian, or annulus quadrature weight"
        ),
        "plungeSinkSemantics": (
            "first inward crossing of an auxiliary face continuation at or "
            "inside the ISCO; an absorbing bookkeeping boundary, not solved flow"
        ),
        "isIndependentRayTransportPrimitive": True,
        "isIndependentRayKernel": False,
        "requiresPublicRevalidationBeforeConsumption": True,
        "isCompleteReturningRadiationKernel": False,
        "isCompleteKerrbb": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesSolvedAtmosphere": False,
        "isHydrostaticVerticalStructureSolution": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this one-ray finite-probe primitive as a complete "
            "returning-radiation kernel, complete KERRBB, F_S stress work, "
            "hydrostatic disk, atmosphere, GRMHD, or NR calculation."
        ),
    }
)

_INITIAL_CONTACT_MAXIMUM_RESIDUAL: Final = 2.0e-11
_RECEIVER_EVENT_TOLERANCE_OVER_MASS: Final = 2.0e-8
_RECEIVER_NULL_RESIDUAL_LIMIT: Final = 1.0e-6
_DEFAULT_COARSE_TOLERANCE_MULTIPLIER: Final = 8.0
_MAXIMUM_COARSE_TOLERANCE_MULTIPLIER: Final = 64.0
_TERMINAL_EVENT_DIFFERENCE_OVER_MASS: Final = 2.0e-5
_TERMINAL_COVECTOR_RELATIVE_DIFFERENCE: Final = 2.0e-5
_CROSSING_EVENT_DIFFERENCE_OVER_MASS: Final = 2.0e-5
_CROSSING_COVECTOR_RELATIVE_DIFFERENCE: Final = 2.0e-5
_RECEIVER_RADIUS_DIFFERENCE: Final = 2.0e-5
_FREQUENCY_RATIO_RELATIVE_DIFFERENCE: Final = 2.0e-5
_SIGNED_COSINE_DIFFERENCE: Final = 2.0e-5
_G4_RELATIVE_DIFFERENCE: Final = 8.0e-5
_WEIGHTED_G4_RELATIVE_DIFFERENCE: Final = 1.0e-4


class KerrReturningRadiationRayError(RuntimeError):
    """Raised when one local direction cannot be certified fail-closed."""


@dataclass(frozen=True, slots=True)
class _IssuedKerrReturningRadiationRayPayload:
    """Immutable fields sealed at canonical trace issuance for one consumer."""

    fate: ReturningRayFate
    receiver_face: PhotosphereFace | None
    receiver_radius_over_mass: float | None
    emitter_to_receiver_frequency_ratio: float | None
    primitive_descriptor_sha256: str
    coarse_receiver_face: PhotosphereFace | None
    coarse_receiver_radius_over_mass: float | None


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationRayError(
            f"{path}.{name} is missing from the authenticated schema"
        ) from error


def _require_exact_schema_types(actual: Any, template: Any, path: str) -> None:
    """Check a trusted template recursively without invoking actual equality."""

    if type(actual) is not type(template):
        raise KerrReturningRadiationRayError(
            f"{path} has a non-exact field type "
            f"{type(actual).__name__}; expected {type(template).__name__}"
        )
    if is_dataclass(template) and not isinstance(template, type):
        for field in fields(template):
            _require_exact_schema_types(
                _trusted_attribute(actual, field.name, path),
                _trusted_attribute(template, field.name, path),
                f"{path}.{field.name}",
            )
        return
    if type(template) is tuple:
        if len(actual) != len(template):
            raise KerrReturningRadiationRayError(
                f"{path} tuple length disagrees with the authenticated schema"
            )
        for index, (actual_item, template_item) in enumerate(zip(actual, template)):
            _require_exact_schema_types(
                actual_item,
                template_item,
                f"{path}[{index}]",
            )
        return
    if type(template) not in (float, int, bool, str, type(None)):
        raise KerrReturningRadiationRayError(
            f"{path} uses unsupported authenticated type {type(template).__name__}"
        )


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare against trusted values without any untrusted ``__eq__`` call."""

    _require_exact_schema_types(actual, expected, path)
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, field.name, path),
                _trusted_attribute(expected, field.name, path),
                f"{path}.{field.name}",
            )
        return
    if type(expected) is tuple:
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_trusted_exact_tree(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    differs = False
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) is int:
        differs = int(actual) != int(expected)
    elif type(expected) is bool:
        differs = actual is not expected
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationRayError(
            f"{path} uses an unsupported trusted primitive"
        )
    if differs:
        raise KerrReturningRadiationRayError(
            f"{path} differs from the trusted reconstructed value"
        )


def _rebuild_metric(metric: KerrKerrSchildMetric) -> KerrKerrSchildMetric:
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must be the exact KerrKerrSchildMetric")
    _require_exact_schema_types(
        metric,
        KerrKerrSchildMetric(),
        "surface.metric",
    )
    expected = KerrKerrSchildMetric(
        mass_m=metric.mass_m,
        spin_a_m=metric.spin_a_m,
        singularity_guard_m=metric.singularity_guard_m,
    )
    _require_trusted_exact_tree(metric, expected, "surface.metric")
    return expected


def _rebuild_calibration(
    calibration: StationaryKerrFiniteThicknessCalibration,
) -> StationaryKerrFiniteThicknessCalibration:
    if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "surface calibration must be the exact built-in finite-thickness calibration"
        )
    _require_exact_schema_types(
        calibration,
        StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.0,
        ),
        "surface.calibration",
    )
    expected = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=calibration.dimensionless_spin,
        eddington_scaled_mass_accretion_rate=(
            calibration.eddington_scaled_mass_accretion_rate
        ),
        orientation=calibration.orientation,
        outer_radius_over_mass=calibration.outer_radius_over_mass,
        thinness_gate_maximum_h_over_rho=(
            calibration.thinness_gate_maximum_h_over_rho
        ),
    )
    _require_trusted_exact_tree(
        calibration,
        expected,
        "surface.calibration",
    )
    return expected


def _validated_surface(
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessMultiSurface:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be the exact KerrFiniteThicknessMultiSurface")
    metric = _rebuild_metric(surface.metric)
    calibration = _rebuild_calibration(surface.calibration)
    expected = KerrFiniteThicknessMultiSurface(metric, calibration)
    _require_trusted_exact_tree(surface, expected, "surface")
    return expected


def _validated_launch(
    launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessEmissionLaunch:
    if type(launch) is not KerrFiniteThicknessEmissionLaunch:
        raise TypeError("launch must be the exact KerrFiniteThicknessEmissionLaunch")
    if type(launch.frame) is not KerrFiniteThicknessSurfaceFrame:
        raise TypeError("launch frame must be the exact finite-thickness surface frame")
    emitter = launch.frame.emitter
    if type(emitter) is not KerrFiniteThicknessFaceEmitter:
        raise TypeError("launch emitter must be the exact finite-thickness emitter")
    _require_trusted_exact_tree(
        emitter.metric,
        surface.metric,
        "launch.frame.emitter.metric",
    )
    _require_trusted_exact_tree(
        emitter.calibration,
        surface.calibration,
        "launch.frame.emitter.calibration",
    )
    expected_emitter = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=(
            emitter.pseudo_cylindrical_radius_over_mass
        ),
        face=emitter.face,
        phi_ks_rad=emitter.phi_ks_rad,
        coordinate_time_m=emitter.coordinate_time_m,
    )
    expected_frame = KerrFiniteThicknessSurfaceFrame(expected_emitter)
    expected = KerrFiniteThicknessEmissionLaunch(
        expected_frame,
        launch.emission_angle_cosine,
        launch.tangent_azimuth_rad,
        launch.local_frequency,
    )
    _require_trusted_exact_tree(launch, expected, "launch")
    return expected


def _validated_termination(
    termination: KerrOblateTermination,
    surface: KerrFiniteThicknessMultiSurface,
    initial_state: HamiltonianState,
) -> KerrOblateTermination:
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be the exact KerrOblateTermination")
    _require_exact_schema_types(
        termination,
        KerrOblateTermination(
            spin_a_m=0.0,
            capture_radius_m=1.0,
            escape_radius_m=2.0,
        ),
        "termination",
    )
    expected = KerrOblateTermination(
        spin_a_m=termination.spin_a_m,
        capture_radius_m=termination.capture_radius_m,
        escape_radius_m=termination.escape_radius_m,
        capture_target_id=termination.capture_target_id,
        escape_target_id=termination.escape_target_id,
    )
    _require_trusted_exact_tree(termination, expected, "termination")
    metric = surface.metric
    scale = max(1.0, metric.mass_m)
    if not math.isclose(
        expected.spin_a_m,
        metric.spin_a_m,
        rel_tol=0.0,
        abs_tol=16.0 * math.ulp(scale),
    ):
        raise ValueError("termination and Kerr metric signed spins disagree")
    inner_radius_m = surface.calibration.isco_radius_over_mass * metric.mass_m
    outer_point = surface.calibration.photosphere_point(
        surface.calibration.outer_radius_over_mass,
        UPPER,
    )
    outer_radius_m = outer_point.radius_over_mass * metric.mass_m
    if expected.capture_radius_m < metric.outer_horizon_radius_m:
        raise ValueError("capture worldtube may not lie inside the Kerr horizon")
    if expected.capture_radius_m >= inner_radius_m:
        raise ValueError("capture worldtube must lie strictly inside the ISCO")
    if expected.escape_radius_m <= outer_radius_m:
        raise ValueError(
            "escape worldtube must lie outside the maximum photosphere radius"
        )
    if expected.classify_initial(initial_state) is not None:
        raise ValueError("emission launch starts on or beyond a terminal worldtube")
    return expected


def _validated_options(
    metric: KerrKerrSchildMetric,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> tuple[RayTraceOptions, SurfaceEventOptions]:
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("ray_options must be the exact RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError("surface_options must be the exact SurfaceEventOptions")
    _require_exact_schema_types(
        ray_options,
        RayTraceOptions(),
        "ray_options",
    )
    _require_exact_schema_types(
        surface_options,
        SurfaceEventOptions(),
        "surface_options",
    )
    # Re-running both public constructors closes low-level frozen-dataclass
    # mutation before any work budget or tolerance enters the tracer.
    try:
        expected_ray_options = RayTraceOptions(**asdict(ray_options))
        expected_surface_options = SurfaceEventOptions(**asdict(surface_options))
    except (TypeError, ValueError) as error:
        raise ValueError(f"returning-ray options are stale: {error}") from error
    _require_trusted_exact_tree(
        ray_options,
        expected_ray_options,
        "ray_options",
    )
    _require_trusted_exact_tree(
        surface_options,
        expected_surface_options,
        "surface_options",
    )
    ray_options = expected_ray_options
    surface_options = expected_surface_options
    scale = max(1.0, metric.mass_m)
    if (
        ray_options.absolute_tolerance > 1.0e-7 * scale
        or ray_options.relative_tolerance > 1.0e-7
        or ray_options.maximum_step > 2.0 * scale
        or ray_options.null_residual_limit > 1.0e-6
        or ray_options.metric_interpolation_error_limit > 1.0e-6
        or ray_options.event_value_tolerance > 1.0e-7 * scale
        or ray_options.event_affine_tolerance > 1.0e-7 * scale
        or ray_options.maximum_affine_length > 1.0e6 * scale
        or ray_options.maximum_accepted_steps > 1_000_000
        or ray_options.maximum_rejected_steps > 1_000_000
        or ray_options.event_maximum_iterations > 256
    ):
        raise ValueError("ray options exceed the returning-ray accuracy policy")
    if (
        surface_options.absolute_tolerance > 1.0e-7 * scale
        or surface_options.relative_tolerance > 1.0e-7
        or surface_options.null_residual_limit > 1.0e-6
        or surface_options.metric_interpolation_error_limit > 1.0e-6
        or surface_options.surface_value_tolerance > 1.0e-7
        or surface_options.affine_tolerance > 1.0e-7 * scale
        or surface_options.subdivisions_per_segment < 2
        or surface_options.subdivisions_per_segment > 128
        or surface_options.maximum_iterations > 256
        or surface_options.maximum_reintegrations > 2_000_000
    ):
        raise ValueError("surface options exceed the returning-ray accuracy policy")
    return ray_options, surface_options


def _validated_fine_options(
    metric: KerrKerrSchildMetric,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> None:
    """Prevent a single permissive trace from being labelled the fine ray."""

    scale = max(1.0, metric.mass_m)
    if (
        ray_options.absolute_tolerance > 1.0e-8 * scale
        or ray_options.relative_tolerance > 1.0e-8
        or ray_options.maximum_step > scale
        or ray_options.event_value_tolerance > 1.0e-8 * scale
        or ray_options.event_affine_tolerance > 1.0e-8 * scale
        or surface_options.absolute_tolerance > 1.0e-8 * scale
        or surface_options.relative_tolerance > 1.0e-8
        or surface_options.surface_value_tolerance > 1.0e-8
        or surface_options.affine_tolerance > 1.0e-8 * scale
        or surface_options.subdivisions_per_segment < 4
    ):
        raise ValueError(
            "fine options exceed the independent whole-ray accuracy policy"
        )


def _derive_coarse_options(
    metric: KerrKerrSchildMetric,
    fine_ray: RayTraceOptions,
    fine_surface: SurfaceEventOptions,
) -> tuple[RayTraceOptions, SurfaceEventOptions]:
    factor = _DEFAULT_COARSE_TOLERANCE_MULTIPLIER
    scale = max(1.0, metric.mass_m)
    coarse_maximum_step = min(
        2.0 * scale,
        max(2.0 * fine_ray.maximum_step, 2.0 * fine_ray.initial_step),
    )
    coarse_initial_step = min(
        coarse_maximum_step,
        max(2.0 * fine_ray.initial_step, fine_ray.minimum_step),
    )
    coarse_minimum_step = min(
        coarse_initial_step,
        factor * fine_ray.minimum_step,
    )
    coarse_ray = replace(
        fine_ray,
        absolute_tolerance=min(1.0e-7 * scale, factor * fine_ray.absolute_tolerance),
        relative_tolerance=min(1.0e-7, factor * fine_ray.relative_tolerance),
        initial_step=coarse_initial_step,
        minimum_step=coarse_minimum_step,
        maximum_step=coarse_maximum_step,
        null_residual_limit=min(1.0e-6, factor * fine_ray.null_residual_limit),
        metric_interpolation_error_limit=min(
            1.0e-6,
            factor * fine_ray.metric_interpolation_error_limit,
        ),
        event_value_tolerance=min(
            1.0e-7 * scale,
            factor * fine_ray.event_value_tolerance,
        ),
        event_affine_tolerance=min(
            1.0e-7 * scale,
            factor * fine_ray.event_affine_tolerance,
        ),
    )
    coarse_subdivisions = max(2, fine_surface.subdivisions_per_segment // 2)
    if coarse_subdivisions % 2:
        coarse_subdivisions -= 1
    coarse_surface = replace(
        fine_surface,
        absolute_tolerance=min(
            1.0e-7 * scale,
            factor * fine_surface.absolute_tolerance,
        ),
        relative_tolerance=min(
            1.0e-7,
            factor * fine_surface.relative_tolerance,
        ),
        null_residual_limit=min(
            1.0e-6,
            factor * fine_surface.null_residual_limit,
        ),
        metric_interpolation_error_limit=min(
            1.0e-6,
            factor * fine_surface.metric_interpolation_error_limit,
        ),
        surface_value_tolerance=min(
            1.0e-7,
            factor * fine_surface.surface_value_tolerance,
        ),
        affine_tolerance=min(
            1.0e-7 * scale,
            factor * fine_surface.affine_tolerance,
        ),
        subdivisions_per_segment=coarse_subdivisions,
    )
    return coarse_ray, coarse_surface


def _coarse_ratio(coarse: float, fine: float) -> float:
    ratio = coarse / fine
    if not math.isfinite(ratio):
        raise ValueError("fine/coarse tolerance ratio is not finite")
    return ratio


def _validate_coarse_relationship(
    fine_ray: RayTraceOptions,
    fine_surface: SurfaceEventOptions,
    coarse_ray: RayTraceOptions,
    coarse_surface: SurfaceEventOptions,
) -> None:
    if fine_ray == coarse_ray and fine_surface == coarse_surface:
        raise ValueError("fine and coarse whole-ray discretizations must differ")
    if (
        coarse_ray.maximum_affine_length != fine_ray.maximum_affine_length
        or coarse_ray.maximum_accepted_steps != fine_ray.maximum_accepted_steps
        or coarse_ray.maximum_rejected_steps != fine_ray.maximum_rejected_steps
        or coarse_ray.record_path != fine_ray.record_path
    ):
        raise ValueError(
            "fine/coarse rays must share physical extent, work budgets, and path policy"
        )
    ray_ratios = tuple(
        _coarse_ratio(coarse, fine)
        for coarse, fine in (
            (coarse_ray.absolute_tolerance, fine_ray.absolute_tolerance),
            (coarse_ray.relative_tolerance, fine_ray.relative_tolerance),
            (coarse_ray.null_residual_limit, fine_ray.null_residual_limit),
            (
                coarse_ray.metric_interpolation_error_limit,
                fine_ray.metric_interpolation_error_limit,
            ),
            (coarse_ray.event_value_tolerance, fine_ray.event_value_tolerance),
            (coarse_ray.event_affine_tolerance, fine_ray.event_affine_tolerance),
        )
    )
    surface_ratios = tuple(
        _coarse_ratio(coarse, fine)
        for coarse, fine in (
            (coarse_surface.absolute_tolerance, fine_surface.absolute_tolerance),
            (coarse_surface.relative_tolerance, fine_surface.relative_tolerance),
            (coarse_surface.null_residual_limit, fine_surface.null_residual_limit),
            (
                coarse_surface.metric_interpolation_error_limit,
                fine_surface.metric_interpolation_error_limit,
            ),
            (
                coarse_surface.surface_value_tolerance,
                fine_surface.surface_value_tolerance,
            ),
            (coarse_surface.affine_tolerance, fine_surface.affine_tolerance),
        )
    )
    if any(
        ratio < 1.0 or ratio > _MAXIMUM_COARSE_TOLERANCE_MULTIPLIER
        for ratio in (*ray_ratios, *surface_ratios)
    ):
        raise ValueError(
            "coarse tolerances must lie between 1x and 64x their fine values"
        )
    if (
        coarse_ray.initial_step < fine_ray.initial_step
        or coarse_ray.minimum_step < fine_ray.minimum_step
        or coarse_ray.maximum_step < fine_ray.maximum_step
        or coarse_surface.subdivisions_per_segment
        > fine_surface.subdivisions_per_segment
    ):
        raise ValueError(
            "coarse discretization must not be finer than the declared fine ray"
        )
    materially_different = (
        any(ratio > 1.0 for ratio in (*ray_ratios, *surface_ratios))
        or coarse_ray.initial_step > fine_ray.initial_step
        or coarse_ray.minimum_step > fine_ray.minimum_step
        or coarse_ray.maximum_step > fine_ray.maximum_step
        or coarse_surface.subdivisions_per_segment
        < fine_surface.subdivisions_per_segment
    )
    if not materially_different:
        raise ValueError(
            "coarse whole ray must use a materially different integration or "
            "surface discretization"
        )


def _face_for_surface_id(surface_id: str) -> PhotosphereFace:
    if surface_id == UPPER_SURFACE_ID:
        return UPPER
    if surface_id == LOWER_SURFACE_ID:
        return LOWER
    raise KerrReturningRadiationRayError("return trace contains an unknown surface id")


class _ReturningSurfaceClassification:
    """Product classifications layered on the physical two-face scalars."""

    surface_ids = FINITE_THICKNESS_SURFACE_IDS

    def __init__(
        self,
        surface: KerrFiniteThicknessMultiSurface,
    ) -> None:
        self.surface = surface

    def value(self, surface_id: str, state: HamiltonianState) -> float:
        return self.surface.value(surface_id, state)

    def classify(
        self,
        surface_id: str,
        crossing: RecordedSurfaceCrossing,
    ) -> InteriorSurfaceDecision:
        face = _face_for_surface_id(surface_id)
        oblate = kerr_ks_event_to_oblate(
            self.surface.metric,
            crossing.state.event,
        )
        rho = (
            oblate.radius_m
            * math.sin(oblate.theta_rad)
            / self.surface.metric.mass_m
        )
        inner = self.surface.calibration.isco_radius_over_mass
        if rho <= inner:
            if crossing.orientation == -1:
                return InteriorSurfaceDecision(
                    f"inward-{face}-continuation-plunge-entry",
                    PLUNGE_OUTCOME,
                    (
                        PLUNGE_UPPER_TARGET_ID
                        if face == UPPER
                        else PLUNGE_LOWER_TARGET_ID
                    ),
                )
            return InteriorSurfaceDecision(
                f"outward-{face}-continuation-plunge-exit-transparent"
            )

        base = self.surface.classify(surface_id, crossing)
        if not base.terminates:
            return base
        return InteriorSurfaceDecision(
            f"subsequent-{face}-photosphere-contact",
            RETURNED_OUTCOME,
            UPPER_TARGET_ID if face == UPPER else LOWER_TARGET_ID,
        )


def _validated_terminal_entry(
    ray: RayTraceResult,
    surface_options: SurfaceEventOptions,
):
    trace = ray.multi_surface_trace
    if trace is None or not trace.topology_converged:
        raise KerrReturningRadiationRayError(
            "returning ray lacks converged finite-resolution multi-surface topology"
        )
    if trace.surface_ids != tuple(sorted(FINITE_THICKNESS_SURFACE_IDS)):
        raise KerrReturningRadiationRayError(
            "returning ray surface ids do not match the owned two-face geometry"
        )
    if (
        trace.base_subdivisions_per_step
        != surface_options.subdivisions_per_segment
        or trace.verification_subdivisions_per_step
        != 2 * surface_options.subdivisions_per_segment
    ):
        raise KerrReturningRadiationRayError(
            "returning ray topology resolution disagrees with its options"
        )
    terminal = trace.crossings[-1] if trace.crossings else None
    if ray.outcome in (RETURNED_OUTCOME, PLUNGE_OUTCOME):
        if terminal is None or not terminal.decision.terminates:
            raise KerrReturningRadiationRayError(
                "surface-terminal returning ray lacks its terminal crossing"
            )
        if (
            terminal.decision.outcome != ray.outcome
            or terminal.decision.target_id != ray.terminal_target_id
            or terminal.crossing.state != ray.terminal_state
        ):
            raise KerrReturningRadiationRayError(
                "terminal surface crossing does not own the ray outcome"
            )
        minimum_subsequent_affine = 8.0 * surface_options.affine_tolerance
        if terminal.crossing.ray_affine_length <= minimum_subsequent_affine:
            raise KerrReturningRadiationRayError(
                "later surface contact is unresolved from the exact initial contact"
            )
        return terminal
    if terminal is not None and terminal.decision.terminates:
        raise KerrReturningRadiationRayError(
            "worldtube-terminal ray hides a terminal photosphere crossing"
        )
    return None


def _worldtube_fate(
    ray: RayTraceResult,
    termination: KerrOblateTermination,
    metric: KerrKerrSchildMetric,
    ray_options: RayTraceOptions,
) -> tuple[ReturningRayFate, float]:
    if ray.outcome == "captured":
        expected_target = termination.capture_target_id
        expected_radius = termination.capture_radius_m
        fate: ReturningRayFate = "captured"
    elif ray.outcome == "escaped":
        expected_target = termination.escape_target_id
        expected_radius = termination.escape_radius_m
        fate = "escaped"
    else:
        raise KerrReturningRadiationRayError("ray has no supported terminal fate")
    if ray.terminal_target_id != expected_target:
        raise KerrReturningRadiationRayError(
            "terminal worldtube target is not owned by the termination"
        )
    actual_radius = termination.radius(ray.terminal_state)
    boundary_tolerance = max(
        2.0 * ray_options.event_value_tolerance,
        32.0 * math.ulp(max(1.0, expected_radius, metric.mass_m)),
    )
    if abs(actual_radius - expected_radius) > boundary_tolerance:
        raise KerrReturningRadiationRayError(
            "terminal state is not on its declared Kerr worldtube"
        )
    return fate, actual_radius


def _event_difference(first: HamiltonianState, second: HamiltonianState) -> float:
    return math.sqrt(
        math.fsum(
            (first.event[index] - second.event[index]) ** 2
            for index in range(4)
        )
    )


def _covector_relative_difference(
    first: HamiltonianState,
    second: HamiltonianState,
) -> float:
    difference = math.sqrt(
        math.fsum(
            (first.covector[index] - second.covector[index]) ** 2
            for index in range(4)
        )
    )
    scale = max(
        math.sqrt(math.fsum(value * value for value in first.covector)),
        math.sqrt(math.fsum(value * value for value in second.covector)),
        1.0e-300,
    )
    return difference / scale


def _relative_scalar_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-300)


def _terminal_rho(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
) -> float:
    oblate = kerr_ks_event_to_oblate(metric, state.event)
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / metric.mass_m
    if not math.isfinite(rho) or rho < 0.0:
        raise KerrReturningRadiationRayError("terminal oblate rho is invalid")
    return rho


@dataclass(frozen=True, slots=True)
class _EvaluatedReturningRay:
    ray: RayTraceResult
    fate: ReturningRayFate
    terminal_entry: Any | None
    terminal_rho_over_mass: float
    worldtube_radius_m: float | None
    receiver_face: PhotosphereFace | None
    receiver: KerrFiniteThicknessFaceEmitter | None
    receiver_radius_over_mass: float | None
    receiver_surface_id: str | None
    signed_receiver_outgoing_cosine: float | None
    receiver_incidence_cosine: float | None
    frequency_ratio: float | None
    g4: float | None
    incidence_weighted_g4: float


def _trace_one_resolution(
    normalized_launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> tuple[RayTraceResult, float]:
    classified_surface = _ReturningSurfaceClassification(surface)
    emitting_surface_id = (
        UPPER_SURFACE_ID
        if normalized_launch.frame.emitter.face == UPPER
        else LOWER_SURFACE_ID
    )
    initial_contact_residual = surface.value(
        emitting_surface_id,
        normalized_launch.future_state,
    )
    if abs(initial_contact_residual) > _INITIAL_CONTACT_MAXIMUM_RESIDUAL:
        raise KerrReturningRadiationRayError(
            "authenticated launch is not on its declared emitting face"
        )
    try:
        ray = trace_null_geodesic(
            surface.metric,
            normalized_launch.future_state,
            termination=termination,
            multi_interior_surface=classified_surface,
            initial_multi_surface_contact=InitialMultiSurfaceContact(
                emitting_surface_id,
                1,
            ),
            surface_options=surface_options,
            options=ray_options,
        )
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
        raise KerrReturningRadiationRayError(
            f"future returning-ray trace failed: {error}"
        ) from error
    if ray.failure_reason is not None or ray.outcome in (
        "integrator-failure",
        "unresolved",
        "completed",
    ):
        raise KerrReturningRadiationRayError(
            "future returning ray did not reach a certified fate: "
            f"{ray.outcome}: {ray.failure_reason or 'no terminal worldtube'}"
        )
    trace = ray.multi_surface_trace
    if trace is None or trace.initial_contact is None:
        raise KerrReturningRadiationRayError(
            "future returning ray lost its authenticated initial contact"
        )
    contact = trace.initial_contact
    if (
        contact.surface_id != emitting_surface_id
        or contact.side != 1
        or contact.actual_surface_value.hex()
        != float(initial_contact_residual).hex()
    ):
        raise KerrReturningRadiationRayError(
            "future returning ray initial-contact provenance is stale"
        )
    return ray, initial_contact_residual


def _evaluate_returning_ray(
    ray: RayTraceResult,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> _EvaluatedReturningRay:
    terminal = _validated_terminal_entry(ray, surface_options)
    terminal_rho = _terminal_rho(surface.metric, ray.terminal_state)
    if ray.outcome in ("captured", "escaped"):
        fate, worldtube_radius = _worldtube_fate(
            ray,
            termination,
            surface.metric,
            ray_options,
        )
        return _EvaluatedReturningRay(
            ray=ray,
            fate=fate,
            terminal_entry=None,
            terminal_rho_over_mass=terminal_rho,
            worldtube_radius_m=worldtube_radius,
            receiver_face=None,
            receiver=None,
            receiver_radius_over_mass=None,
            receiver_surface_id=None,
            signed_receiver_outgoing_cosine=None,
            receiver_incidence_cosine=None,
            frequency_ratio=None,
            g4=None,
            incidence_weighted_g4=0.0,
        )
    if ray.outcome == PLUNGE_OUTCOME:
        return _EvaluatedReturningRay(
            ray=ray,
            fate="plunge-sink",
            terminal_entry=terminal,
            terminal_rho_over_mass=terminal_rho,
            worldtube_radius_m=None,
            receiver_face=None,
            receiver=None,
            receiver_radius_over_mass=None,
            receiver_surface_id=None,
            signed_receiver_outgoing_cosine=None,
            receiver_incidence_cosine=None,
            frequency_ratio=None,
            g4=None,
            incidence_weighted_g4=0.0,
        )
    if ray.outcome != RETURNED_OUTCOME or terminal is None:
        raise KerrReturningRadiationRayError(
            "surface-terminal ray has an unsupported physical classification"
        )
    if terminal.crossing.orientation != -1:
        raise KerrReturningRadiationRayError(
            "first photosphere contact is not front-face inward incidence"
        )
    face = _face_for_surface_id(terminal.surface_id)
    oblate = kerr_ks_event_to_oblate(surface.metric, terminal.crossing.state.event)
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / surface.metric.mass_m
    receiver = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=rho,
        face=face,
        phi_ks_rad=oblate.phi_ks_rad,
        coordinate_time_m=terminal.crossing.state.event[0],
    )
    event_tolerance_m = max(
        _RECEIVER_EVENT_TOLERANCE_OVER_MASS * surface.metric.mass_m,
        2.0 * surface_options.surface_value_tolerance * surface.metric.mass_m,
    )
    event_difference = max(
        abs(actual - expected)
        for actual, expected in zip(terminal.crossing.state.event, receiver.event)
    )
    if event_difference > event_tolerance_m:
        raise KerrReturningRadiationRayError(
            "localized return is too far from its reconstructed receiver face"
        )
    receiver_past_state = HamiltonianState(
        terminal.crossing.state.event,
        tuple(-value for value in terminal.crossing.state.covector),
    )
    projection = receiver.project_past_directed_photon(
        receiver_past_state,
        null_residual_limit=min(
            _RECEIVER_NULL_RESIDUAL_LIMIT,
            surface_options.null_residual_limit,
        ),
        event_tolerance_m=event_tolerance_m,
        backside_policy="classify",
    )
    if (
        projection.face_classification != "backside"
        or projection.outgoing_cosine >= 0.0
    ):
        raise KerrReturningRadiationRayError(
            "returned future photon does not illuminate the receiver front face"
        )
    receiver_incidence = -projection.outgoing_cosine
    frequency_ratio = projection.local_frequency
    try:
        g4 = frequency_ratio**4
        weighted = receiver_incidence * g4
    except OverflowError as error:
        raise KerrReturningRadiationRayError(
            "bolometric receiver factor overflowed"
        ) from error
    if not all(math.isfinite(value) and value > 0.0 for value in (g4, weighted)):
        raise KerrReturningRadiationRayError("bolometric receiver factor is invalid")
    fate: ReturningRayFate = "return-upper" if face == UPPER else "return-lower"
    return _EvaluatedReturningRay(
        ray=ray,
        fate=fate,
        terminal_entry=terminal,
        terminal_rho_over_mass=terminal_rho,
        worldtube_radius_m=None,
        receiver_face=face,
        receiver=receiver,
        receiver_radius_over_mass=rho,
        receiver_surface_id=terminal.surface_id,
        signed_receiver_outgoing_cosine=projection.outgoing_cosine,
        receiver_incidence_cosine=receiver_incidence,
        frequency_ratio=frequency_ratio,
        g4=g4,
        incidence_weighted_g4=weighted,
    )


def _topology_signature(ray: RayTraceResult) -> tuple[Any, ...]:
    trace = ray.multi_surface_trace
    if trace is None:
        raise KerrReturningRadiationRayError("ray has no multi-surface topology")
    return tuple(
        (
            entry.surface_id,
            entry.crossing.orientation,
            entry.decision.classification,
            entry.decision.outcome,
            entry.decision.target_id,
        )
        for entry in trace.crossings
    )


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationRayConvergence:
    """Independent fine/coarse whole-ray agreement evidence."""

    outcome_agrees: bool
    target_agrees: bool
    fate_agrees: bool
    complete_topology_agrees: bool
    terminal_event_difference_m: float
    terminal_covector_relative_difference: float
    maximum_crossing_event_difference_m: float
    maximum_crossing_covector_relative_difference: float
    receiver_radius_difference_over_mass: float | None
    frequency_ratio_relative_difference: float | None
    signed_receiver_cosine_difference: float | None
    g4_relative_difference: float | None
    incidence_weighted_g4_relative_difference: float | None
    worldtube_radius_difference_m: float | None
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "outcome_agrees",
            "target_agrees",
            "fate_agrees",
            "complete_topology_agrees",
            "converged",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        for name in (
            "terminal_event_difference_m",
            "terminal_covector_relative_difference",
            "maximum_crossing_event_difference_m",
            "maximum_crossing_covector_relative_difference",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "receiver_radius_difference_over_mass",
            "frequency_ratio_relative_difference",
            "signed_receiver_cosine_difference",
            "g4_relative_difference",
            "incidence_weighted_g4_relative_difference",
            "worldtube_radius_difference_m",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be None or finite and non-negative")
        if self.converged and not all(
            (
                self.outcome_agrees,
                self.target_agrees,
                self.fate_agrees,
                self.complete_topology_agrees,
            )
        ):
            raise ValueError("converged whole rays must agree categorically")


def _compare_whole_rays(
    fine: _EvaluatedReturningRay,
    coarse: _EvaluatedReturningRay,
    metric: KerrKerrSchildMetric,
) -> KerrReturningRadiationRayConvergence:
    outcome_agrees = fine.ray.outcome == coarse.ray.outcome
    target_agrees = fine.ray.terminal_target_id == coarse.ray.terminal_target_id
    fate_agrees = fine.fate == coarse.fate
    fine_topology = _topology_signature(fine.ray)
    coarse_topology = _topology_signature(coarse.ray)
    topology_agrees = fine_topology == coarse_topology
    if not all((outcome_agrees, target_agrees, fate_agrees, topology_agrees)):
        raise KerrReturningRadiationRayError(
            "independent fine/coarse whole-ray fate, target, or complete topology "
            f"disagree (fine={fine.fate}/{fine.ray.outcome}, "
            f"coarse={coarse.fate}/{coarse.ray.outcome})"
        )

    terminal_event_difference = _event_difference(
        fine.ray.terminal_state,
        coarse.ray.terminal_state,
    )
    terminal_covector_difference = _covector_relative_difference(
        fine.ray.terminal_state,
        coarse.ray.terminal_state,
    )
    if (
        terminal_event_difference > _TERMINAL_EVENT_DIFFERENCE_OVER_MASS * metric.mass_m
        or terminal_covector_difference > _TERMINAL_COVECTOR_RELATIVE_DIFFERENCE
    ):
        raise KerrReturningRadiationRayError(
            "independent fine/coarse terminal phase-space states disagree"
        )

    fine_crossings = fine.ray.multi_surface_trace.crossings
    coarse_crossings = coarse.ray.multi_surface_trace.crossings
    maximum_crossing_event = 0.0
    maximum_crossing_covector = 0.0
    for fine_entry, coarse_entry in zip(fine_crossings, coarse_crossings):
        maximum_crossing_event = max(
            maximum_crossing_event,
            _event_difference(fine_entry.crossing.state, coarse_entry.crossing.state),
        )
        maximum_crossing_covector = max(
            maximum_crossing_covector,
            _covector_relative_difference(
                fine_entry.crossing.state,
                coarse_entry.crossing.state,
            ),
        )
    if (
        maximum_crossing_event > _CROSSING_EVENT_DIFFERENCE_OVER_MASS * metric.mass_m
        or maximum_crossing_covector > _CROSSING_COVECTOR_RELATIVE_DIFFERENCE
    ):
        raise KerrReturningRadiationRayError(
            "independent fine/coarse complete crossing phase spaces disagree"
        )

    receiver_radius_difference = None
    frequency_ratio_difference = None
    signed_cosine_difference = None
    g4_difference = None
    weighted_difference = None
    worldtube_difference = None
    if fine.fate in ("return-upper", "return-lower"):
        if (
            fine.receiver_face != coarse.receiver_face
            or fine.receiver_surface_id != coarse.receiver_surface_id
            or fine.receiver_radius_over_mass is None
            or coarse.receiver_radius_over_mass is None
            or fine.frequency_ratio is None
            or coarse.frequency_ratio is None
            or fine.signed_receiver_outgoing_cosine is None
            or coarse.signed_receiver_outgoing_cosine is None
            or fine.g4 is None
            or coarse.g4 is None
        ):
            raise KerrReturningRadiationRayError(
                "independent returned rays disagree on receiver ownership"
            )
        receiver_radius_difference = abs(
            fine.receiver_radius_over_mass - coarse.receiver_radius_over_mass
        )
        frequency_ratio_difference = _relative_scalar_difference(
            fine.frequency_ratio,
            coarse.frequency_ratio,
        )
        signed_cosine_difference = abs(
            fine.signed_receiver_outgoing_cosine
            - coarse.signed_receiver_outgoing_cosine
        )
        g4_difference = _relative_scalar_difference(fine.g4, coarse.g4)
        weighted_difference = _relative_scalar_difference(
            fine.incidence_weighted_g4,
            coarse.incidence_weighted_g4,
        )
        if (
            receiver_radius_difference > _RECEIVER_RADIUS_DIFFERENCE
            or frequency_ratio_difference > _FREQUENCY_RATIO_RELATIVE_DIFFERENCE
            or signed_cosine_difference > _SIGNED_COSINE_DIFFERENCE
            or g4_difference > _G4_RELATIVE_DIFFERENCE
            or weighted_difference > _WEIGHTED_G4_RELATIVE_DIFFERENCE
        ):
            raise KerrReturningRadiationRayError(
                "independent fine/coarse receiver rho, g, signed mu, g4, or "
                "incidence-weighted coefficient disagree"
            )
    elif fine.fate in ("captured", "escaped"):
        if fine.worldtube_radius_m is None or coarse.worldtube_radius_m is None:
            raise KerrReturningRadiationRayError(
                "worldtube fate lacks fine/coarse terminal radius"
            )
        worldtube_difference = abs(
            fine.worldtube_radius_m - coarse.worldtube_radius_m
        )
        if worldtube_difference > _TERMINAL_EVENT_DIFFERENCE_OVER_MASS * metric.mass_m:
            raise KerrReturningRadiationRayError(
                "independent fine/coarse worldtube terminals disagree"
            )
    elif abs(
        fine.terminal_rho_over_mass - coarse.terminal_rho_over_mass
    ) > _RECEIVER_RADIUS_DIFFERENCE:
        raise KerrReturningRadiationRayError(
            "independent fine/coarse plunge terminals disagree"
        )

    return KerrReturningRadiationRayConvergence(
        outcome_agrees=outcome_agrees,
        target_agrees=target_agrees,
        fate_agrees=fate_agrees,
        complete_topology_agrees=topology_agrees,
        terminal_event_difference_m=terminal_event_difference,
        terminal_covector_relative_difference=terminal_covector_difference,
        maximum_crossing_event_difference_m=maximum_crossing_event,
        maximum_crossing_covector_relative_difference=maximum_crossing_covector,
        receiver_radius_difference_over_mass=receiver_radius_difference,
        frequency_ratio_relative_difference=frequency_ratio_difference,
        signed_receiver_cosine_difference=signed_cosine_difference,
        g4_relative_difference=g4_difference,
        incidence_weighted_g4_relative_difference=weighted_difference,
        worldtube_radius_difference_m=worldtube_difference,
        converged=True,
    )


@dataclass(frozen=True, slots=True, init=False)
class KerrReturningRadiationRayPrimitive:
    """Fine result accepted only after an independent whole-ray comparison."""

    launch: KerrFiniteThicknessEmissionLaunch
    normalized_launch: KerrFiniteThicknessEmissionLaunch
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions
    coarse_surface_options: SurfaceEventOptions
    ray: RayTraceResult
    coarse_ray: RayTraceResult
    convergence: KerrReturningRadiationRayConvergence
    fate: ReturningRayFate
    receiver_face: PhotosphereFace | None
    receiver: KerrFiniteThicknessFaceEmitter | None
    receiver_radius_over_mass: float | None
    receiver_surface_id: str | None
    emitter_direction_cosine: float
    receiver_incidence_cosine: float | None
    emitter_to_receiver_frequency_ratio: float | None
    bolometric_g4_factor: float | None
    receiver_incidence_weighted_g4: float
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrReturningRadiationRayPrimitive is built only by the certified tracer"
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)

    def photon_fate_indicators(self) -> Mapping[str, int]:
        if type(self.fate) is not str:
            raise KerrReturningRadiationRayError(
                "fate has a non-exact type; revalidate before consumption"
            )
        encoded_fate = self.fate.encode("utf-8")
        return MappingProxyType(
            {
                candidate: int(candidate.encode("utf-8") == encoded_fate)
                for candidate in _FATES
            }
        )

    def revalidate(self) -> None:
        """Re-run both whole rays before a downstream consumer uses this result."""

        verify_kerr_returning_radiation_direction(self)


def _evaluation_descriptor(evaluation: _EvaluatedReturningRay) -> Mapping[str, Any]:
    return {
        "fate": evaluation.fate,
        "outcome": evaluation.ray.outcome,
        "terminalTargetId": evaluation.ray.terminal_target_id,
        "terminalEventKs": evaluation.ray.terminal_state.event,
        "terminalCovectorKs": evaluation.ray.terminal_state.covector,
        "terminalRhoOverMass": evaluation.terminal_rho_over_mass,
        "worldtubeRadiusM": evaluation.worldtube_radius_m,
        "receiver": None
        if evaluation.receiver is None
        else {
            "bolometricG4": evaluation.g4,
            "descriptorSha256": evaluation.receiver.model_descriptor_sha256,
            "face": evaluation.receiver_face,
            "frequencyRatioEmitterToReceiver": evaluation.frequency_ratio,
            "signedOutgoingCosine": evaluation.signed_receiver_outgoing_cosine,
            "incidenceCosine": evaluation.receiver_incidence_cosine,
            "incidenceWeightedG4": evaluation.incidence_weighted_g4,
            "pseudoCylindricalRadiusOverMass": (
                evaluation.receiver_radius_over_mass
            ),
            "surfaceId": evaluation.receiver_surface_id,
        },
    }


def _build_result(
    *,
    launch: KerrFiniteThicknessEmissionLaunch,
    normalized_launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    fine_ray_options: RayTraceOptions,
    fine_surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions,
    coarse_surface_options: SurfaceEventOptions,
    fine: _EvaluatedReturningRay,
    coarse: _EvaluatedReturningRay,
    convergence: KerrReturningRadiationRayConvergence,
    fine_initial_contact_residual: float,
    coarse_initial_contact_residual: float,
) -> KerrReturningRadiationRayPrimitive:
    descriptor: dict[str, Any] = {
        "capabilities": dict(SCIENTIFIC_STATUS),
        "emission": {
            "directionCosine": launch.emission_angle_cosine,
            "inputAffineLocalFrequency": launch.local_frequency,
            "launchDescriptorSha256": launch.model_descriptor_sha256,
            "normalizedLaunchDescriptorSha256": (
                normalized_launch.model_descriptor_sha256
            ),
            "traceAffineConvention": (
                "authenticated launch rebuilt at unit emitter-local frequency"
            ),
        },
        "fate": fine.fate,
        "fateIndicators": {
            candidate: int(candidate == fine.fate) for candidate in _FATES
        },
        "implementationId": IMPLEMENTATION_ID,
        "initialContact": {
            "coarseAuthenticatedSignedResidual": coarse_initial_contact_residual,
            "epsilonEventDisplacement": False,
            "exactStateTopologySide": "outward-positive",
            "fineAuthenticatedSignedResidual": fine_initial_contact_residual,
            "fineMinimumSubsequentAffineSeparation": (
                8.0 * fine_surface_options.affine_tolerance
            ),
            "coarseMinimumSubsequentAffineSeparation": (
                8.0 * coarse_surface_options.affine_tolerance
            ),
        },
        "modelOwnership": {
            "calibration": asdict(surface.calibration),
            "metric": asdict(surface.metric),
            "termination": asdict(termination),
        },
        "numericalPolicy": {
            "coarseRayOptions": asdict(coarse_ray_options),
            "coarseSurfaceOptions": asdict(coarse_surface_options),
            "fineRayOptions": asdict(fine_ray_options),
            "fineSurfaceOptions": asdict(fine_surface_options),
            "receiverEventToleranceOverMassFloor": (
                _RECEIVER_EVENT_TOLERANCE_OVER_MASS
            ),
        },
        "wholeRayConvergence": {
            "actual": asdict(convergence),
            "coarse": _evaluation_descriptor(coarse),
            "coarseRayExecutionSha256": _sha256_json(asdict(coarse.ray)),
            "fine": _evaluation_descriptor(fine),
            "fineRayExecutionSha256": _sha256_json(asdict(fine.ray)),
            "thresholds": {
                "crossingCovectorRelative": (
                    _CROSSING_COVECTOR_RELATIVE_DIFFERENCE
                ),
                "crossingEventOverMass": _CROSSING_EVENT_DIFFERENCE_OVER_MASS,
                "frequencyRatioRelative": (
                    _FREQUENCY_RATIO_RELATIVE_DIFFERENCE
                ),
                "g4Relative": _G4_RELATIVE_DIFFERENCE,
                "receiverRadiusOverMass": _RECEIVER_RADIUS_DIFFERENCE,
                "signedReceiverCosineAbsolute": _SIGNED_COSINE_DIFFERENCE,
                "terminalCovectorRelative": (
                    _TERMINAL_COVECTOR_RELATIVE_DIFFERENCE
                ),
                "terminalEventOverMass": _TERMINAL_EVENT_DIFFERENCE_OVER_MASS,
                "weightedG4Relative": _WEIGHTED_G4_RELATIVE_DIFFERENCE,
            },
        },
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrReturningRadiationRayPrimitive)
    for name, value in (
        ("launch", launch),
        ("normalized_launch", normalized_launch),
        ("surface", surface),
        ("termination", termination),
        ("ray_options", fine_ray_options),
        ("surface_options", fine_surface_options),
        ("coarse_ray_options", coarse_ray_options),
        ("coarse_surface_options", coarse_surface_options),
        ("ray", fine.ray),
        ("coarse_ray", coarse.ray),
        ("convergence", convergence),
        ("fate", fine.fate),
        ("receiver_face", fine.receiver_face),
        ("receiver", fine.receiver),
        ("receiver_radius_over_mass", fine.receiver_radius_over_mass),
        ("receiver_surface_id", fine.receiver_surface_id),
        ("emitter_direction_cosine", launch.emission_angle_cosine),
        ("receiver_incidence_cosine", fine.receiver_incidence_cosine),
        ("emitter_to_receiver_frequency_ratio", fine.frequency_ratio),
        ("bolometric_g4_factor", fine.g4),
        ("receiver_incidence_weighted_g4", fine.incidence_weighted_g4),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def trace_kerr_returning_radiation_direction(
    launch: KerrFiniteThicknessEmissionLaunch,
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
) -> KerrReturningRadiationRayPrimitive:
    """Trace and compare independent fine/coarse whole rays for one launch."""

    surface = _validated_surface(surface)
    launch = _validated_launch(launch, surface)
    normalized_launch = KerrFiniteThicknessEmissionLaunch(
        launch.frame,
        launch.emission_angle_cosine,
        launch.tangent_azimuth_rad,
        1.0,
    )
    termination = _validated_termination(
        termination,
        surface,
        normalized_launch.future_state,
    )
    fine_ray_options, fine_surface_options = _validated_options(
        surface.metric,
        ray_options,
        surface_options,
    )
    _validated_fine_options(
        surface.metric,
        fine_ray_options,
        fine_surface_options,
    )
    if (coarse_ray_options is None) != (coarse_surface_options is None):
        raise ValueError(
            "coarse_ray_options and coarse_surface_options must be supplied together"
        )
    if coarse_ray_options is None:
        coarse_ray_options, coarse_surface_options = _derive_coarse_options(
            surface.metric,
            fine_ray_options,
            fine_surface_options,
        )
    assert coarse_surface_options is not None
    coarse_ray_options, coarse_surface_options = _validated_options(
        surface.metric,
        coarse_ray_options,
        coarse_surface_options,
    )
    _validate_coarse_relationship(
        fine_ray_options,
        fine_surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )

    fine_ray, fine_contact_residual = _trace_one_resolution(
        normalized_launch,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
    )
    coarse_ray, coarse_contact_residual = _trace_one_resolution(
        normalized_launch,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
    )
    fine = _evaluate_returning_ray(
        fine_ray,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
    )
    coarse = _evaluate_returning_ray(
        coarse_ray,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
    )
    convergence = _compare_whole_rays(fine, coarse, surface.metric)
    return _build_result(
        launch=launch,
        normalized_launch=normalized_launch,
        surface=surface,
        termination=termination,
        fine_ray_options=fine_ray_options,
        fine_surface_options=fine_surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        fine=fine,
        coarse=coarse,
        convergence=convergence,
        fine_initial_contact_residual=fine_contact_residual,
        coarse_initial_contact_residual=coarse_contact_residual,
    )


def verify_kerr_returning_radiation_direction(
    result: KerrReturningRadiationRayPrimitive,
) -> None:
    """Mandatory public revalidator for every downstream result consumer.

    Both whole rays and every derived receiver quantity are rebuilt.  Exact
    equality with the reconstructed immutable result then authenticates all
    live fields as well as the canonical descriptor and its SHA-256 identity.
    """

    if type(result) is not KerrReturningRadiationRayPrimitive:
        raise TypeError(
            "result must be the exact KerrReturningRadiationRayPrimitive"
        )
    try:
        if (
            type(result._descriptor_json) is not str
            or type(result._descriptor_sha256) is not str
        ):
            raise TypeError("descriptor fields must be exact strings")
        canonical_live_descriptor = _canonical_json(
            json.loads(result._descriptor_json)
        )
        live_sha = hashlib.sha256(
            result._descriptor_json.encode("utf-8")
        ).hexdigest()
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise KerrReturningRadiationRayError(
            "returning-ray result descriptor is malformed"
        ) from error
    if (
        canonical_live_descriptor != result._descriptor_json
        or live_sha != result._descriptor_sha256
    ):
        raise KerrReturningRadiationRayError(
            "returning-ray result descriptor or SHA-256 identity is stale"
        )
    expected = trace_kerr_returning_radiation_direction(
        result.launch,
        result.surface,
        termination=result.termination,
        ray_options=result.ray_options,
        surface_options=result.surface_options,
        coarse_ray_options=result.coarse_ray_options,
        coarse_surface_options=result.coarse_surface_options,
    )
    try:
        _require_trusted_exact_tree(result, expected, "result")
    except KerrReturningRadiationRayError as error:
        raise KerrReturningRadiationRayError(
            "returning-ray live fields disagree with mandatory fine/coarse replay: "
            f"{error}"
        ) from error


def _build_process_local_issued_primitive_capability(
    canonical_tracer: Any,
) -> tuple[Any, Any]:
    """Bind a canonical tracer to one process-local, single-use consumer.

    The token class, issuance registry, and lock never enter the public API.
    Registry identity authenticates a token without serializable bearer data;
    a weak key drops abandoned issuances, while the stored PID invalidates all
    inherited capabilities after ``fork()``.  Consumption atomically removes
    the record before validating the exact primitive snapshot, so at most one
    concurrent caller can succeed and every failed attempt burns the token.
    """

    registry_lock = threading.RLock()
    registry: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
    missing = object()

    class _IssueToken:
        __slots__ = ("__weakref__",)

        def __new__(cls) -> Any:
            raise TypeError("issued primitive tokens cannot be constructed")

        def __init_subclass__(cls, **kwargs: Any) -> None:
            del kwargs
            raise TypeError("issued primitive tokens cannot be subclassed")

        def __setattr__(self, name: str, value: Any) -> None:
            del name, value
            raise TypeError("issued primitive tokens are immutable")

        def __copy__(self) -> Any:
            raise TypeError("issued primitive tokens cannot be copied")

        def __deepcopy__(self, memo: Any) -> Any:
            del memo
            raise TypeError("issued primitive tokens cannot be copied")

        def __reduce__(self) -> Any:
            raise TypeError("issued primitive tokens cannot be serialized")

        def __reduce_ex__(self, protocol: int) -> Any:
            del protocol
            raise TypeError("issued primitive tokens cannot be serialized")

        def __repr__(self) -> str:
            return "<fresh process-local Kerr ray capability>"

    @dataclass(frozen=True, slots=True)
    class _IssueRecord:
        process_id: int
        primitive: KerrReturningRadiationRayPrimitive
        exact_tree_sha256: bytes
        payload: _IssuedKerrReturningRadiationRayPayload

    def _update_sized(digest: Any, tag: bytes, data: bytes) -> None:
        digest.update(tag)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    def _update_exact_tree(digest: Any, value: Any) -> None:
        value_type = type(value)
        if is_dataclass(value) and not isinstance(value, type):
            identity = (
                f"{value_type.__module__}.{value_type.__qualname__}"
            ).encode("utf-8")
            _update_sized(digest, b"D", identity)
            dataclass_fields = fields(value)
            digest.update(len(dataclass_fields).to_bytes(8, "big"))
            for item in dataclass_fields:
                _update_sized(digest, b"N", item.name.encode("utf-8"))
                _update_exact_tree(
                    digest,
                    object.__getattribute__(value, item.name),
                )
            return
        if value_type is tuple:
            digest.update(b"T")
            digest.update(len(value).to_bytes(8, "big"))
            for item in value:
                _update_exact_tree(digest, item)
            return
        if value_type is float:
            _update_sized(digest, b"F", value.hex().encode("ascii"))
            return
        if value_type is int:
            _update_sized(digest, b"I", str(value).encode("ascii"))
            return
        if value_type is bool:
            digest.update(b"B1" if value else b"B0")
            return
        if value_type is str:
            _update_sized(digest, b"S", value.encode("utf-8"))
            return
        if value is None:
            digest.update(b"Z")
            return
        raise KerrReturningRadiationRayError(
            "issued primitive contains an unsupported exact-tree field type"
        )

    def _exact_tree_sha256(
        primitive: KerrReturningRadiationRayPrimitive,
    ) -> bytes:
        digest = hashlib.sha256()
        _update_exact_tree(digest, primitive)
        return digest.digest()

    def _sealed_payload(
        primitive: KerrReturningRadiationRayPrimitive,
    ) -> _IssuedKerrReturningRadiationRayPayload:
        fate = object.__getattribute__(primitive, "fate")
        descriptor_sha = object.__getattribute__(
            primitive,
            "_descriptor_sha256",
        )
        if type(fate) is not str or fate not in _FATES:
            raise KerrReturningRadiationRayError(
                "fresh primitive has an unsupported fate"
            )
        if (
            type(descriptor_sha) is not str
            or len(descriptor_sha) != 64
            or descriptor_sha.lower() != descriptor_sha
        ):
            raise KerrReturningRadiationRayError(
                "fresh primitive has an invalid descriptor identity"
            )
        try:
            bytes.fromhex(descriptor_sha)
        except ValueError as error:
            raise KerrReturningRadiationRayError(
                "fresh primitive has a non-hexadecimal descriptor identity"
            ) from error

        receiver_face = object.__getattribute__(primitive, "receiver_face")
        receiver_radius = object.__getattribute__(
            primitive,
            "receiver_radius_over_mass",
        )
        frequency_ratio = object.__getattribute__(
            primitive,
            "emitter_to_receiver_frequency_ratio",
        )
        coarse_face: PhotosphereFace | None = None
        coarse_radius: float | None = None
        if fate.startswith("return-"):
            if (
                type(receiver_face) is not str
                or receiver_face not in (UPPER, LOWER)
                or type(receiver_radius) is not float
                or not math.isfinite(receiver_radius)
                or receiver_radius <= 0.0
                or type(frequency_ratio) is not float
                or not math.isfinite(frequency_ratio)
                or frequency_ratio <= 0.0
            ):
                raise KerrReturningRadiationRayError(
                    "fresh returned primitive has invalid receiver data"
                )
            try:
                descriptor = json.loads(
                    object.__getattribute__(primitive, "_descriptor_json")
                )
                coarse_receiver = descriptor["wholeRayConvergence"]["coarse"][
                    "receiver"
                ]
                coarse_face = coarse_receiver["face"]
                coarse_radius = coarse_receiver[
                    "pseudoCylindricalRadiusOverMass"
                ]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise KerrReturningRadiationRayError(
                    "fresh returned primitive lacks coarse receiver evidence"
                ) from error
            if (
                type(coarse_receiver) is not dict
                or type(coarse_face) is not str
                or coarse_face not in (UPPER, LOWER)
                or type(coarse_radius) is not float
                or not math.isfinite(coarse_radius)
                or coarse_radius <= 0.0
            ):
                raise KerrReturningRadiationRayError(
                    "fresh coarse receiver evidence has invalid exact fields"
                )
        elif any(
            value is not None
            for value in (receiver_face, receiver_radius, frequency_ratio)
        ):
            raise KerrReturningRadiationRayError(
                "fresh non-returning primitive carries receiver data"
            )
        return _IssuedKerrReturningRadiationRayPayload(
            fate,
            receiver_face,
            receiver_radius,
            frequency_ratio,
            descriptor_sha,
            coarse_face,
            coarse_radius,
        )

    def issue(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        primitive = canonical_tracer(*args, **kwargs)
        if type(primitive) is not KerrReturningRadiationRayPrimitive:
            raise TypeError("canonical tracer returned a non-exact ray primitive")
        payload = _sealed_payload(primitive)
        snapshot = _exact_tree_sha256(primitive)
        token = object.__new__(_IssueToken)
        record = _IssueRecord(os.getpid(), primitive, snapshot, payload)
        with registry_lock:
            registry[token] = record
        return primitive, token

    def consume(
        primitive: KerrReturningRadiationRayPrimitive,
        token: object,
    ) -> _IssuedKerrReturningRadiationRayPayload:
        if type(token) is not _IssueToken:
            raise TypeError("issued primitive token has a non-exact private type")
        if type(primitive) is not KerrReturningRadiationRayPrimitive:
            raise TypeError("issued result must be the exact ray primitive")
        with registry_lock:
            record = registry.pop(token, missing)
        if record is missing:
            raise KerrReturningRadiationRayError(
                "issued primitive token is forged, stale, or already consumed"
            )
        if record.process_id != os.getpid():
            raise KerrReturningRadiationRayError(
                "issued primitive token belongs to a different process"
            )
        if primitive is not record.primitive:
            raise KerrReturningRadiationRayError(
                "issued primitive token does not own this exact result"
            )
        if not hmac.compare_digest(
            _exact_tree_sha256(primitive),
            record.exact_tree_sha256,
        ):
            raise KerrReturningRadiationRayError(
                "issued primitive changed after canonical tracing"
            )
        return record.payload

    issue.__name__ = "_trace_issued_kerr_returning_radiation_direction"
    consume.__name__ = "_consume_issued_kerr_returning_radiation_direction"

    return issue, consume


(
    _trace_issued_kerr_returning_radiation_direction,
    _consume_issued_kerr_returning_radiation_direction,
) = _build_process_local_issued_primitive_capability(
    trace_kerr_returning_radiation_direction
)


__all__ = (
    "IMPLEMENTATION_ID",
    "KerrReturningRadiationRayError",
    "KerrReturningRadiationRayConvergence",
    "KerrReturningRadiationRayPrimitive",
    "PLUNGE_LOWER_TARGET_ID",
    "PLUNGE_OUTCOME",
    "PLUNGE_UPPER_TARGET_ID",
    "RETURNED_OUTCOME",
    "ReturningRayFate",
    "SCIENTIFIC_STATUS",
    "trace_kerr_returning_radiation_direction",
    "verify_kerr_returning_radiation_direction",
)
