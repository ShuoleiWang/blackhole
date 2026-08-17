"""Receiver-centred backward-sky Kerr returning-radiation ray primitive.

For one authenticated finite-thickness receiver frame and local incoming
direction, this module constructs the future photon

``k = u - mu_i n + sqrt(1-mu_i^2)(cos(psi_i)e_rho + sin(psi_i)e_phi)``

at receiver frequency ``nu_i=1`` and traces the past-directed covector ``-k``.
The first physical upper/lower photosphere hit is a source candidate; a valid
candidate must be outgoing through that source face (``mu_e>0``).  Otherwise
the trace fails closed.  If a past worldtube is reached first, the public fate
is simply ``past-worldtube-no-disk-source``--never a future capture/escape
claim.

For a valid source, ``g=nu_i/nu_e=1/nu_e`` and the reported directional
receiver integrand is ``mu_i*g^4*D20(mu_e)``, with
``D20(mu)=1/2+3mu/4``.  This still lacks receiver solid-angle integration and
source/receiver area Jacobians.  It is one directional integrand, not a
returning-radiation kernel ``K``, incident flux ``F_in``, KERRBB, or ``F_S``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    HamiltonianState,
    InitialMultiSurfaceContact,
    RayTraceOptions,
    RayTraceResult,
    SurfaceEventOptions,
    hamiltonian_null_residual,
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
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    OPAQUE_OUTCOME,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.spacetime import Vector4, matrix_vector


IMPLEMENTATION_ID: Final = "finite-thickness-kerr-receiver-backward-ray/v1"
PAST_WORLDTUBE_NO_SOURCE: Final = "past-worldtube-no-disk-source"
ReceiverRayOutcome = Literal[
    "source-upper",
    "source-lower",
    "past-worldtube-no-disk-source",
]

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "receiver-centred single-direction backward-sky finite-thickness "
            "Kerr source transport primitive"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "receiverLaunch": (
            "future incoming k at nu_i=1; trace past-directed -k"
        ),
        "initialContact": (
            "public authenticated multi-surface affine-zero +1 side; every "
            "positive-affine probe uses the physical signed photosphere"
        ),
        "surfaceCompleteness": (
            "independent fine/coarse whole rays, each with accepted-step N/2N "
            "finite-probe topology"
        ),
        "sourceGate": "first disk source must have front-face outgoing mu_e>0",
        "frequencyConvention": "nu_i=1 and g=nu_i/nu_e=1/nu_e",
        "directionalIntegrand": "mu_i*g^4*(1/2+3mu_e/4)",
        "pastWorldtubeSemantics": PAST_WORLDTUBE_NO_SOURCE,
        "requiresPublicRevalidationBeforeConsumption": True,
        "isReceiverCentredBackwardSkyPrimitive": True,
        "includesReceiverSolidAngleIntegration": False,
        "includesAreaJacobian": False,
        "outputsReturningRadiationKernelK": False,
        "isReceiverCentredIncidentFlux": False,
        "includesReturningRadiationStressWorkFS": False,
        "isCompleteKerrbb": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe one directional mu_i*g^4*D20 value as K, F_in, "
            "complete KERRBB, F_S, a solved atmosphere, or GRMHD."
        ),
    }
)

_INITIAL_CONTACT_MAXIMUM_RESIDUAL: Final = 2.0e-11
_LOCAL_NULL_RESIDUAL_LIMIT: Final = 2.0e-10
_SOURCE_NULL_RESIDUAL_LIMIT: Final = 1.0e-6
_SOURCE_EVENT_TOLERANCE_OVER_MASS: Final = 2.0e-8
_DEFAULT_COARSE_TOLERANCE_MULTIPLIER: Final = 8.0
_MAXIMUM_COARSE_TOLERANCE_MULTIPLIER: Final = 64.0
_TERMINAL_EVENT_DIFFERENCE_OVER_MASS: Final = 2.0e-5
_TERMINAL_COVECTOR_RELATIVE_DIFFERENCE: Final = 2.0e-5
_CROSSING_EVENT_DIFFERENCE_OVER_MASS: Final = 2.0e-5
_CROSSING_COVECTOR_RELATIVE_DIFFERENCE: Final = 2.0e-5
_SOURCE_RADIUS_DIFFERENCE: Final = 2.0e-5
_SOURCE_COSINE_DIFFERENCE: Final = 2.0e-5
_SOURCE_FREQUENCY_RELATIVE_DIFFERENCE: Final = 2.0e-5
_FREQUENCY_RATIO_RELATIVE_DIFFERENCE: Final = 2.0e-5
_G4_RELATIVE_DIFFERENCE: Final = 8.0e-5
_INTEGRAND_RELATIVE_DIFFERENCE: Final = 1.0e-4


class KerrReturningRadiationReceiverRayError(RuntimeError):
    """Raised when a receiver-centred direction cannot be certified."""


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
        raise KerrReturningRadiationReceiverRayError(
            "receiver-ray descriptor is not finite canonical JSON"
        ) from error


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationReceiverRayError(
            f"{path}.{name} is missing from the authenticated schema"
        ) from error


def _require_exact_schema_types(actual: Any, template: Any, path: str) -> None:
    """Validate exact field types without invoking untrusted comparisons."""

    if type(actual) is not type(template):
        raise KerrReturningRadiationReceiverRayError(
            f"{path} has non-exact field type {type(actual).__name__}; "
            f"expected {type(template).__name__}"
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
            raise KerrReturningRadiationReceiverRayError(
                f"{path} tuple length differs from its schema"
            )
        for index, (actual_item, template_item) in enumerate(
            zip(actual, template)
        ):
            _require_exact_schema_types(
                actual_item,
                template_item,
                f"{path}[{index}]",
            )
        return
    if type(template) not in (float, int, bool, str, type(None)):
        raise KerrReturningRadiationReceiverRayError(
            f"{path} uses unsupported schema type {type(template).__name__}"
        )


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare reconstructed data without invoking untrusted equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationReceiverRayError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, field.name, path),
                _trusted_attribute(expected, field.name, path),
                f"{path}.{field.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationReceiverRayError(
                f"{path} tuple length differs from replay"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _require_trusted_exact_tree(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationReceiverRayError(
            f"{path} uses unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationReceiverRayError(
            f"{path} differs from the trusted replay"
        )


def _exact_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value


def _linear_combination(terms: tuple[tuple[float, Vector4], ...]) -> Vector4:
    result = tuple(
        math.fsum(scale * vector[index] for scale, vector in terms)
        for index in range(4)
    )
    if not all(math.isfinite(value) for value in result):
        raise KerrReturningRadiationReceiverRayError(
            "receiver local-frame vector is non-finite"
        )
    return result  # type: ignore[return-value]


def _rebuild_metric(metric: KerrKerrSchildMetric) -> KerrKerrSchildMetric:
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must be the exact Kerr metric")
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
        raise TypeError("surface calibration must be the exact built-in type")
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
    _require_trusted_exact_tree(calibration, expected, "surface.calibration")
    return expected


def _validated_surface(
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessMultiSurface:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be the exact finite-thickness multi-surface")
    expected = KerrFiniteThicknessMultiSurface(
        _rebuild_metric(surface.metric),
        _rebuild_calibration(surface.calibration),
    )
    _require_trusted_exact_tree(surface, expected, "surface")
    return expected


def _validated_receiver_frame(
    frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessSurfaceFrame:
    if type(frame) is not KerrFiniteThicknessSurfaceFrame:
        raise TypeError("receiver_frame must be the exact surface-frame type")
    emitter = frame.emitter
    if type(emitter) is not KerrFiniteThicknessFaceEmitter:
        raise TypeError("receiver emitter must be the exact finite-thickness type")
    _require_trusted_exact_tree(
        emitter.metric,
        surface.metric,
        "receiver_frame.emitter.metric",
    )
    _require_trusted_exact_tree(
        emitter.calibration,
        surface.calibration,
        "receiver_frame.emitter.calibration",
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
    expected = KerrFiniteThicknessSurfaceFrame(expected_emitter)
    _require_trusted_exact_tree(frame, expected, "receiver_frame")
    return expected


def _face_surface_id(face: PhotosphereFace) -> str:
    if face == UPPER:
        return UPPER_SURFACE_ID
    if face == LOWER:
        return LOWER_SURFACE_ID
    raise KerrReturningRadiationReceiverRayError("unknown photosphere face")


def _surface_face(surface_id: str) -> PhotosphereFace:
    if surface_id == UPPER_SURFACE_ID:
        return UPPER
    if surface_id == LOWER_SURFACE_ID:
        return LOWER
    raise KerrReturningRadiationReceiverRayError("unknown photosphere surface id")


def _build_receiver_states(
    frame: KerrFiniteThicknessSurfaceFrame,
    incidence_cosine: float,
    tangent_azimuth_rad: float,
) -> tuple[float, HamiltonianState, HamiltonianState, float]:
    mu = _exact_finite_float(incidence_cosine, "receiver_incidence_cosine")
    if mu <= 0.0 or mu > 1.0:
        raise ValueError("receiver_incidence_cosine must lie in (0, 1]")
    raw_azimuth = _exact_finite_float(
        tangent_azimuth_rad,
        "receiver_tangent_azimuth_rad",
    )
    azimuth = raw_azimuth % (2.0 * math.pi)
    tangent_weight = math.sqrt(max(0.0, 1.0 - mu * mu))
    future_vector = _linear_combination(
        (
            (1.0, frame.emitter.four_velocity),
            (-mu, frame.emitter.outward_unit_normal),
            (
                tangent_weight * math.cos(azimuth),
                frame.meridional_tangent,
            ),
            (
                tangent_weight * math.sin(azimuth),
                frame.azimuthal_tangent,
            ),
        )
    )
    sample = frame.emitter.metric.sample(frame.emitter.event)
    future_covector = matrix_vector(sample.covariant, future_vector)
    future_state = HamiltonianState(frame.emitter.event, future_covector)
    past_state = HamiltonianState(
        frame.emitter.event,
        tuple(-value for value in future_covector),
    )
    residual = hamiltonian_null_residual(frame.emitter.metric, past_state)
    if not math.isfinite(residual) or residual > _LOCAL_NULL_RESIDUAL_LIMIT:
        raise KerrReturningRadiationReceiverRayError(
            "receiver incoming photon is not null"
        )
    projection = frame.emitter.project_past_directed_photon(
        past_state,
        null_residual_limit=_LOCAL_NULL_RESIDUAL_LIMIT,
        backside_policy="classify",
    )
    if (
        projection.face_classification != "backside"
        or projection.outgoing_cosine >= 0.0
        or not math.isclose(
            -projection.outgoing_cosine,
            mu,
            rel_tol=5.0e-10,
            abs_tol=5.0e-12,
        )
        or not math.isclose(
            projection.local_frequency,
            1.0,
            rel_tol=5.0e-10,
            abs_tol=0.0,
        )
    ):
        raise KerrReturningRadiationReceiverRayError(
            "receiver incoming direction disagrees with its local frame"
        )
    return azimuth, future_state, past_state, residual


def _validated_termination(
    termination: KerrOblateTermination,
    surface: KerrFiniteThicknessMultiSurface,
    initial_state: HamiltonianState,
) -> KerrOblateTermination:
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be the exact Kerr oblate termination")
    expected = KerrOblateTermination(
        spin_a_m=termination.spin_a_m,
        capture_radius_m=termination.capture_radius_m,
        escape_radius_m=termination.escape_radius_m,
        capture_target_id=termination.capture_target_id,
        escape_target_id=termination.escape_target_id,
    )
    _require_trusted_exact_tree(termination, expected, "termination")
    metric = surface.metric
    if expected.spin_a_m.hex() != metric.spin_a_m.hex():
        raise ValueError("termination and Kerr metric signed spins disagree")
    inner = surface.calibration.isco_radius_over_mass * metric.mass_m
    outer = surface.calibration.photosphere_point(
        surface.calibration.outer_radius_over_mass,
        UPPER,
    ).radius_over_mass * metric.mass_m
    if expected.capture_radius_m < metric.outer_horizon_radius_m:
        raise ValueError("past inner worldtube may not lie inside the horizon")
    if expected.capture_radius_m >= inner:
        raise ValueError("past inner worldtube must lie strictly inside the ISCO")
    if expected.escape_radius_m <= outer:
        raise ValueError("past outer worldtube must enclose the photosphere")
    if expected.classify_initial(initial_state) is not None:
        raise ValueError("receiver starts on or beyond a past worldtube")
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
    # Constructors accept numerically compatible values such as ``True`` or
    # integer ``1`` for several float fields.  Prove the exact public schema
    # before reconstruction, equality checks, policy comparisons, or
    # descriptor generation so hostile numeric subclasses cannot participate
    # in any scientific decision.
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
    try:
        expected_ray = RayTraceOptions(**asdict(ray_options))
        expected_surface = SurfaceEventOptions(**asdict(surface_options))
    except (TypeError, ValueError) as error:
        raise ValueError(f"receiver-ray options are stale: {error}") from error
    _require_trusted_exact_tree(ray_options, expected_ray, "ray_options")
    _require_trusted_exact_tree(surface_options, expected_surface, "surface_options")
    scale = max(1.0, metric.mass_m)
    if (
        expected_ray.absolute_tolerance > 1.0e-7 * scale
        or expected_ray.relative_tolerance > 1.0e-7
        or expected_ray.maximum_step > 2.0 * scale
        or expected_ray.null_residual_limit > 1.0e-6
        or expected_ray.metric_interpolation_error_limit > 1.0e-6
        or expected_ray.event_value_tolerance > 1.0e-7 * scale
        or expected_ray.event_affine_tolerance > 1.0e-7 * scale
        or expected_ray.maximum_affine_length > 1.0e6 * scale
        or expected_ray.maximum_accepted_steps > 1_000_000
        or expected_ray.maximum_rejected_steps > 1_000_000
        or expected_ray.event_maximum_iterations > 256
    ):
        raise ValueError("ray options exceed the receiver-ray policy")
    if (
        expected_surface.absolute_tolerance > 1.0e-7 * scale
        or expected_surface.relative_tolerance > 1.0e-7
        or expected_surface.null_residual_limit > 1.0e-6
        or expected_surface.metric_interpolation_error_limit > 1.0e-6
        or expected_surface.surface_value_tolerance > 1.0e-7
        or expected_surface.affine_tolerance > 1.0e-7 * scale
        or expected_surface.subdivisions_per_segment < 2
        or expected_surface.subdivisions_per_segment > 128
        or expected_surface.maximum_iterations > 256
        or expected_surface.maximum_reintegrations > 2_000_000
    ):
        raise ValueError("surface options exceed the receiver-ray policy")
    return expected_ray, expected_surface


def _validated_fine_options(
    metric: KerrKerrSchildMetric,
    ray: RayTraceOptions,
    surface: SurfaceEventOptions,
) -> None:
    scale = max(1.0, metric.mass_m)
    if (
        ray.absolute_tolerance > 1.0e-8 * scale
        or ray.relative_tolerance > 1.0e-8
        or ray.maximum_step > scale
        or ray.event_value_tolerance > 1.0e-8 * scale
        or ray.event_affine_tolerance > 1.0e-8 * scale
        or surface.absolute_tolerance > 1.0e-8 * scale
        or surface.relative_tolerance > 1.0e-8
        or surface.surface_value_tolerance > 1.0e-8
        or surface.affine_tolerance > 1.0e-8 * scale
        or surface.subdivisions_per_segment < 4
    ):
        raise ValueError("fine options exceed the receiver-ray accuracy policy")


def _derive_coarse_options(
    metric: KerrKerrSchildMetric,
    ray: RayTraceOptions,
    surface: SurfaceEventOptions,
) -> tuple[RayTraceOptions, SurfaceEventOptions]:
    factor = _DEFAULT_COARSE_TOLERANCE_MULTIPLIER
    scale = max(1.0, metric.mass_m)
    maximum_step = min(
        2.0 * scale,
        max(2.0 * ray.maximum_step, 2.0 * ray.initial_step),
    )
    initial_step = min(
        maximum_step,
        max(2.0 * ray.initial_step, ray.minimum_step),
    )
    coarse_ray = replace(
        ray,
        absolute_tolerance=min(1.0e-7 * scale, factor * ray.absolute_tolerance),
        relative_tolerance=min(1.0e-7, factor * ray.relative_tolerance),
        initial_step=initial_step,
        minimum_step=min(initial_step, factor * ray.minimum_step),
        maximum_step=maximum_step,
        null_residual_limit=min(1.0e-6, factor * ray.null_residual_limit),
        metric_interpolation_error_limit=min(
            1.0e-6,
            factor * ray.metric_interpolation_error_limit,
        ),
        event_value_tolerance=min(
            1.0e-7 * scale,
            factor * ray.event_value_tolerance,
        ),
        event_affine_tolerance=min(
            1.0e-7 * scale,
            factor * ray.event_affine_tolerance,
        ),
    )
    coarse_surface = replace(
        surface,
        absolute_tolerance=min(
            1.0e-7 * scale,
            factor * surface.absolute_tolerance,
        ),
        relative_tolerance=min(1.0e-7, factor * surface.relative_tolerance),
        null_residual_limit=min(
            1.0e-6,
            factor * surface.null_residual_limit,
        ),
        metric_interpolation_error_limit=min(
            1.0e-6,
            factor * surface.metric_interpolation_error_limit,
        ),
        surface_value_tolerance=min(
            1.0e-7,
            factor * surface.surface_value_tolerance,
        ),
        affine_tolerance=min(
            1.0e-7 * scale,
            factor * surface.affine_tolerance,
        ),
        subdivisions_per_segment=max(
            2,
            surface.subdivisions_per_segment // 2,
        ),
    )
    return coarse_ray, coarse_surface


def _validate_coarse_relationship(
    fine_ray: RayTraceOptions,
    fine_surface: SurfaceEventOptions,
    coarse_ray: RayTraceOptions,
    coarse_surface: SurfaceEventOptions,
) -> None:
    ray_ratios = tuple(
        coarse / fine
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
        coarse / fine
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
        raise ValueError("coarse tolerances must lie between 1x and 64x fine")
    if (
        coarse_ray.initial_step < fine_ray.initial_step
        or coarse_ray.minimum_step < fine_ray.minimum_step
        or coarse_ray.maximum_step < fine_ray.maximum_step
        or coarse_surface.subdivisions_per_segment
        > fine_surface.subdivisions_per_segment
    ):
        raise ValueError("coarse discretization must not be finer than fine")
    if not (
        any(ratio > 1.0 for ratio in (*ray_ratios, *surface_ratios))
        or coarse_ray.initial_step > fine_ray.initial_step
        or coarse_ray.minimum_step > fine_ray.minimum_step
        or coarse_ray.maximum_step > fine_ray.maximum_step
        or coarse_surface.subdivisions_per_segment
        < fine_surface.subdivisions_per_segment
    ):
        raise ValueError("coarse whole ray must be materially different")


def _trace_one_resolution(
    frame: KerrFiniteThicknessSurfaceFrame,
    past_state: HamiltonianState,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> RayTraceResult:
    receiver_surface_id = _face_surface_id(frame.emitter.face)
    residual = surface.value(receiver_surface_id, past_state)
    if abs(residual) > _INITIAL_CONTACT_MAXIMUM_RESIDUAL:
        raise KerrReturningRadiationReceiverRayError(
            "authenticated receiver is not on its declared face"
        )
    try:
        ray = trace_null_geodesic(
            surface.metric,
            past_state,
            termination=termination,
            multi_interior_surface=surface,
            initial_multi_surface_contact=InitialMultiSurfaceContact(
                receiver_surface_id,
                1,
            ),
            surface_options=surface_options,
            options=ray_options,
        )
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
        raise KerrReturningRadiationReceiverRayError(
            f"past receiver-ray trace failed: {error}"
        ) from error
    if ray.failure_reason is not None or ray.outcome in (
        "integrator-failure",
        "unresolved",
        "completed",
    ):
        raise KerrReturningRadiationReceiverRayError(
            "past receiver ray did not reach a certified source/worldtube: "
            f"{ray.outcome}: {ray.failure_reason or 'no terminal'}"
        )
    trace = ray.multi_surface_trace
    if trace is None or not trace.topology_converged or trace.initial_contact is None:
        raise KerrReturningRadiationReceiverRayError(
            "past receiver ray lacks authenticated N/2N topology"
        )
    contact = trace.initial_contact
    if (
        contact.surface_id != receiver_surface_id
        or contact.side != 1
        or contact.actual_surface_value.hex() != float(residual).hex()
        or trace.surface_ids != tuple(sorted(FINITE_THICKNESS_SURFACE_IDS))
        or trace.base_subdivisions_per_step
        != surface_options.subdivisions_per_segment
        or trace.verification_subdivisions_per_step
        != 2 * surface_options.subdivisions_per_segment
    ):
        raise KerrReturningRadiationReceiverRayError(
            "past receiver-ray topology provenance is stale"
        )
    return ray


@dataclass(frozen=True, slots=True)
class _EvaluatedReceiverRay:
    ray: RayTraceResult
    outcome: ReceiverRayOutcome
    terminal_rho_over_mass: float
    past_worldtube_target_id: str | None
    past_worldtube_radius_m: float | None
    source_face: PhotosphereFace | None
    source: KerrFiniteThicknessFaceEmitter | None
    source_surface_id: str | None
    source_radius_over_mass: float | None
    source_emission_cosine: float | None
    source_local_frequency: float | None
    frequency_ratio: float | None
    g4: float | None
    d20_multiplier: float | None
    receiver_integrand: float


def _terminal_rho(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
) -> float:
    oblate = kerr_ks_event_to_oblate(metric, state.event)
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / metric.mass_m
    if not math.isfinite(rho) or rho < 0.0:
        raise KerrReturningRadiationReceiverRayError(
            "terminal oblate radius is invalid"
        )
    return rho


def _evaluate_ray(
    ray: RayTraceResult,
    receiver_incidence_cosine: float,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
) -> _EvaluatedReceiverRay:
    trace = ray.multi_surface_trace
    if trace is None:
        raise KerrReturningRadiationReceiverRayError("ray lost multi-surface trace")
    terminal_rho = _terminal_rho(surface.metric, ray.terminal_state)
    terminal_entry = trace.crossings[-1] if trace.crossings else None
    if ray.outcome in ("captured", "escaped"):
        expected_target = (
            termination.capture_target_id
            if ray.outcome == "captured"
            else termination.escape_target_id
        )
        expected_radius = (
            termination.capture_radius_m
            if ray.outcome == "captured"
            else termination.escape_radius_m
        )
        if ray.terminal_target_id != expected_target:
            raise KerrReturningRadiationReceiverRayError(
                "past worldtube target is not owned by termination"
            )
        if terminal_entry is not None and terminal_entry.decision.terminates:
            raise KerrReturningRadiationReceiverRayError(
                "past worldtube ray hides a disk source"
            )
        actual_radius = termination.radius(ray.terminal_state)
        tolerance = max(
            2.0 * ray_options.event_value_tolerance,
            32.0 * math.ulp(
                max(1.0, expected_radius, surface.metric.mass_m)
            ),
        )
        if abs(actual_radius - expected_radius) > tolerance:
            raise KerrReturningRadiationReceiverRayError(
                "past worldtube terminal is off its declared boundary"
            )
        return _EvaluatedReceiverRay(
            ray=ray,
            outcome=PAST_WORLDTUBE_NO_SOURCE,
            terminal_rho_over_mass=terminal_rho,
            past_worldtube_target_id=expected_target,
            past_worldtube_radius_m=actual_radius,
            source_face=None,
            source=None,
            source_surface_id=None,
            source_radius_over_mass=None,
            source_emission_cosine=None,
            source_local_frequency=None,
            frequency_ratio=None,
            g4=None,
            d20_multiplier=None,
            receiver_integrand=0.0,
        )
    if ray.outcome != OPAQUE_OUTCOME or terminal_entry is None:
        raise KerrReturningRadiationReceiverRayError(
            "surface terminal is not a supported disk source"
        )
    if (
        not terminal_entry.decision.terminates
        or terminal_entry.decision.outcome != ray.outcome
        or terminal_entry.decision.target_id != ray.terminal_target_id
        or terminal_entry.crossing.state != ray.terminal_state
        or terminal_entry.crossing.orientation != -1
        or terminal_entry.crossing.ray_affine_length
        <= 8.0 * surface_options.affine_tolerance
    ):
        raise KerrReturningRadiationReceiverRayError(
            "disk source does not own a resolved positive-affine terminal"
        )
    face = _surface_face(terminal_entry.surface_id)
    expected_target = UPPER_TARGET_ID if face == UPPER else LOWER_TARGET_ID
    if terminal_entry.decision.target_id != expected_target:
        raise KerrReturningRadiationReceiverRayError(
            "disk source target disagrees with its face"
        )
    oblate = kerr_ks_event_to_oblate(
        surface.metric,
        terminal_entry.crossing.state.event,
    )
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / surface.metric.mass_m
    source = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=rho,
        face=face,
        phi_ks_rad=oblate.phi_ks_rad,
        coordinate_time_m=terminal_entry.crossing.state.event[0],
    )
    event_tolerance = max(
        _SOURCE_EVENT_TOLERANCE_OVER_MASS * surface.metric.mass_m,
        2.0 * surface_options.surface_value_tolerance * surface.metric.mass_m,
    )
    if max(
        abs(actual - expected)
        for actual, expected in zip(
            terminal_entry.crossing.state.event,
            source.event,
        )
    ) > event_tolerance:
        raise KerrReturningRadiationReceiverRayError(
            "localized source is too far from its reconstructed face"
        )
    projection = source.project_past_directed_photon(
        terminal_entry.crossing.state,
        null_residual_limit=min(
            _SOURCE_NULL_RESIDUAL_LIMIT,
            surface_options.null_residual_limit,
        ),
        event_tolerance_m=event_tolerance,
        backside_policy="classify",
    )
    if (
        projection.face_classification != "outgoing"
        or projection.outgoing_cosine <= 0.0
    ):
        raise KerrReturningRadiationReceiverRayError(
            "first disk source is not front-face outgoing"
        )
    mu_e = projection.outgoing_cosine
    nu_e = projection.local_frequency
    g = 1.0 / nu_e
    try:
        g4 = g**4
    except OverflowError as error:
        raise KerrReturningRadiationReceiverRayError(
            "receiver-source g^4 overflowed"
        ) from error
    angular_law = FluxConservingLinearLimbDarkening()
    if (
        type(angular_law) is not FluxConservingLinearLimbDarkening
        or angular_law.coefficient.hex() != float(1.5).hex()
    ):
        raise KerrReturningRadiationReceiverRayError(
            "built-in KERRBB D20 law identity is unavailable"
        )
    d20 = angular_law.intensity_multiplier(mu_e)
    integrand = receiver_incidence_cosine * g4 * d20
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (nu_e, g, g4, d20, integrand)
    ):
        raise KerrReturningRadiationReceiverRayError(
            "receiver directional transport factor is invalid"
        )
    return _EvaluatedReceiverRay(
        ray=ray,
        outcome="source-upper" if face == UPPER else "source-lower",
        terminal_rho_over_mass=terminal_rho,
        past_worldtube_target_id=None,
        past_worldtube_radius_m=None,
        source_face=face,
        source=source,
        source_surface_id=terminal_entry.surface_id,
        source_radius_over_mass=rho,
        source_emission_cosine=mu_e,
        source_local_frequency=nu_e,
        frequency_ratio=g,
        g4=g4,
        d20_multiplier=d20,
        receiver_integrand=integrand,
    )


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


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-300)


def _topology_signature(ray: RayTraceResult) -> tuple[Any, ...]:
    trace = ray.multi_surface_trace
    if trace is None:
        raise KerrReturningRadiationReceiverRayError("ray lacks topology")
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
class KerrReturningRadiationReceiverRayConvergence:
    """Independent whole-ray and source-observable agreement diagnostics."""

    raw_outcome_agrees: bool
    terminal_target_agrees: bool
    receiver_outcome_agrees: bool
    complete_topology_agrees: bool
    terminal_event_difference_m: float
    terminal_covector_relative_difference: float
    maximum_crossing_event_difference_m: float
    maximum_crossing_covector_relative_difference: float
    source_radius_difference_over_mass: float | None
    source_emission_cosine_difference: float | None
    source_frequency_relative_difference: float | None
    frequency_ratio_relative_difference: float | None
    g4_relative_difference: float | None
    receiver_integrand_relative_difference: float
    past_worldtube_radius_difference_m: float | None
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "raw_outcome_agrees",
            "terminal_target_agrees",
            "receiver_outcome_agrees",
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
            "source_radius_difference_over_mass",
            "source_emission_cosine_difference",
            "source_frequency_relative_difference",
            "frequency_ratio_relative_difference",
            "g4_relative_difference",
            "receiver_integrand_relative_difference",
            "past_worldtube_radius_difference_m",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative float")
        if not self.converged:
            raise ValueError("non-converged receiver rays may not form a result")


def _compare_rays(
    fine: _EvaluatedReceiverRay,
    coarse: _EvaluatedReceiverRay,
    metric: KerrKerrSchildMetric,
) -> KerrReturningRadiationReceiverRayConvergence:
    raw_outcome = fine.ray.outcome == coarse.ray.outcome
    target = fine.ray.terminal_target_id == coarse.ray.terminal_target_id
    receiver_outcome = fine.outcome == coarse.outcome
    topology = _topology_signature(fine.ray) == _topology_signature(coarse.ray)
    fine_trace = fine.ray.multi_surface_trace
    coarse_trace = coarse.ray.multi_surface_trace
    if fine_trace is None or coarse_trace is None:
        raise KerrReturningRadiationReceiverRayError("whole ray lacks topology")
    contacts_agree = (
        fine_trace.initial_contact is not None
        and coarse_trace.initial_contact is not None
        and fine_trace.initial_contact.surface_id
        == coarse_trace.initial_contact.surface_id
        and fine_trace.initial_contact.side == coarse_trace.initial_contact.side
        and fine_trace.initial_contact.actual_surface_value.hex()
        == coarse_trace.initial_contact.actual_surface_value.hex()
    )
    topology = topology and contacts_agree
    terminal_event = _event_difference(
        fine.ray.terminal_state,
        coarse.ray.terminal_state,
    )
    terminal_covector = _covector_relative_difference(
        fine.ray.terminal_state,
        coarse.ray.terminal_state,
    )
    maximum_crossing_event = 0.0
    maximum_crossing_covector = 0.0
    if len(fine_trace.crossings) != len(coarse_trace.crossings):
        topology = False
    else:
        for fine_entry, coarse_entry in zip(
            fine_trace.crossings,
            coarse_trace.crossings,
        ):
            maximum_crossing_event = max(
                maximum_crossing_event,
                _event_difference(
                    fine_entry.crossing.state,
                    coarse_entry.crossing.state,
                ),
            )
            maximum_crossing_covector = max(
                maximum_crossing_covector,
                _covector_relative_difference(
                    fine_entry.crossing.state,
                    coarse_entry.crossing.state,
                ),
            )
    if not (raw_outcome and target and receiver_outcome and topology):
        raise KerrReturningRadiationReceiverRayError(
            "independent fine/coarse receiver-ray topology disagrees"
        )
    scale = metric.mass_m
    if (
        terminal_event > _TERMINAL_EVENT_DIFFERENCE_OVER_MASS * scale
        or terminal_covector > _TERMINAL_COVECTOR_RELATIVE_DIFFERENCE
        or maximum_crossing_event
        > _CROSSING_EVENT_DIFFERENCE_OVER_MASS * scale
        or maximum_crossing_covector
        > _CROSSING_COVECTOR_RELATIVE_DIFFERENCE
    ):
        raise KerrReturningRadiationReceiverRayError(
            "independent fine/coarse receiver-ray terminal state disagrees"
        )

    source_radius = None
    source_cosine = None
    source_frequency = None
    frequency_ratio = None
    g4 = None
    worldtube_radius = None
    integrand = _relative_difference(
        fine.receiver_integrand,
        coarse.receiver_integrand,
    ) if fine.receiver_integrand != 0.0 or coarse.receiver_integrand != 0.0 else 0.0
    if fine.source is not None or coarse.source is not None:
        if (
            fine.source is None
            or coarse.source is None
            or fine.source_face != coarse.source_face
            or fine.source_surface_id != coarse.source_surface_id
            or fine.source_radius_over_mass is None
            or coarse.source_radius_over_mass is None
            or fine.source_emission_cosine is None
            or coarse.source_emission_cosine is None
            or fine.source_local_frequency is None
            or coarse.source_local_frequency is None
            or fine.frequency_ratio is None
            or coarse.frequency_ratio is None
            or fine.g4 is None
            or coarse.g4 is None
        ):
            raise KerrReturningRadiationReceiverRayError(
                "fine/coarse disk source schema disagrees"
            )
        source_radius = abs(
            fine.source_radius_over_mass - coarse.source_radius_over_mass
        )
        source_cosine = abs(
            fine.source_emission_cosine - coarse.source_emission_cosine
        )
        source_frequency = _relative_difference(
            fine.source_local_frequency,
            coarse.source_local_frequency,
        )
        frequency_ratio = _relative_difference(
            fine.frequency_ratio,
            coarse.frequency_ratio,
        )
        g4 = _relative_difference(fine.g4, coarse.g4)
        if (
            source_radius > _SOURCE_RADIUS_DIFFERENCE
            or source_cosine > _SOURCE_COSINE_DIFFERENCE
            or source_frequency > _SOURCE_FREQUENCY_RELATIVE_DIFFERENCE
            or frequency_ratio > _FREQUENCY_RATIO_RELATIVE_DIFFERENCE
            or g4 > _G4_RELATIVE_DIFFERENCE
            or integrand > _INTEGRAND_RELATIVE_DIFFERENCE
        ):
            raise KerrReturningRadiationReceiverRayError(
                "independent fine/coarse source transport disagrees"
            )
    else:
        if (
            fine.past_worldtube_target_id is None
            or coarse.past_worldtube_target_id is None
            or fine.past_worldtube_target_id != coarse.past_worldtube_target_id
            or fine.past_worldtube_radius_m is None
            or coarse.past_worldtube_radius_m is None
        ):
            raise KerrReturningRadiationReceiverRayError(
                "fine/coarse past-worldtube schema disagrees"
            )
        worldtube_radius = abs(
            fine.past_worldtube_radius_m - coarse.past_worldtube_radius_m
        )
        if worldtube_radius > _TERMINAL_EVENT_DIFFERENCE_OVER_MASS * scale:
            raise KerrReturningRadiationReceiverRayError(
                "independent past-worldtube terminals disagree"
            )
    return KerrReturningRadiationReceiverRayConvergence(
        raw_outcome_agrees=raw_outcome,
        terminal_target_agrees=target,
        receiver_outcome_agrees=receiver_outcome,
        complete_topology_agrees=topology,
        terminal_event_difference_m=terminal_event,
        terminal_covector_relative_difference=terminal_covector,
        maximum_crossing_event_difference_m=maximum_crossing_event,
        maximum_crossing_covector_relative_difference=maximum_crossing_covector,
        source_radius_difference_over_mass=source_radius,
        source_emission_cosine_difference=source_cosine,
        source_frequency_relative_difference=source_frequency,
        frequency_ratio_relative_difference=frequency_ratio,
        g4_relative_difference=g4,
        receiver_integrand_relative_difference=integrand,
        past_worldtube_radius_difference_m=worldtube_radius,
        converged=True,
    )


@dataclass(frozen=True, slots=True, init=False)
class KerrReturningRadiationReceiverRayPrimitive:
    """One fine result admitted only after whole-ray convergence and replay."""

    receiver_frame: KerrFiniteThicknessSurfaceFrame
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    receiver_incidence_cosine: float
    receiver_tangent_azimuth_rad: float
    receiver_future_state: HamiltonianState
    receiver_past_state: HamiltonianState
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions
    coarse_surface_options: SurfaceEventOptions
    ray: RayTraceResult
    coarse_ray: RayTraceResult
    convergence: KerrReturningRadiationReceiverRayConvergence
    outcome: ReceiverRayOutcome
    past_worldtube_target_id: str | None
    past_worldtube_radius_m: float | None
    source_face: PhotosphereFace | None
    source: KerrFiniteThicknessFaceEmitter | None
    source_surface_id: str | None
    source_radius_over_mass: float | None
    source_emission_cosine: float | None
    source_local_frequency: float | None
    source_to_receiver_frequency_ratio: float | None
    bolometric_g4_factor: float | None
    d20_angular_multiplier: float | None
    receiver_directional_integrand: float
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrReturningRadiationReceiverRayPrimitive is built only by its tracer"
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_receiver_direction(self)


def _build_result(
    *,
    frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    mu_i: float,
    psi_i: float,
    future_state: HamiltonianState,
    past_state: HamiltonianState,
    fine_ray_options: RayTraceOptions,
    fine_surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions,
    coarse_surface_options: SurfaceEventOptions,
    fine: _EvaluatedReceiverRay,
    coarse: _EvaluatedReceiverRay,
    convergence: KerrReturningRadiationReceiverRayConvergence,
) -> KerrReturningRadiationReceiverRayPrimitive:
    descriptor = {
        "implementationId": IMPLEMENTATION_ID,
        "receiver": {
            "frameDescriptorSha256": frame.model_descriptor_sha256,
            "incidenceCosine": mu_i,
            "tangentAzimuthRad": psi_i,
            "localFrequency": 1.0,
            "futureCovectorKs": future_state.covector,
            "pastCovectorKs": past_state.covector,
        },
        "initialContact": {
            "surfaceId": _face_surface_id(frame.emitter.face),
            "declaredSide": 1,
            "epsilonEventDisplacement": False,
            "fine": asdict(fine.ray.multi_surface_trace.initial_contact),
            "coarse": asdict(coarse.ray.multi_surface_trace.initial_contact),
        },
        "metric": asdict(surface.metric),
        "calibration": asdict(surface.calibration),
        "termination": asdict(termination),
        "fineRayOptions": asdict(fine_ray_options),
        "fineSurfaceOptions": asdict(fine_surface_options),
        "coarseRayOptions": asdict(coarse_ray_options),
        "coarseSurfaceOptions": asdict(coarse_surface_options),
        "outcome": fine.outcome,
        "source": {
            "face": fine.source_face,
            "surfaceId": fine.source_surface_id,
            "radiusOverMass": fine.source_radius_over_mass,
            "emissionCosine": fine.source_emission_cosine,
            "localFrequency": fine.source_local_frequency,
        },
        "transport": {
            "frequencyRatioNuIOverNuE": fine.frequency_ratio,
            "g4": fine.g4,
            "d20AngularMultiplier": fine.d20_multiplier,
            "receiverDirectionalIntegrand": fine.receiver_integrand,
            "includesSolidAngleIntegration": False,
            "includesAreaJacobian": False,
            "isKernelK": False,
        },
        "pastWorldtube": {
            "semanticOutcome": (
                PAST_WORLDTUBE_NO_SOURCE
                if fine.past_worldtube_target_id is not None
                else None
            ),
            "rawTraceOutcome": (
                fine.ray.outcome
                if fine.past_worldtube_target_id is not None
                else None
            ),
            "targetId": fine.past_worldtube_target_id,
            "radiusM": fine.past_worldtube_radius_m,
        },
        "wholeRayConvergence": asdict(convergence),
        "scientificBoundary": dict(SCIENTIFIC_STATUS),
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrReturningRadiationReceiverRayPrimitive)
    values = (
        ("receiver_frame", frame),
        ("surface", surface),
        ("termination", termination),
        ("receiver_incidence_cosine", mu_i),
        ("receiver_tangent_azimuth_rad", psi_i),
        ("receiver_future_state", future_state),
        ("receiver_past_state", past_state),
        ("ray_options", fine_ray_options),
        ("surface_options", fine_surface_options),
        ("coarse_ray_options", coarse_ray_options),
        ("coarse_surface_options", coarse_surface_options),
        ("ray", fine.ray),
        ("coarse_ray", coarse.ray),
        ("convergence", convergence),
        ("outcome", fine.outcome),
        ("past_worldtube_target_id", fine.past_worldtube_target_id),
        ("past_worldtube_radius_m", fine.past_worldtube_radius_m),
        ("source_face", fine.source_face),
        ("source", fine.source),
        ("source_surface_id", fine.source_surface_id),
        ("source_radius_over_mass", fine.source_radius_over_mass),
        ("source_emission_cosine", fine.source_emission_cosine),
        ("source_local_frequency", fine.source_local_frequency),
        ("source_to_receiver_frequency_ratio", fine.frequency_ratio),
        ("bolometric_g4_factor", fine.g4),
        ("d20_angular_multiplier", fine.d20_multiplier),
        ("receiver_directional_integrand", fine.receiver_integrand),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    )
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def trace_kerr_returning_radiation_receiver_direction(
    receiver_frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
    receiver_incidence_cosine: float,
    receiver_tangent_azimuth_rad: float,
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
) -> KerrReturningRadiationReceiverRayPrimitive:
    """Trace one receiver sky direction with independent fine/coarse rays."""

    surface = _validated_surface(surface)
    frame = _validated_receiver_frame(receiver_frame, surface)
    psi_i, future_state, past_state, _residual = _build_receiver_states(
        frame,
        receiver_incidence_cosine,
        receiver_tangent_azimuth_rad,
    )
    mu_i = receiver_incidence_cosine
    termination = _validated_termination(termination, surface, past_state)
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
    fine_ray = _trace_one_resolution(
        frame,
        past_state,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
    )
    coarse_ray = _trace_one_resolution(
        frame,
        past_state,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
    )
    fine = _evaluate_ray(
        fine_ray,
        mu_i,
        surface,
        termination,
        fine_ray_options,
        fine_surface_options,
    )
    coarse = _evaluate_ray(
        coarse_ray,
        mu_i,
        surface,
        termination,
        coarse_ray_options,
        coarse_surface_options,
    )
    convergence = _compare_rays(fine, coarse, surface.metric)
    return _build_result(
        frame=frame,
        surface=surface,
        termination=termination,
        mu_i=mu_i,
        psi_i=psi_i,
        future_state=future_state,
        past_state=past_state,
        fine_ray_options=fine_ray_options,
        fine_surface_options=fine_surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        fine=fine,
        coarse=coarse,
        convergence=convergence,
    )


def verify_kerr_returning_radiation_receiver_direction(
    result: KerrReturningRadiationReceiverRayPrimitive,
) -> None:
    """Replay both whole rays and every derived source transport quantity."""

    if type(result) is not KerrReturningRadiationReceiverRayPrimitive:
        raise TypeError(
            "result must be the exact receiver-ray primitive"
        )
    try:
        if (
            type(result._descriptor_json) is not str
            or type(result._descriptor_sha256) is not str
        ):
            raise TypeError("descriptor fields must be exact strings")
        canonical = _canonical_json(json.loads(result._descriptor_json))
        live_sha = hashlib.sha256(
            result._descriptor_json.encode("utf-8")
        ).hexdigest()
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise KerrReturningRadiationReceiverRayError(
            "receiver-ray result descriptor is malformed"
        ) from error
    if (
        canonical.encode("utf-8") != result._descriptor_json.encode("utf-8")
        or live_sha.encode("ascii") != result._descriptor_sha256.encode("ascii")
    ):
        raise KerrReturningRadiationReceiverRayError(
            "receiver-ray descriptor or SHA-256 identity is stale"
        )
    expected = trace_kerr_returning_radiation_receiver_direction(
        result.receiver_frame,
        result.surface,
        result.receiver_incidence_cosine,
        result.receiver_tangent_azimuth_rad,
        termination=result.termination,
        ray_options=result.ray_options,
        surface_options=result.surface_options,
        coarse_ray_options=result.coarse_ray_options,
        coarse_surface_options=result.coarse_surface_options,
    )
    try:
        _require_trusted_exact_tree(result, expected, "result")
    except KerrReturningRadiationReceiverRayError as error:
        raise KerrReturningRadiationReceiverRayError(
            "receiver-ray live fields disagree with mandatory replay: "
            f"{error}"
        ) from error


__all__ = (
    "IMPLEMENTATION_ID",
    "KerrReturningRadiationReceiverRayConvergence",
    "KerrReturningRadiationReceiverRayError",
    "KerrReturningRadiationReceiverRayPrimitive",
    "PAST_WORLDTUBE_NO_SOURCE",
    "ReceiverRayOutcome",
    "SCIENTIFIC_STATUS",
    "trace_kerr_returning_radiation_receiver_direction",
    "verify_kerr_returning_radiation_receiver_direction",
)
