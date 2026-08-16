"""Certified local emission launch frame on a finite Kerr photosphere.

The stationary finite-height emitter already owns the matter four-velocity
and the outward unit face normal.  Returning-radiation and illumination
calculations additionally need two authenticated tangent directions in the
emitter rest space.  This module completes that orthonormal tetrad without
copying the Kerr metric or photosphere equations.

``meridional_tangent`` is the normalized projection of increasing
Boyer--Lindquist radius into the selected face.  ``azimuthal_tangent`` is the
normalized projection of increasing Boyer--Lindquist azimuth, made orthogonal
to the matter velocity, face normal, and meridional tangent.  A future null
launch with local frequency ``nu`` is then

``k = nu * (u + mu*n + sqrt(1-mu^2)*(cos(psi)*e_rho + sin(psi)*e_phi))``.

This is a local kinematic construction only.  It does not trace a ray, solve
returning radiation, prescribe an atmosphere, or upgrade the Zhou stationary
surface to hydrostatic structure or GRMHD.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

from offline.geodesic import (
    HamiltonianState,
    hamiltonian_null_residual,
)
from offline.kerr import kerr_bl_vector_to_ks_cartesian
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.spacetime import Vector4, bilinear, matrix_vector


IMPLEMENTATION_ID: Final = "zhou-finite-thickness-local-emission-launch/v1"

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "certified local orthonormal emission-launch frame on the "
            "stationary Zhou finite-height Kerr photosphere"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "launchConvention": (
            "future null k=nu[u+mu n+sqrt(1-mu^2)(cos psi e_rho+sin psi e_phi)]"
        ),
        "includesGeodesicTracing": False,
        "includesReturningRadiationKernel": False,
        "includesSolvedAtmosphere": False,
        "isHydrostaticVerticalStructureSolution": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this local tetrad and launch constructor as a "
            "returning-radiation solution, atmosphere, hydrostatic disk, "
            "ray tracer, or GRMHD model."
        ),
    }
)

_GRAM_TOLERANCE: Final = 8.0e-10
_NULL_RESIDUAL_LIMIT: Final = 2.0e-10


class KerrFiniteThicknessLaunchError(RuntimeError):
    """Raised when a nominal local emission launch is not physical."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validated_emitter(
    emitter: KerrFiniteThicknessFaceEmitter,
) -> KerrFiniteThicknessFaceEmitter:
    """Rebuild every emitter-derived field from its authenticated inputs."""

    if type(emitter) is not KerrFiniteThicknessFaceEmitter:
        raise TypeError(
            "emitter must be the exact KerrFiniteThicknessFaceEmitter"
        )
    expected = KerrFiniteThicknessFaceEmitter(
        metric=emitter.metric,
        calibration=emitter.calibration,
        pseudo_cylindrical_radius_over_mass=(
            emitter.pseudo_cylindrical_radius_over_mass
        ),
        face=emitter.face,
        phi_ks_rad=emitter.phi_ks_rad,
        coordinate_time_m=emitter.coordinate_time_m,
    )
    if emitter != expected:
        raise KerrFiniteThicknessLaunchError(
            "finite-thickness emitter live fields or provenance are stale"
        )
    return expected


def _linear_combination(
    terms: tuple[tuple[float, Vector4], ...],
) -> Vector4:
    result = tuple(
        math.fsum(scale * vector[index] for scale, vector in terms)
        for index in range(4)
    )
    if not all(math.isfinite(value) for value in result):
        raise KerrFiniteThicknessLaunchError("local-frame vector is not finite")
    return result  # type: ignore[return-value]


def _normalize_spacelike(
    vector: Vector4,
    metric_covariant: tuple[tuple[float, ...], ...],
    label: str,
) -> Vector4:
    norm = bilinear(vector, metric_covariant, vector)
    if not math.isfinite(norm) or norm <= 0.0:
        raise KerrFiniteThicknessLaunchError(
            f"{label} is not a finite spacelike direction"
        )
    inverse_norm = 1.0 / math.sqrt(norm)
    result = tuple(inverse_norm * value for value in vector)
    if not all(math.isfinite(value) for value in result):
        raise KerrFiniteThicknessLaunchError(f"{label} normalization overflowed")
    return result  # type: ignore[return-value]


def _project_spatial_tangent(
    raw: Vector4,
    emitter: KerrFiniteThicknessFaceEmitter,
    metric_covariant: tuple[tuple[float, ...], ...],
    prior: tuple[Vector4, ...],
) -> Vector4:
    velocity_projection = bilinear(
        raw,
        metric_covariant,
        emitter.four_velocity,
    )
    normal_projection = bilinear(
        raw,
        metric_covariant,
        emitter.outward_unit_normal,
    )
    terms: list[tuple[float, Vector4]] = [
        (1.0, raw),
        (velocity_projection, emitter.four_velocity),
        (-normal_projection, emitter.outward_unit_normal),
    ]
    for direction in prior:
        projection = bilinear(raw, metric_covariant, direction)
        terms.append((-projection, direction))
    return _linear_combination(tuple(terms))


def _frame_gram_error(
    emitter: KerrFiniteThicknessFaceEmitter,
    meridional: Vector4,
    azimuthal: Vector4,
) -> float:
    sample = emitter.metric.sample(emitter.event)
    basis = (
        emitter.four_velocity,
        meridional,
        azimuthal,
        emitter.outward_unit_normal,
    )
    maximum = 0.0
    for first in range(4):
        for second in range(4):
            expected = -1.0 if first == second == 0 else float(first == second)
            error = abs(
                bilinear(
                    basis[first],
                    sample.covariant,
                    basis[second],
                )
                - expected
            )
            maximum = max(maximum, error)
    if not math.isfinite(maximum):
        raise KerrFiniteThicknessLaunchError("local-frame Gram error is non-finite")
    return maximum


@dataclass(frozen=True, slots=True, init=False)
class KerrFiniteThicknessSurfaceFrame:
    """Authenticated emitter tetrad with two face-tangent directions."""

    emitter: KerrFiniteThicknessFaceEmitter
    meridional_tangent: Vector4
    azimuthal_tangent: Vector4
    maximum_gram_error: float
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self, emitter: KerrFiniteThicknessFaceEmitter) -> None:
        emitter = _validated_emitter(emitter)
        metric = emitter.metric
        point = emitter.photosphere_point
        radius_m = point.radius_over_mass * metric.mass_m
        sample = metric.sample(emitter.event)

        raw_meridional = kerr_bl_vector_to_ks_cartesian(
            (0.0, 1.0, 0.0, 0.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=emitter.phi_ks_rad,
        )
        meridional = _normalize_spacelike(
            _project_spatial_tangent(
                raw_meridional,
                emitter,
                sample.covariant,
                (),
            ),
            sample.covariant,
            "meridional tangent",
        )

        raw_azimuthal = kerr_bl_vector_to_ks_cartesian(
            (0.0, 0.0, 0.0, 1.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=emitter.phi_ks_rad,
        )
        azimuthal = _normalize_spacelike(
            _project_spatial_tangent(
                raw_azimuthal,
                emitter,
                sample.covariant,
                (meridional,),
            ),
            sample.covariant,
            "azimuthal tangent",
        )
        gram_error = _frame_gram_error(emitter, meridional, azimuthal)
        if gram_error > _GRAM_TOLERANCE:
            raise KerrFiniteThicknessLaunchError(
                "finite-thickness local frame is not orthonormal"
            )
        # Preserve the declared positive-r and positive-phi orientations.
        if (
            bilinear(meridional, sample.covariant, raw_meridional) <= 0.0
            or bilinear(azimuthal, sample.covariant, raw_azimuthal) <= 0.0
        ):
            raise KerrFiniteThicknessLaunchError(
                "finite-thickness tangent orientation is ambiguous"
            )

        descriptor = {
            "azimuthalTangentKs": azimuthal,
            "emitterDescriptorSha256": emitter.model_descriptor_sha256,
            "implementationId": IMPLEMENTATION_ID,
            "maximumGramError": gram_error,
            "meridionalTangentKs": meridional,
            "outwardNormalKs": emitter.outward_unit_normal,
            "tangentOrientation": {
                "azimuthal": "projected increasing Boyer-Lindquist phi",
                "meridional": "projected increasing Boyer-Lindquist radius",
            },
        }
        descriptor_json = _canonical_json(descriptor)
        object.__setattr__(self, "emitter", emitter)
        object.__setattr__(self, "meridional_tangent", meridional)
        object.__setattr__(self, "azimuthal_tangent", azimuthal)
        object.__setattr__(self, "maximum_gram_error", gram_error)
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)


@dataclass(frozen=True, slots=True, init=False)
class KerrFiniteThicknessEmissionLaunch:
    """One self-recomputed future null launch and its reversed past state."""

    frame: KerrFiniteThicknessSurfaceFrame
    emission_angle_cosine: float
    tangent_azimuth_rad: float
    local_frequency: float
    future_state: HamiltonianState
    reversed_past_state: HamiltonianState
    null_residual: float
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(
        self,
        frame: KerrFiniteThicknessSurfaceFrame,
        emission_angle_cosine: float,
        tangent_azimuth_rad: float,
        local_frequency: float = 1.0,
    ) -> None:
        if type(frame) is not KerrFiniteThicknessSurfaceFrame:
            raise TypeError("frame must be the exact finite-thickness surface frame")
        # A frozen dataclass is not a cryptographic trust boundary: low-level
        # callers can still use ``object.__setattr__``.  Reconstruct the whole
        # frame from its sole scientific input before any launch calculation,
        # then require both the live axes and stored descriptor to agree.
        expected_frame = KerrFiniteThicknessSurfaceFrame(frame.emitter)
        if frame != expected_frame:
            raise KerrFiniteThicknessLaunchError(
                "finite-thickness surface frame live fields or provenance are stale"
            )
        frame = expected_frame
        mu = _finite_number(emission_angle_cosine, "emission_angle_cosine")
        if mu <= 0.0 or mu > 1.0:
            raise ValueError("emission_angle_cosine must lie in (0, 1]")
        raw_azimuth = _finite_number(tangent_azimuth_rad, "tangent_azimuth_rad")
        azimuth = raw_azimuth % (2.0 * math.pi)
        frequency = _finite_number(local_frequency, "local_frequency")
        if frequency <= 0.0:
            raise ValueError("local_frequency must be positive")

        tangent_weight = math.sqrt(max(0.0, 1.0 - mu * mu))
        spatial = _linear_combination(
            (
                (mu, frame.emitter.outward_unit_normal),
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
        future_vector = _linear_combination(
            (
                (frequency, frame.emitter.four_velocity),
                (frequency, spatial),
            )
        )
        sample = frame.emitter.metric.sample(frame.emitter.event)
        future_covector = matrix_vector(sample.covariant, future_vector)
        future_state = HamiltonianState(frame.emitter.event, future_covector)
        reversed_state = HamiltonianState(
            frame.emitter.event,
            tuple(-value for value in future_covector),
        )
        residual = hamiltonian_null_residual(
            frame.emitter.metric,
            future_state,
        )
        measured_frequency = -math.fsum(
            frame.emitter.four_velocity[index] * future_covector[index]
            for index in range(4)
        )
        measured_mu = math.fsum(
            frame.emitter.outward_unit_normal[index] * future_covector[index]
            for index in range(4)
        ) / measured_frequency
        if (
            not math.isfinite(residual)
            or residual > _NULL_RESIDUAL_LIMIT
            or not math.isclose(
                measured_frequency,
                frequency,
                rel_tol=5.0e-10,
                abs_tol=0.0,
            )
            or not math.isclose(
                measured_mu,
                mu,
                rel_tol=5.0e-10,
                abs_tol=5.0e-12,
            )
        ):
            raise KerrFiniteThicknessLaunchError(
                "local future photon failed its null/frequency/direction audit"
            )
        reverse_projection = frame.emitter.project_past_directed_photon(
            reversed_state,
            null_residual_limit=_NULL_RESIDUAL_LIMIT,
            backside_policy="reject",
        )
        if (
            reverse_projection.face_classification != "outgoing"
            or not math.isclose(
                reverse_projection.outgoing_cosine,
                mu,
                rel_tol=5.0e-10,
                abs_tol=5.0e-12,
            )
            or not math.isclose(
                reverse_projection.local_frequency,
                frequency,
                rel_tol=5.0e-10,
                abs_tol=0.0,
            )
        ):
            raise KerrFiniteThicknessLaunchError(
                "reversed past photon disagrees with the launch frame"
            )

        descriptor = {
            "emissionAngleCosine": mu,
            "frameDescriptorSha256": frame.model_descriptor_sha256,
            "futureCovectorKs": future_covector,
            "implementationId": IMPLEMENTATION_ID,
            "localFrequency": frequency,
            "tangentAzimuthRad": azimuth,
        }
        descriptor_json = _canonical_json(descriptor)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "emission_angle_cosine", mu)
        object.__setattr__(self, "tangent_azimuth_rad", azimuth)
        object.__setattr__(self, "local_frequency", frequency)
        object.__setattr__(self, "future_state", future_state)
        object.__setattr__(self, "reversed_past_state", reversed_state)
        object.__setattr__(self, "null_residual", residual)
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)


__all__ = (
    "IMPLEMENTATION_ID",
    "KerrFiniteThicknessEmissionLaunch",
    "KerrFiniteThicknessLaunchError",
    "KerrFiniteThicknessSurfaceFrame",
    "SCIENTIFIC_STATUS",
)
