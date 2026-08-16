"""Certified local matter frame on a stationary finite-thickness Kerr face.

This module implements the *velocity prescription* used by Zhou et al. (2020,
``arXiv:2004.12589v2``): material on a photosphere point with
pseudo-cylindrical radius ``rho`` is assigned the equatorial circular angular
velocity ``Omega(rho)``.  The time component is **not** copied from the
equatorial orbit.  It is normalized with the exact Kerr metric at the actual
upper/lower photosphere event,

``u = u^t (partial_t + Omega(rho) partial_phi)`` and ``g(u, u) = -1``.

The fixed photosphere itself is owned by :mod:`offline.kerr_finite_thickness`;
the exact metric and coordinate Jacobian are owned by :mod:`offline.kerr`;
the equatorial orbit scalar is owned by :mod:`offline.novikov_thorne`.  This
module deliberately does not copy any of those formulas and is not connected
to the production v2 thin-disk sampler.

The resulting matter frame is a stationary reference prescription on an
assumed surface.  It is not an off-equatorial geodesic, hydrostatic vertical
structure, GRMHD, returning-radiation transport, or an atmosphere model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

from offline.geodesic import HamiltonianState, hamiltonian_null_residual
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_vector_to_ks_cartesian,
    kerr_oblate_event_to_ks_cartesian,
)
from offline.kerr_finite_thickness import (
    MODEL_IMPLEMENTATION_ID as SURFACE_IMPLEMENTATION_ID,
    BoyerLindquistPhotospherePoint,
    PhotosphereFace,
    StationaryKerrFiniteThicknessCalibration,
    VALID_FACES,
)
from offline.novikov_thorne import circular_orbit_scalars
from offline.spacetime import Vector4, bilinear, matrix_vector


IMPLEMENTATION_ID: Final = "zhou-2020-stationary-kerr-face-emitter/v1"
PRIMARY_SOURCE_URL: Final = "https://arxiv.org/abs/2004.12589v2"

_SOURCE_REFERENCE_IDENTITY: Final[dict[str, Any]] = {
    "arxivId": "2004.12589",
    "citation": "Zhou et al. (2020)",
    "role": (
        "fixed-photosphere circular-velocity prescription: use the "
        "equatorial circular Omega at the same pseudo-cylindrical rho and "
        "normalize u^t at the actual photosphere point"
    ),
    "url": PRIMARY_SOURCE_URL,
    "version": "v2",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


PRIMARY_SOURCE_REFERENCE: Final[Mapping[str, Any]] = MappingProxyType(
    _SOURCE_REFERENCE_IDENTITY
)
PRIMARY_SOURCE_REFERENCE_SHA256: Final = hashlib.sha256(
    _canonical_json(_SOURCE_REFERENCE_IDENTITY).encode("utf-8")
).hexdigest()
PRIMARY_SOURCE_HASH_SEMANTICS: Final = (
    "SHA-256 of canonical UTF-8 JSON source-reference identity metadata; "
    "not a hash of arXiv HTML, TeX, or PDF bytes"
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "Zhou fixed-photosphere stationary circular-velocity matter reference"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "primarySource": PRIMARY_SOURCE_URL,
        "velocityPrescription": (
            "equatorial circular Omega at the same pseudo-cylindrical rho; "
            "u^t normalized by the exact Kerr metric at the selected face event"
        ),
        "includesCertifiedLocalPhotonProjection": True,
        "isOffEquatorialGeodesic": False,
        "isHydrostaticVerticalStructureSolution": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "includesEmissionSpectrum": False,
        "prohibitedClaim": (
            "Do not describe this fixed-surface circular-velocity reference as "
            "an off-equatorial geodesic, hydrostatic disk, GRMHD, returning "
            "radiation, or a solved atmosphere."
        ),
    }
)

PhotonFaceClassification = Literal["outgoing", "tangent", "backside"]
BacksidePolicy = Literal["reject", "classify"]

_FOUR_VELOCITY_TOLERANCE: Final = 4.0e-10
_NORMAL_TOLERANCE: Final = 4.0e-10
_ORTHOGONALITY_TOLERANCE: Final = 6.0e-10
_DIRECTION_COSINE_TOLERANCE: Final = 8.0e-10
_DEFAULT_NULL_RESIDUAL_LIMIT: Final = 1.0e-8
_DEFAULT_EVENT_TOLERANCE_OVER_MASS: Final = 1.0e-10


class KerrFiniteThicknessEmitterError(RuntimeError):
    """Raised when a nominal finite-thickness matter frame is not physical."""


class BacksidePhotonError(KerrFiniteThicknessEmitterError):
    """Raised when a caller requires an outgoing ray but ``mu < 0``."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance * max(
        1.0,
        abs(actual),
        abs(expected),
    )


def _linear_combination(
    first_scale: float,
    first: Vector4,
    second_scale: float,
    second: Vector4,
) -> Vector4:
    result = tuple(
        math.fsum((first_scale * first[index], second_scale * second[index]))
        for index in range(4)
    )
    if not all(math.isfinite(value) for value in result):
        raise KerrFiniteThicknessEmitterError(
            "finite-thickness frame vector is not finite"
        )
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessPhotonProjection:
    """Local, signed projection of one stored past-directed null covector.

    ``local_frequency`` retains the input covector's positive affine scale.
    ``outgoing_cosine`` is scale invariant and deliberately signed.  No
    absolute value is applied: negative values are backside incidence.
    """

    local_frequency: float
    normalized_local_frequency: float
    affine_covector_scale: float
    outgoing_cosine: float
    face_classification: PhotonFaceClassification
    null_residual: float

    def __post_init__(self) -> None:
        positive = (
            self.local_frequency,
            self.normalized_local_frequency,
            self.affine_covector_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("photon projection frequencies and scale must be positive")
        if (
            not math.isfinite(self.outgoing_cosine)
            or self.outgoing_cosine < -1.0
            or self.outgoing_cosine > 1.0
        ):
            raise ValueError("outgoing cosine must lie in [-1, 1]")
        if self.face_classification not in ("outgoing", "tangent", "backside"):
            raise ValueError("invalid photon face classification")
        if not math.isfinite(self.null_residual) or self.null_residual < 0.0:
            raise ValueError("null residual must be finite and non-negative")


@dataclass(frozen=True, slots=True, init=False)
class KerrFiniteThicknessFaceEmitter:
    """Self-validated future-timelike matter frame on one photosphere face.

    Construction accepts only the owning metric, calibration, radius, face,
    and coordinates.  Every derived field is recomputed internally, so a
    caller cannot supply an unauthenticated event, four-velocity, or normal.
    """

    metric: KerrKerrSchildMetric
    calibration: StationaryKerrFiniteThicknessCalibration
    pseudo_cylindrical_radius_over_mass: float
    face: PhotosphereFace
    phi_ks_rad: float
    coordinate_time_m: float
    photosphere_point: BoyerLindquistPhotospherePoint
    event: Vector4
    four_velocity: Vector4
    outward_unit_normal: Vector4
    outward_unit_normal_covector: Vector4
    angular_velocity_inverse_m: float
    four_velocity_bl_time_component: float
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(
        self,
        *,
        metric: KerrKerrSchildMetric,
        calibration: StationaryKerrFiniteThicknessCalibration,
        pseudo_cylindrical_radius_over_mass: float,
        face: PhotosphereFace,
        phi_ks_rad: float = 0.0,
        coordinate_time_m: float = 0.0,
    ) -> None:
        # Exact types close the implementation boundary: a subclass may
        # override either the metric or calibration methods while retaining a
        # misleading inherited identity.
        if type(metric) is not KerrKerrSchildMetric:
            raise TypeError("metric must be the exact built-in KerrKerrSchildMetric")
        if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
            raise TypeError(
                "calibration must be the exact built-in "
                "StationaryKerrFiniteThicknessCalibration"
            )
        if not isinstance(face, str) or face not in VALID_FACES:
            raise ValueError("face must be 'upper' or 'lower'")
        rho = _finite_number(
            pseudo_cylindrical_radius_over_mass,
            "pseudo_cylindrical_radius_over_mass",
        )
        phi = _finite_number(phi_ks_rad, "phi_ks_rad")
        coordinate_time = _finite_number(coordinate_time_m, "coordinate_time_m")

        metric_spin_magnitude = abs(metric.dimensionless_spin)
        if not math.isclose(
            metric_spin_magnitude,
            calibration.dimensionless_spin,
            rel_tol=64.0 * math.ulp(1.0),
            abs_tol=64.0 * math.ulp(1.0),
        ):
            raise ValueError(
                "metric spin magnitude and finite-thickness calibration spin disagree"
            )
        if calibration.eddington_scaled_mass_accretion_rate <= 0.0:
            raise ValueError(
                "positive finite thickness is required; dotm=0 has no distinct faces"
            )
        # The two faces join at the zero-height ISCO.  Treating that seam as a
        # uniquely authenticated upper or lower face would be false.
        if rho <= calibration.isco_radius_over_mass:
            raise ValueError(
                "face emitter radius must lie strictly outside the ISCO seam"
            )
        if rho > calibration.outer_radius_over_mass:
            raise ValueError("face emitter radius lies outside the calibration annulus")

        point = calibration.photosphere_point(rho, face)
        if point.signed_height_over_mass == 0.0:
            raise ValueError("selected photosphere face has numerically zero height")
        radius_m = point.radius_over_mass * metric.mass_m
        rho_m = rho * metric.mass_m
        height_m = point.signed_height_over_mass * metric.mass_m
        if not all(math.isfinite(value) for value in (radius_m, rho_m, height_m)):
            raise ValueError("dimensionful photosphere coordinates overflowed")
        if radius_m <= metric.singularity_guard_m or rho_m <= metric.singularity_guard_m:
            raise ValueError("photosphere event is inside the metric scale/axis guard")
        if abs(math.sin(point.theta_rad)) <= 128.0 * math.ulp(1.0):
            raise ValueError("photosphere event is too close to the spin axis")

        event = kerr_oblate_event_to_ks_cartesian(
            coordinate_time_m=coordinate_time,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
            spin_a_m=metric.spin_a_m,
        )
        sample = metric.sample(event)

        orbit = circular_orbit_scalars(
            rho,
            calibration.dimensionless_spin,
            calibration.orientation,
        )
        spin_axis_sign = -1.0 if metric.spin_a_m < 0.0 else 1.0
        angular_velocity = spin_axis_sign * orbit.omega_m / metric.mass_m
        if not math.isfinite(angular_velocity) or angular_velocity == 0.0:
            raise KerrFiniteThicknessEmitterError(
                "finite-thickness angular velocity is not finite and non-zero"
            )

        # Zhou's off-equatorial reference uses Omega(rho), while normalization
        # belongs to the actual selected photosphere point.
        helical_vector = kerr_bl_vector_to_ks_cartesian(
            (1.0, 0.0, 0.0, angular_velocity),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
        )
        helical_norm = bilinear(helical_vector, sample.covariant, helical_vector)
        if not math.isfinite(helical_norm) or helical_norm >= 0.0:
            raise KerrFiniteThicknessEmitterError(
                "Zhou circular-velocity reference is not timelike at this face event"
            )
        u_t = 1.0 / math.sqrt(-helical_norm)
        four_velocity = tuple(u_t * value for value in helical_vector)
        if not all(math.isfinite(value) for value in four_velocity):
            raise KerrFiniteThicknessEmitterError("four-velocity is not finite")
        velocity_norm = bilinear(
            four_velocity,  # type: ignore[arg-type]
            sample.covariant,
            four_velocity,  # type: ignore[arg-type]
        )
        if (
            four_velocity[0] <= 0.0
            or not _close(velocity_norm, -1.0, _FOUR_VELOCITY_TOLERANCE)
        ):
            raise KerrFiniteThicknessEmitterError(
                "face matter four-velocity is not future unit-timelike"
            )

        # Reuse the calibration's unit BL covector.  To metric-dual it without
        # copying Kerr inverse-metric formulas, solve in the exact public
        # transformed (partial_r, partial_theta/M) basis.  Both basis vectors
        # and both covector components are dimensionless, which also avoids a
        # spurious M^2 condition number at extreme coordinate scales.
        normal_covector_bl = calibration.unit_face_normal_covector_bl(rho, face)
        inverse_mass = 1.0 / metric.mass_m
        if not math.isfinite(inverse_mass):
            raise ValueError("metric mass scale is too small for the certified basis")
        radial_basis = kerr_bl_vector_to_ks_cartesian(
            (0.0, 1.0, 0.0, 0.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
        )
        polar_basis_over_mass = kerr_bl_vector_to_ks_cartesian(
            (0.0, 0.0, inverse_mass, 0.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
        )
        g_rr = bilinear(radial_basis, sample.covariant, radial_basis)
        g_rtheta = bilinear(
            radial_basis,
            sample.covariant,
            polar_basis_over_mass,
        )
        g_thetatheta = bilinear(
            polar_basis_over_mass,
            sample.covariant,
            polar_basis_over_mass,
        )
        determinant = math.fsum((g_rr * g_thetatheta, -g_rtheta * g_rtheta))
        determinant_scale = max(
            abs(g_rr * g_thetatheta),
            abs(g_rtheta * g_rtheta),
            1.0e-300,
        )
        if (
            not all(
                math.isfinite(value)
                for value in (g_rr, g_rtheta, g_thetatheta, determinant)
            )
            or g_rr <= 0.0
            or g_thetatheta <= 0.0
            or determinant <= 512.0 * math.ulp(1.0) * determinant_scale
        ):
            raise KerrFiniteThicknessEmitterError(
                "photosphere meridional normal basis is ill-conditioned"
            )
        radial_covector = normal_covector_bl[1]
        polar_covector = normal_covector_bl[2]
        normal_radial_component = math.fsum(
            (
                radial_covector * g_thetatheta,
                -polar_covector * g_rtheta,
            )
        ) / determinant
        normal_polar_component_scaled = math.fsum(
            (
                polar_covector * g_rr,
                -radial_covector * g_rtheta,
            )
        ) / determinant
        outward_normal = _linear_combination(
            normal_radial_component,
            radial_basis,
            normal_polar_component_scaled,
            polar_basis_over_mass,
        )
        normal_norm = bilinear(outward_normal, sample.covariant, outward_normal)
        orthogonality = bilinear(
            four_velocity,  # type: ignore[arg-type]
            sample.covariant,
            outward_normal,
        )
        outward_orientation = math.fsum(
            (
                radial_covector * normal_radial_component,
                polar_covector * normal_polar_component_scaled,
            )
        )
        if (
            not _close(normal_norm, 1.0, _NORMAL_TOLERANCE)
            or abs(orthogonality) > _ORTHOGONALITY_TOLERANCE
            or not math.isfinite(outward_orientation)
            or outward_orientation <= 0.0
        ):
            raise KerrFiniteThicknessEmitterError(
                "photosphere normal is not outward unit-spacelike and orthogonal"
            )
        outward_normal_covector = matrix_vector(sample.covariant, outward_normal)
        if not all(math.isfinite(value) for value in outward_normal_covector):
            raise KerrFiniteThicknessEmitterError("lowered face normal is not finite")

        descriptor: dict[str, Any] = {
            "calibration": {
                "dimensionlessSpinMagnitude": calibration.dimensionless_spin,
                "eddingtonScaledMassAccretionRate": (
                    calibration.eddington_scaled_mass_accretion_rate
                ),
                "implementationId": SURFACE_IMPLEMENTATION_ID,
                "orientation": calibration.orientation,
                "outerRadiusOverMass": calibration.outer_radius_over_mass,
                "thinnessGateMaximumHOverRho": (
                    calibration.thinness_gate_maximum_h_over_rho
                ),
            },
            "capabilities": {
                "includesAtmosphere": False,
                "includesGRMHD": False,
                "includesHydrostaticStructure": False,
                "includesReturningRadiation": False,
                "isOffEquatorialGeodesic": False,
            },
            "certifiedFrame": {
                "eventKs": list(event),
                "fourVelocityBlTimeComponent": u_t,
                "fourVelocityKs": list(four_velocity),
                "outwardUnitNormalCovectorKs": list(outward_normal_covector),
                "outwardUnitNormalKs": list(outward_normal),
            },
            "implementationId": IMPLEMENTATION_ID,
            "metric": {
                "massM": metric.mass_m,
                "singularityGuardM": metric.singularity_guard_m,
                "signedSpinAM": metric.spin_a_m,
                "sourceId": metric.source_id,
            },
            "sourceReference": dict(PRIMARY_SOURCE_REFERENCE),
            "sourceReferenceHashSemantics": PRIMARY_SOURCE_HASH_SEMANTICS,
            "sourceReferenceSha256": PRIMARY_SOURCE_REFERENCE_SHA256,
            "surfaceEvent": {
                "blRadiusOverMass": point.radius_over_mass,
                "blThetaRad": point.theta_rad,
                "coordinateTimeM": coordinate_time,
                "face": face,
                "phiKsRad": phi,
                "pseudoCylindricalRadiusOverMass": rho,
                "signedHeightOverMass": point.signed_height_over_mass,
            },
            "velocity": {
                "angularVelocityInverseM": angular_velocity,
                "equatorialReferenceOmegaM": orbit.omega_m,
                "normalization": "exact Kerr metric at actual photosphere event",
                "prescription": "equatorial circular Omega at matching rho",
            },
        }
        descriptor_json = _canonical_json(descriptor)
        descriptor_sha256 = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()

        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "pseudo_cylindrical_radius_over_mass", rho)
        object.__setattr__(self, "face", face)
        object.__setattr__(self, "phi_ks_rad", phi)
        object.__setattr__(self, "coordinate_time_m", coordinate_time)
        object.__setattr__(self, "photosphere_point", point)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "four_velocity", four_velocity)
        object.__setattr__(self, "outward_unit_normal", outward_normal)
        object.__setattr__(
            self,
            "outward_unit_normal_covector",
            outward_normal_covector,
        )
        object.__setattr__(self, "angular_velocity_inverse_m", angular_velocity)
        object.__setattr__(self, "four_velocity_bl_time_component", u_t)
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(self, "_descriptor_sha256", descriptor_sha256)

    @property
    def model_descriptor_sha256(self) -> str:
        """SHA-256 of the complete canonical model descriptor JSON."""

        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        """Return a fresh JSON-compatible descriptor bound to this event."""

        return json.loads(self._descriptor_json)

    def project_past_directed_photon(
        self,
        photon_state: HamiltonianState,
        *,
        null_residual_limit: float = _DEFAULT_NULL_RESIDUAL_LIMIT,
        event_tolerance_m: float | None = None,
        backside_policy: BacksidePolicy = "reject",
    ) -> KerrFiniteThicknessPhotonProjection:
        """Project a null past-directed covector into this face matter frame.

        The returned convention is exactly
        ``nu_local = u^mu p_mu > 0`` and
        ``mu_out = -n^mu p_mu / (u^mu p_mu)``.  Positive affine rescaling
        changes ``nu_local`` by the same factor but leaves every gate,
        ``mu_out``, and the face classification unchanged.
        """

        if type(photon_state) is not HamiltonianState:
            raise TypeError("photon_state must be the exact HamiltonianState")
        residual_limit = _finite_number(null_residual_limit, "null_residual_limit")
        if residual_limit <= 0.0:
            raise ValueError("null_residual_limit must be positive")
        if backside_policy not in ("reject", "classify"):
            raise ValueError("backside_policy must be 'reject' or 'classify'")
        event_tolerance = (
            _DEFAULT_EVENT_TOLERANCE_OVER_MASS * self.metric.mass_m
            if event_tolerance_m is None
            else _finite_number(event_tolerance_m, "event_tolerance_m")
        )
        if not math.isfinite(event_tolerance) or event_tolerance < 0.0:
            raise ValueError("event_tolerance_m must be finite and non-negative")
        if any(
            abs(photon_state.event[index] - self.event[index]) > event_tolerance
            for index in range(4)
        ):
            raise ValueError("photon state does not match the authenticated face event")

        covector_scale = max(abs(value) for value in photon_state.covector)
        if not math.isfinite(covector_scale) or covector_scale <= 0.0:
            raise ValueError("photon covector must have a finite non-zero scale")
        normalized_covector = tuple(
            value / covector_scale for value in photon_state.covector
        )
        if not all(math.isfinite(value) for value in normalized_covector):
            raise ValueError("normalized photon covector is not finite")
        normalized_state = HamiltonianState(
            event=photon_state.event,
            covector=normalized_covector,  # type: ignore[arg-type]
        )
        null_residual = hamiltonian_null_residual(self.metric, normalized_state)
        if not math.isfinite(null_residual) or null_residual > residual_limit:
            raise ValueError("photon covector exceeds the null residual limit")

        normalized_local_frequency = math.fsum(
            self.four_velocity[index] * normalized_covector[index]
            for index in range(4)
        )
        if (
            not math.isfinite(normalized_local_frequency)
            or normalized_local_frequency <= 0.0
        ):
            raise ValueError("past-directed local photon frequency must be positive")
        local_frequency = covector_scale * normalized_local_frequency
        if not math.isfinite(local_frequency) or local_frequency <= 0.0:
            raise ValueError("local photon frequency over/underflowed its affine scale")

        outward_normal_projection = math.fsum(
            self.outward_unit_normal[index] * normalized_covector[index]
            for index in range(4)
        )
        outgoing_cosine = -outward_normal_projection / normalized_local_frequency
        if not math.isfinite(outgoing_cosine):
            raise KerrFiniteThicknessEmitterError(
                "local photon direction cosine is not finite"
            )
        if outgoing_cosine > 1.0 + _DIRECTION_COSINE_TOLERANCE:
            raise KerrFiniteThicknessEmitterError(
                "local photon direction exceeds the future null cone"
            )
        if outgoing_cosine < -1.0 - _DIRECTION_COSINE_TOLERANCE:
            raise KerrFiniteThicknessEmitterError(
                "local photon backside direction exceeds the future null cone"
            )
        outgoing_cosine = min(1.0, max(-1.0, outgoing_cosine))
        # Preserve the sign exactly.  In particular, a numerically small
        # negative value is still backside incidence rather than a tangent ray
        # manufactured by a tolerance or absolute value.
        if outgoing_cosine > 0.0:
            classification: PhotonFaceClassification = "outgoing"
        elif outgoing_cosine < 0.0:
            classification = "backside"
        else:
            classification = "tangent"
        if classification == "backside" and backside_policy == "reject":
            raise BacksidePhotonError(
                f"photon illuminates the back of the {self.face} face "
                f"(signed mu={outgoing_cosine:.17g})"
            )
        return KerrFiniteThicknessPhotonProjection(
            local_frequency=local_frequency,
            normalized_local_frequency=normalized_local_frequency,
            affine_covector_scale=covector_scale,
            outgoing_cosine=outgoing_cosine,
            face_classification=classification,
            null_residual=null_residual,
        )


__all__ = (
    "BacksidePhotonError",
    "IMPLEMENTATION_ID",
    "KerrFiniteThicknessEmitterError",
    "KerrFiniteThicknessFaceEmitter",
    "KerrFiniteThicknessPhotonProjection",
    "PRIMARY_SOURCE_HASH_SEMANTICS",
    "PRIMARY_SOURCE_REFERENCE",
    "PRIMARY_SOURCE_REFERENCE_SHA256",
    "PRIMARY_SOURCE_URL",
    "SCIENTIFIC_STATUS",
)
