"""Comoving proper area of the prescribed finite-thickness Kerr faces.

For one upper or lower photosphere, use the local Boyer--Lindquist embedding

``X^mu(rho, phi) = (t_0, r(rho), theta(rho), phi)``

with ``r = hypot(rho, z)`` and ``theta = atan2(rho, z)``.  The two embedding
tangents are transformed through the repository's exact BL-to-Cartesian-KS
Jacobian at the *actual* face event.  The actual-face stationary emitter owns
the matter four-velocity.  With signature ``-+++``, the matter-rest-space
projector and the induced two-metric are

``h_mu_nu = g_mu_nu + u_mu u_nu``

``q_AB = h_mu_nu X^mu_,A X^nu_,B``.

The local area density is ``sqrt(det(q))`` per ``d(rho/M) dphi``.  Annulus
area uses independent ``N`` and ``2N`` Gauss--Legendre radial integrals and
the exact stationary-axisymmetric factor ``2 pi``.  Failure of the declared
finite convergence gate raises instead of publishing an unconverged value.

This is the comoving proper area of a stationary *prescribed* photosphere.
It is not a returning-radiation kernel, a receiver solid-angle Jacobian, an
atmosphere, a hydrostatic solution, or GRMHD.  In particular, it must not be
substituted for the ray-bundle Jacobian needed by a returning-radiation
transfer kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_vector_to_ks_cartesian,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    MODEL_IMPLEMENTATION_ID as SURFACE_IMPLEMENTATION_ID,
    StationaryKerrFiniteThicknessCalibration,
    VALID_FACES,
)
from offline.kerr_finite_thickness_emitter import (
    IMPLEMENTATION_ID as EMITTER_IMPLEMENTATION_ID,
    KerrFiniteThicknessFaceEmitter,
)
from offline.spacetime import Matrix4, Vector4, bilinear, matrix_vector


IMPLEMENTATION_ID: Final = "stationary-kerr-finite-thickness-comoving-area/v1"

MINIMUM_GAUSS_LEGENDRE_ORDER: Final = 4
MAXIMUM_GAUSS_LEGENDRE_ORDER: Final = 128
MAXIMUM_POINT_EVALUATIONS: Final = 384
MAXIMUM_RELATIVE_CONVERGENCE_TOLERANCE: Final = 1.0e-7
MAXIMUM_ABSOLUTE_CONVERGENCE_TOLERANCE_OVER_MASS_SQUARED: Final = 1.0e-7
_GAUSS_MAXIMUM_ITERATIONS: Final = 64
_TANGENCY_TOLERANCE_OVER_MASS: Final = 2.0e-9
_DETERMINANT_ROUNDOFF_FACTOR: Final = 1024.0

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "stationary prescribed finite-height Kerr photosphere comoving "
            "proper area"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "surfaceImplementationId": SURFACE_IMPLEMENTATION_ID,
        "emitterImplementationId": EMITTER_IMPLEMENTATION_ID,
        "embedding": (
            "constant-Boyer-Lindquist-time X^mu(rho,phi) evaluated through "
            "the exact Kerr BL-to-Cartesian-KS Jacobian"
        ),
        "restSpaceProjector": "h_mu_nu = g_mu_nu + u_mu u_nu",
        "inducedMetric": "q_AB = h_mu_nu X^mu_,A X^nu_,B",
        "areaDensity": "sqrt(det(q)) per d(rho/M) dphi",
        "annulusQuadrature": (
            "independent N and 2N radial Gauss-Legendre rules; exact 2 pi "
            "axisymmetry factor"
        ),
        "isStationaryPrescribedSurfaceArea": True,
        "isReturningRadiationKernel": False,
        "isReceiverSolidAngleAreaJacobian": False,
        "includesReturningRadiationStressWork": False,
        "isHydrostaticVerticalStructureSolution": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "Do not describe this stationary prescribed-surface proper area "
            "as a returning-radiation kernel, receiver ray-bundle Jacobian, "
            "hydrostatic solution, atmosphere, or GRMHD."
        ),
    }
)


class KerrFiniteThicknessAreaError(RuntimeError):
    """Raised when a finite-thickness area calculation is not physical."""


class KerrFiniteThicknessAreaConvergenceError(KerrFiniteThicknessAreaError):
    """Raised when the mandatory N/2N annulus gate does not converge."""


class KerrFiniteThicknessAreaVerificationError(KerrFiniteThicknessAreaError):
    """Raised when a stored area result cannot be reproduced exactly."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrFiniteThicknessAreaError(
            "finite-thickness area descriptor is not finite canonical JSON"
        ) from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _exact_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise KerrFiniteThicknessAreaVerificationError(
            f"{label} must be a finite exact float"
        )
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise KerrFiniteThicknessAreaVerificationError(
            f"{label} must be an exact int"
        )
    return value


def _exact_str(value: Any, label: str) -> str:
    if type(value) is not str:
        raise KerrFiniteThicknessAreaVerificationError(
            f"{label} must be an exact str"
        )
    return value


def _face(value: Any) -> str:
    if type(value) is not str or value not in VALID_FACES:
        raise ValueError("face must be the exact built-in 'upper' or 'lower' str")
    return value


def _vector_exact(value: Any, label: str) -> Vector4:
    if type(value) is not tuple or len(value) != 4:
        raise KerrFiniteThicknessAreaVerificationError(
            f"{label} must be an exact four-component tuple"
        )
    checked = tuple(
        _exact_finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )
    return checked  # type: ignore[return-value]


def _metric_and_calibration(
    metric: Any,
    calibration: Any,
) -> tuple[KerrKerrSchildMetric, StationaryKerrFiniteThicknessCalibration]:
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("metric must be the exact built-in KerrKerrSchildMetric")
    if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError(
            "calibration must be the exact built-in "
            "StationaryKerrFiniteThicknessCalibration"
        )
    # The actual-face emitter owns the final consistency check.  This early
    # check gives annulus callers a deterministic failure before quadrature.
    if not math.isclose(
        abs(metric.dimensionless_spin),
        calibration.dimensionless_spin,
        rel_tol=0.0,
        abs_tol=64.0 * math.ulp(1.0),
    ):
        raise ValueError("metric and calibration spin magnitudes disagree")
    if calibration.eddington_scaled_mass_accretion_rate <= 0.0:
        raise ValueError(
            "positive finite thickness is required; the coincident dotm=0 "
            "faces belong to the zero-thickness disk path"
        )
    return metric, calibration


def _calibration_descriptor(
    calibration: StationaryKerrFiniteThicknessCalibration,
) -> dict[str, Any]:
    return {
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
    }


def _metric_descriptor(metric: KerrKerrSchildMetric) -> dict[str, Any]:
    return {
        "massM": metric.mass_m,
        "signedSpinAM": metric.spin_a_m,
        "singularityGuardM": metric.singularity_guard_m,
        "sourceId": metric.source_id,
    }


def _projector(
    metric_covariant: Matrix4,
    four_velocity: Vector4,
) -> Matrix4:
    velocity_covector = matrix_vector(metric_covariant, four_velocity)
    result: Matrix4 = tuple(  # type: ignore[assignment]
        tuple(
            math.fsum(
                (
                    metric_covariant[row][column],
                    velocity_covector[row] * velocity_covector[column],
                )
            )
            for column in range(4)
        )
        for row in range(4)
    )
    if not all(math.isfinite(value) for row in result for value in row):
        raise KerrFiniteThicknessAreaError("matter-rest-space projector is invalid")
    return result


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessAreaQuadraturePolicy:
    """Finite N/2N Gauss--Legendre convergence and work budget."""

    gauss_legendre_order: int = 24
    relative_tolerance: float = 2.0e-10
    absolute_tolerance_over_mass_squared: float = 2.0e-11
    maximum_point_evaluations: int = MAXIMUM_POINT_EVALUATIONS

    def __post_init__(self) -> None:
        if type(self.gauss_legendre_order) is not int:
            raise TypeError("gauss_legendre_order must be an exact int")
        if not (
            MINIMUM_GAUSS_LEGENDRE_ORDER
            <= self.gauss_legendre_order
            <= MAXIMUM_GAUSS_LEGENDRE_ORDER
        ):
            raise ValueError(
                "gauss_legendre_order lies outside the certified range "
                f"[{MINIMUM_GAUSS_LEGENDRE_ORDER}, "
                f"{MAXIMUM_GAUSS_LEGENDRE_ORDER}]"
            )
        relative = _finite_number(self.relative_tolerance, "relative_tolerance")
        absolute = _finite_number(
            self.absolute_tolerance_over_mass_squared,
            "absolute_tolerance_over_mass_squared",
        )
        if relative <= 0.0 or relative > MAXIMUM_RELATIVE_CONVERGENCE_TOLERANCE:
            raise ValueError(
                "relative_tolerance must be positive and cannot loosen the "
                f"policy maximum {MAXIMUM_RELATIVE_CONVERGENCE_TOLERANCE}"
            )
        if (
            absolute <= 0.0
            or absolute
            > MAXIMUM_ABSOLUTE_CONVERGENCE_TOLERANCE_OVER_MASS_SQUARED
        ):
            raise ValueError(
                "absolute_tolerance_over_mass_squared must be positive and "
                "cannot loosen the policy maximum "
                f"{MAXIMUM_ABSOLUTE_CONVERGENCE_TOLERANCE_OVER_MASS_SQUARED}"
            )
        if type(self.maximum_point_evaluations) is not int:
            raise TypeError("maximum_point_evaluations must be an exact int")
        required = 3 * self.gauss_legendre_order
        if not required <= self.maximum_point_evaluations <= MAXIMUM_POINT_EVALUATIONS:
            raise ValueError(
                "maximum_point_evaluations must cover the independent N and "
                f"2N rules ({required}) without exceeding "
                f"{MAXIMUM_POINT_EVALUATIONS}"
            )
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(
            self,
            "absolute_tolerance_over_mass_squared",
            absolute,
        )

    @property
    def required_point_evaluations(self) -> int:
        return 3 * self.gauss_legendre_order

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "absoluteToleranceOverMassSquared": (
                    self.absolute_tolerance_over_mass_squared
                ),
                "coarseOrder": self.gauss_legendre_order,
                "maximumPointEvaluations": self.maximum_point_evaluations,
                "relativeTolerance": self.relative_tolerance,
                "requiredPointEvaluations": self.required_point_evaluations,
                "fineOrder": 2 * self.gauss_legendre_order,
            }
        )


@dataclass(frozen=True, slots=True)
class _AreaPointRaw:
    embedding_radial_tangent_ks: Vector4
    embedding_azimuthal_tangent_ks: Vector4
    q_rho_rho_m2: float
    q_rho_phi_m2: float
    q_phi_phi_m2: float
    determinant_m4: float
    proper_area_density_m2: float
    proper_area_density_over_mass_squared: float
    maximum_tangency_residual_over_mass: float
    emitter_model_descriptor_sha256: str


def _compute_area_point(
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    rho: float,
    face: str,
    phi_ks_rad: float,
    coordinate_time_m: float,
) -> _AreaPointRaw:
    emitter = KerrFiniteThicknessFaceEmitter(
        metric=metric,
        calibration=calibration,
        pseudo_cylindrical_radius_over_mass=rho,
        face=face,  # type: ignore[arg-type]
        phi_ks_rad=phi_ks_rad,
        coordinate_time_m=coordinate_time_m,
    )
    point = emitter.photosphere_point
    face_sign = 1.0 if face == UPPER else -1.0
    signed_slope = face_sign * calibration.photosphere_height_derivative(rho)
    signed_height = point.signed_height_over_mass
    radius_over_mass = point.radius_over_mass
    radius_derivative = math.fsum((rho, signed_height * signed_slope)) / radius_over_mass
    theta_derivative = math.fsum(
        (signed_height, -rho * signed_slope)
    ) / (radius_over_mass * radius_over_mass)
    if not all(
        math.isfinite(value) for value in (radius_derivative, theta_derivative)
    ):
        raise KerrFiniteThicknessAreaError("surface embedding derivative is invalid")

    radius_m = radius_over_mass * metric.mass_m
    # The embedding parameter is rho/M.  Consequently dr/d(rho/M) carries
    # one factor of M, while dtheta/d(rho/M) is dimensionless.  Both returned
    # KS tangents therefore have components measured per dimensionless chart
    # parameter and q_AB has units M^2 for A,B in (rho/M, phi).
    radial_tangent = kerr_bl_vector_to_ks_cartesian(
        (0.0, metric.mass_m * radius_derivative, theta_derivative, 0.0),
        mass_m=metric.mass_m,
        spin_a_m=metric.spin_a_m,
        radius_m=radius_m,
        theta_rad=point.theta_rad,
        phi_ks_rad=phi_ks_rad,
    )
    azimuthal_tangent = kerr_bl_vector_to_ks_cartesian(
        (0.0, 0.0, 0.0, 1.0),
        mass_m=metric.mass_m,
        spin_a_m=metric.spin_a_m,
        radius_m=radius_m,
        theta_rad=point.theta_rad,
        phi_ks_rad=phi_ks_rad,
    )
    if not all(
        math.isfinite(value)
        for tangent in (radial_tangent, azimuthal_tangent)
        for value in tangent
    ):
        raise KerrFiniteThicknessAreaError("surface embedding tangent is invalid")

    sample = metric.sample(emitter.event)
    projector = _projector(sample.covariant, emitter.four_velocity)
    q_rho_rho = bilinear(radial_tangent, projector, radial_tangent)
    q_rho_phi = bilinear(radial_tangent, projector, azimuthal_tangent)
    q_phi_phi = bilinear(azimuthal_tangent, projector, azimuthal_tangent)
    determinant = math.fsum(
        (q_rho_rho * q_phi_phi, -q_rho_phi * q_rho_phi)
    )
    determinant_scale = max(
        abs(q_rho_rho * q_phi_phi),
        abs(q_rho_phi * q_rho_phi),
        metric.mass_m**4,
        1.0e-300,
    )
    if (
        not all(
            math.isfinite(value)
            for value in (q_rho_rho, q_rho_phi, q_phi_phi, determinant)
        )
        or q_rho_rho <= 0.0
        or q_phi_phi <= 0.0
        or determinant
        <= _DETERMINANT_ROUNDOFF_FACTOR * math.ulp(1.0) * determinant_scale
    ):
        raise KerrFiniteThicknessAreaError(
            "projected photosphere two-metric is not positive definite"
        )
    density_m2 = math.sqrt(determinant)
    mass_squared = metric.mass_m * metric.mass_m
    density_over_m2 = density_m2 / mass_squared
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (density_m2, density_over_m2)
    ):
        raise KerrFiniteThicknessAreaError("proper area density is invalid")

    normal_covector = emitter.outward_unit_normal_covector
    tangency = max(
        abs(math.fsum(normal_covector[index] * tangent[index] for index in range(4)))
        for tangent in (radial_tangent, azimuthal_tangent)
    ) / metric.mass_m
    if not math.isfinite(tangency) or tangency > _TANGENCY_TOLERANCE_OVER_MASS:
        raise KerrFiniteThicknessAreaError(
            "embedding tangent is inconsistent with the actual photosphere face"
        )
    return _AreaPointRaw(
        embedding_radial_tangent_ks=radial_tangent,
        embedding_azimuthal_tangent_ks=azimuthal_tangent,
        q_rho_rho_m2=q_rho_rho,
        q_rho_phi_m2=q_rho_phi,
        q_phi_phi_m2=q_phi_phi,
        determinant_m4=determinant,
        proper_area_density_m2=density_m2,
        proper_area_density_over_mass_squared=density_over_m2,
        maximum_tangency_residual_over_mass=tangency,
        emitter_model_descriptor_sha256=emitter.model_descriptor_sha256,
    )


def _point_descriptor(
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    rho: float,
    face: str,
    phi_ks_rad: float,
    coordinate_time_m: float,
    raw: _AreaPointRaw,
) -> dict[str, Any]:
    return {
        "areaDensity": {
            "determinantM4": raw.determinant_m4,
            "properAreaDensityM2": raw.proper_area_density_m2,
            "properAreaDensityOverMassSquared": (
                raw.proper_area_density_over_mass_squared
            ),
            "qPhiPhiM2": raw.q_phi_phi_m2,
            "qRhoPhiM2": raw.q_rho_phi_m2,
            "qRhoRhoM2": raw.q_rho_rho_m2,
        },
        "calibration": _calibration_descriptor(calibration),
        "embedding": {
            "azimuthalTangentKs": raw.embedding_azimuthal_tangent_ks,
            "chart": "X=(t_BL constant,r(rho),theta(rho),phi_BL)",
            "coordinateTimeKsAtReferenceEventM": coordinate_time_m,
            "face": face,
            "maximumTangencyResidualOverMass": (
                raw.maximum_tangency_residual_over_mass
            ),
            "phiKsAtReferenceEventRad": phi_ks_rad,
            "pseudoCylindricalRadiusOverMass": rho,
            "radialTangentKs": raw.embedding_radial_tangent_ks,
        },
        "emitterImplementationId": EMITTER_IMPLEMENTATION_ID,
        "emitterModelDescriptorSha256": raw.emitter_model_descriptor_sha256,
        "formula": {
            "areaDensity": "sqrt(det(q))",
            "inducedMetric": "q_AB=h_mu_nu X^mu_,A X^nu_,B",
            "projector": "h_mu_nu=g_mu_nu+u_mu u_nu",
        },
        "implementationId": IMPLEMENTATION_ID,
        "metric": _metric_descriptor(metric),
        "scientificStatus": dict(SCIENTIFIC_STATUS),
    }


@dataclass(frozen=True, slots=True, init=False)
class KerrFiniteThicknessAreaDensity:
    """Self-replayable comoving proper-area density at one actual face point."""

    pseudo_cylindrical_radius_over_mass: float
    face: str
    phi_ks_rad: float
    coordinate_time_m: float
    embedding_radial_tangent_ks: Vector4
    embedding_azimuthal_tangent_ks: Vector4
    q_rho_rho_m2: float
    q_rho_phi_m2: float
    q_phi_phi_m2: float
    determinant_m4: float
    proper_area_density_m2: float
    proper_area_density_over_mass_squared: float
    maximum_tangency_residual_over_mass: float
    emitter_model_descriptor_sha256: str
    _metric: KerrKerrSchildMetric
    _calibration: StationaryKerrFiniteThicknessCalibration
    _descriptor_json: str
    _descriptor_sha256: str

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_finite_thickness_area_density(self)


def kerr_finite_thickness_area_density(
    *,
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    pseudo_cylindrical_radius_over_mass: float,
    face: str,
    phi_ks_rad: float = 0.0,
    coordinate_time_m: float = 0.0,
) -> KerrFiniteThicknessAreaDensity:
    """Return ``sqrt(det(q))`` per ``d(rho/M) dphi`` at one face event."""

    metric, calibration = _metric_and_calibration(metric, calibration)
    rho = _finite_number(
        pseudo_cylindrical_radius_over_mass,
        "pseudo_cylindrical_radius_over_mass",
    )
    selected_face = _face(face)
    phi = _finite_number(phi_ks_rad, "phi_ks_rad")
    coordinate_time = _finite_number(coordinate_time_m, "coordinate_time_m")
    if rho <= calibration.isco_radius_over_mass:
        raise ValueError("area-density radius must lie strictly outside the ISCO seam")
    if rho > calibration.outer_radius_over_mass:
        raise ValueError("area-density radius lies outside the calibration annulus")
    raw = _compute_area_point(
        metric,
        calibration,
        rho,
        selected_face,
        phi,
        coordinate_time,
    )
    descriptor_json = _canonical_json(
        _point_descriptor(
            metric,
            calibration,
            rho,
            selected_face,
            phi,
            coordinate_time,
            raw,
        )
    )
    result = object.__new__(KerrFiniteThicknessAreaDensity)
    for name, value in (
        ("pseudo_cylindrical_radius_over_mass", rho),
        ("face", selected_face),
        ("phi_ks_rad", phi),
        ("coordinate_time_m", coordinate_time),
        ("embedding_radial_tangent_ks", raw.embedding_radial_tangent_ks),
        ("embedding_azimuthal_tangent_ks", raw.embedding_azimuthal_tangent_ks),
        ("q_rho_rho_m2", raw.q_rho_rho_m2),
        ("q_rho_phi_m2", raw.q_rho_phi_m2),
        ("q_phi_phi_m2", raw.q_phi_phi_m2),
        ("determinant_m4", raw.determinant_m4),
        ("proper_area_density_m2", raw.proper_area_density_m2),
        (
            "proper_area_density_over_mass_squared",
            raw.proper_area_density_over_mass_squared,
        ),
        (
            "maximum_tangency_residual_over_mass",
            raw.maximum_tangency_residual_over_mass,
        ),
        (
            "emitter_model_descriptor_sha256",
            raw.emitter_model_descriptor_sha256,
        ),
        ("_metric", metric),
        ("_calibration", calibration),
        ("_descriptor_json", descriptor_json),
        ("_descriptor_sha256", _sha256_text(descriptor_json)),
    ):
        object.__setattr__(result, name, value)
    return result


def _verify_point_fields(
    actual: KerrFiniteThicknessAreaDensity,
    expected: KerrFiniteThicknessAreaDensity,
) -> None:
    float_fields = (
        "pseudo_cylindrical_radius_over_mass",
        "phi_ks_rad",
        "coordinate_time_m",
        "q_rho_rho_m2",
        "q_rho_phi_m2",
        "q_phi_phi_m2",
        "determinant_m4",
        "proper_area_density_m2",
        "proper_area_density_over_mass_squared",
        "maximum_tangency_residual_over_mass",
    )
    for name in float_fields:
        actual_value = _exact_finite_float(object.__getattribute__(actual, name), name)
        expected_value = object.__getattribute__(expected, name)
        if actual_value.hex() != expected_value.hex():
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match exact metric replay"
            )
    for name in ("embedding_radial_tangent_ks", "embedding_azimuthal_tangent_ks"):
        actual_vector = _vector_exact(object.__getattribute__(actual, name), name)
        expected_vector = object.__getattribute__(expected, name)
        if any(
            actual_vector[index].hex() != expected_vector[index].hex()
            for index in range(4)
        ):
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match exact metric replay"
            )
    for name in (
        "face",
        "emitter_model_descriptor_sha256",
        "_descriptor_json",
        "_descriptor_sha256",
    ):
        actual_value = _exact_str(object.__getattribute__(actual, name), name)
        expected_value = object.__getattribute__(expected, name)
        if actual_value != expected_value:
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match exact metric replay"
            )


def verify_kerr_finite_thickness_area_density(
    result: KerrFiniteThicknessAreaDensity,
) -> None:
    """Replay one point from its exact metric/calibration and reject tampering."""

    if type(result) is not KerrFiniteThicknessAreaDensity:
        raise TypeError("result must be the exact KerrFiniteThicknessAreaDensity")
    metric = object.__getattribute__(result, "_metric")
    calibration = object.__getattribute__(result, "_calibration")
    _metric_and_calibration(metric, calibration)
    rho = _exact_finite_float(
        object.__getattribute__(result, "pseudo_cylindrical_radius_over_mass"),
        "pseudo_cylindrical_radius_over_mass",
    )
    face = _exact_str(object.__getattribute__(result, "face"), "face")
    phi = _exact_finite_float(object.__getattribute__(result, "phi_ks_rad"), "phi_ks_rad")
    coordinate_time = _exact_finite_float(
        object.__getattribute__(result, "coordinate_time_m"),
        "coordinate_time_m",
    )
    try:
        expected = kerr_finite_thickness_area_density(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,
            phi_ks_rad=phi,
            coordinate_time_m=coordinate_time,
        )
    except (TypeError, ValueError, KerrFiniteThicknessAreaError) as error:
        raise KerrFiniteThicknessAreaVerificationError(
            "area-density replay failed"
        ) from error
    _verify_point_fields(result, expected)


@lru_cache(maxsize=2 * MAXIMUM_GAUSS_LEGENDRE_ORDER)
def _gauss_legendre_unit_interval(order: int) -> tuple[tuple[float, float], ...]:
    """Return deterministic float64 Gauss--Legendre nodes and weights."""

    if type(order) is not int or not 1 <= order <= 2 * MAXIMUM_GAUSS_LEGENDRE_ORDER:
        raise ValueError("Gauss-Legendre order is outside the internal bound")
    nodes = [0.0] * order
    weights = [0.0] * order
    half = (order + 1) // 2
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (order + 0.5))
        derivative = 0.0
        visited_iterates: list[tuple[float, float, float]] = []
        for _iteration in range(_GAUSS_MAXIMUM_ITERATIONS):
            previous = 1.0
            current = root
            for degree in range(2, order + 1):
                following = (
                    (2.0 * degree - 1.0) * root * current
                    - (degree - 1.0) * previous
                ) / degree
                previous, current = current, following
            if order == 1:
                current = root
                previous = 1.0
            derivative = order * (root * current - previous) / (root * root - 1.0)
            update = current / derivative
            next_root = root - update
            if next_root == root or abs(update) <= 2.0 * math.ulp(root):
                root = next_root
                break
            visited_iterates.append((root, current, derivative))
            cycle_start = next(
                (
                    item_index
                    for item_index, (item_root, _residual, _derivative) in enumerate(
                        visited_iterates
                    )
                    if next_root == item_root
                ),
                None,
            )
            if cycle_start is not None:
                # Some high orders converge to a short binary64 cycle.  Select
                # the smaller-residual visited endpoint without perturbing any
                # non-cycling Newton trajectory.
                root, _residual, derivative = min(
                    visited_iterates[cycle_start:],
                    key=lambda item: (abs(item[1]), item[0]),
                )
                break
            root = next_root
        else:
            raise KerrFiniteThicknessAreaError(
                "Gauss-Legendre root solve did not converge"
            )
        weight = 1.0 / ((1.0 - root * root) * derivative * derivative)
        lower = 0.5 * (1.0 - root)
        upper = 0.5 * (1.0 + root)
        nodes[index] = lower
        nodes[order - 1 - index] = upper
        weights[index] = weight
        weights[order - 1 - index] = weight
    result = tuple(zip(nodes, weights))
    if any(
        not (0.0 < node < 1.0 and math.isfinite(weight) and weight > 0.0)
        for node, weight in result
    ):
        raise KerrFiniteThicknessAreaError(
            "Gauss-Legendre rule is not finite and interior-positive"
        )
    if abs(math.fsum(weight for _node, weight in result) - 1.0) > 64.0 * math.ulp(1.0):
        raise KerrFiniteThicknessAreaError(
            "Gauss-Legendre weights do not integrate a constant"
        )
    return result


@dataclass(frozen=True, slots=True)
class _IntegratedOrder:
    area_over_mass_squared: float
    minimum_density_over_mass_squared: float
    maximum_density_over_mass_squared: float
    maximum_tangency_residual_over_mass: float


def _integrate_order(
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    inner: float,
    outer: float,
    face: str,
    phi_ks_rad: float,
    coordinate_time_m: float,
    order: int,
) -> _IntegratedOrder:
    width = outer - inner
    weighted_densities: list[float] = []
    densities: list[float] = []
    tangencies: list[float] = []
    for node, weight in _gauss_legendre_unit_interval(order):
        rho = math.fsum((inner, width * node))
        raw = _compute_area_point(
            metric,
            calibration,
            rho,
            face,
            phi_ks_rad,
            coordinate_time_m,
        )
        density = raw.proper_area_density_over_mass_squared
        densities.append(density)
        tangencies.append(raw.maximum_tangency_residual_over_mass)
        weighted_densities.append(weight * density)
    area = 2.0 * math.pi * width * math.fsum(weighted_densities)
    if not math.isfinite(area) or area <= 0.0:
        raise KerrFiniteThicknessAreaError("annulus proper area is invalid")
    return _IntegratedOrder(
        area_over_mass_squared=area,
        minimum_density_over_mass_squared=min(densities),
        maximum_density_over_mass_squared=max(densities),
        maximum_tangency_residual_over_mass=max(tangencies),
    )


@dataclass(frozen=True, slots=True, init=False)
class KerrFiniteThicknessAnnulusArea:
    """Self-replayable N/2N-converged proper area of one photosphere annulus."""

    inner_radius_over_mass: float
    outer_radius_over_mass: float
    face: str
    phi_ks_rad: float
    coordinate_time_m: float
    coarse_order: int
    fine_order: int
    point_evaluations: int
    maximum_point_evaluations: int
    coarse_area_over_mass_squared: float
    fine_area_over_mass_squared: float
    proper_area_over_mass_squared: float
    proper_area_m2: float
    estimated_absolute_error_over_mass_squared: float
    estimated_relative_error: float
    convergence_threshold_over_mass_squared: float
    minimum_sampled_density_over_mass_squared: float
    maximum_sampled_density_over_mass_squared: float
    maximum_tangency_residual_over_mass: float
    _metric: KerrKerrSchildMetric
    _calibration: StationaryKerrFiniteThicknessCalibration
    _policy: KerrFiniteThicknessAreaQuadraturePolicy
    _descriptor_json: str
    _descriptor_sha256: str

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_finite_thickness_annulus_area(self)


def _annulus_descriptor(
    result_values: Mapping[str, Any],
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    policy: KerrFiniteThicknessAreaQuadraturePolicy,
) -> dict[str, Any]:
    return {
        "annulus": dict(result_values),
        "axisymmetry": {
            "azimuthalFactor": 2.0 * math.pi,
            "policy": (
                "stationary axisymmetry is exact; radial density is evaluated "
                "at one reference azimuth"
            ),
        },
        "calibration": _calibration_descriptor(calibration),
        "implementationId": IMPLEMENTATION_ID,
        "metric": _metric_descriptor(metric),
        "quadrature": dict(policy.descriptor()),
        "scientificStatus": dict(SCIENTIFIC_STATUS),
    }


def integrate_kerr_finite_thickness_annulus_area(
    *,
    metric: KerrKerrSchildMetric,
    calibration: StationaryKerrFiniteThicknessCalibration,
    inner_radius_over_mass: float,
    outer_radius_over_mass: float,
    face: str,
    policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    phi_ks_rad: float = 0.0,
    coordinate_time_m: float = 0.0,
) -> KerrFiniteThicknessAnnulusArea:
    """Integrate one face annulus and require independent N/2N agreement."""

    metric, calibration = _metric_and_calibration(metric, calibration)
    inner = _finite_number(inner_radius_over_mass, "inner_radius_over_mass")
    outer = _finite_number(outer_radius_over_mass, "outer_radius_over_mass")
    selected_face = _face(face)
    phi = _finite_number(phi_ks_rad, "phi_ks_rad")
    coordinate_time = _finite_number(coordinate_time_m, "coordinate_time_m")
    selected_policy = (
        KerrFiniteThicknessAreaQuadraturePolicy() if policy is None else policy
    )
    if type(selected_policy) is not KerrFiniteThicknessAreaQuadraturePolicy:
        raise TypeError(
            "policy must be the exact KerrFiniteThicknessAreaQuadraturePolicy"
        )
    # Re-run post-init semantics to reject object.__setattr__ policy tampering.
    selected_policy = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=selected_policy.gauss_legendre_order,
        relative_tolerance=selected_policy.relative_tolerance,
        absolute_tolerance_over_mass_squared=(
            selected_policy.absolute_tolerance_over_mass_squared
        ),
        maximum_point_evaluations=selected_policy.maximum_point_evaluations,
    )
    if inner < calibration.isco_radius_over_mass:
        raise ValueError("annulus inner radius lies inside the ISCO")
    if outer > calibration.outer_radius_over_mass:
        raise ValueError("annulus outer radius exceeds the calibrated surface")
    if not inner < outer:
        raise ValueError("annulus requires inner_radius_over_mass < outer_radius_over_mass")

    coarse = _integrate_order(
        metric,
        calibration,
        inner,
        outer,
        selected_face,
        phi,
        coordinate_time,
        selected_policy.gauss_legendre_order,
    )
    fine = _integrate_order(
        metric,
        calibration,
        inner,
        outer,
        selected_face,
        phi,
        coordinate_time,
        2 * selected_policy.gauss_legendre_order,
    )
    absolute_error = abs(
        fine.area_over_mass_squared - coarse.area_over_mass_squared
    )
    relative_error = absolute_error / max(
        abs(fine.area_over_mass_squared),
        1.0e-300,
    )
    threshold = max(
        selected_policy.absolute_tolerance_over_mass_squared,
        selected_policy.relative_tolerance * abs(fine.area_over_mass_squared),
    )
    if not all(
        math.isfinite(value)
        for value in (absolute_error, relative_error, threshold)
    ):
        raise KerrFiniteThicknessAreaError("annulus convergence diagnostic is invalid")
    if absolute_error > threshold:
        raise KerrFiniteThicknessAreaConvergenceError(
            "finite-thickness annulus failed N/2N convergence: "
            f"|A_2N-A_N|/M^2={absolute_error:.17g} exceeds "
            f"{threshold:.17g}"
        )
    mass_squared = metric.mass_m * metric.mass_m
    proper_area_m2 = fine.area_over_mass_squared * mass_squared
    if not math.isfinite(proper_area_m2) or proper_area_m2 <= 0.0:
        raise KerrFiniteThicknessAreaError(
            "dimensionful annulus proper area over/underflowed"
        )
    result_values: dict[str, Any] = {
        "coarseAreaOverMassSquared": coarse.area_over_mass_squared,
        "coarseOrder": selected_policy.gauss_legendre_order,
        "convergenceThresholdOverMassSquared": threshold,
        "coordinateTimeKsAtReferenceEventM": coordinate_time,
        "estimatedAbsoluteErrorOverMassSquared": absolute_error,
        "estimatedRelativeError": relative_error,
        "face": selected_face,
        "fineAreaOverMassSquared": fine.area_over_mass_squared,
        "fineOrder": 2 * selected_policy.gauss_legendre_order,
        "innerRadiusOverMass": inner,
        "maximumPointEvaluations": selected_policy.maximum_point_evaluations,
        "maximumSampledDensityOverMassSquared": max(
            coarse.maximum_density_over_mass_squared,
            fine.maximum_density_over_mass_squared,
        ),
        "maximumTangencyResidualOverMass": max(
            coarse.maximum_tangency_residual_over_mass,
            fine.maximum_tangency_residual_over_mass,
        ),
        "minimumSampledDensityOverMassSquared": min(
            coarse.minimum_density_over_mass_squared,
            fine.minimum_density_over_mass_squared,
        ),
        "outerRadiusOverMass": outer,
        "phiKsAtReferenceEventRad": phi,
        "pointEvaluations": selected_policy.required_point_evaluations,
        "properAreaM2": proper_area_m2,
        "properAreaOverMassSquared": fine.area_over_mass_squared,
    }
    descriptor_json = _canonical_json(
        _annulus_descriptor(
            result_values,
            metric,
            calibration,
            selected_policy,
        )
    )
    result = object.__new__(KerrFiniteThicknessAnnulusArea)
    field_values = {
        "inner_radius_over_mass": inner,
        "outer_radius_over_mass": outer,
        "face": selected_face,
        "phi_ks_rad": phi,
        "coordinate_time_m": coordinate_time,
        "coarse_order": selected_policy.gauss_legendre_order,
        "fine_order": 2 * selected_policy.gauss_legendre_order,
        "point_evaluations": selected_policy.required_point_evaluations,
        "maximum_point_evaluations": selected_policy.maximum_point_evaluations,
        "coarse_area_over_mass_squared": coarse.area_over_mass_squared,
        "fine_area_over_mass_squared": fine.area_over_mass_squared,
        "proper_area_over_mass_squared": fine.area_over_mass_squared,
        "proper_area_m2": proper_area_m2,
        "estimated_absolute_error_over_mass_squared": absolute_error,
        "estimated_relative_error": relative_error,
        "convergence_threshold_over_mass_squared": threshold,
        "minimum_sampled_density_over_mass_squared": (
            result_values["minimumSampledDensityOverMassSquared"]
        ),
        "maximum_sampled_density_over_mass_squared": (
            result_values["maximumSampledDensityOverMassSquared"]
        ),
        "maximum_tangency_residual_over_mass": (
            result_values["maximumTangencyResidualOverMass"]
        ),
        "_metric": metric,
        "_calibration": calibration,
        "_policy": selected_policy,
        "_descriptor_json": descriptor_json,
        "_descriptor_sha256": _sha256_text(descriptor_json),
    }
    for name, value in field_values.items():
        object.__setattr__(result, name, value)
    return result


def verify_kerr_finite_thickness_annulus_area(
    result: KerrFiniteThicknessAnnulusArea,
) -> None:
    """Replay both quadrature orders and reject any stored-field tampering."""

    if type(result) is not KerrFiniteThicknessAnnulusArea:
        raise TypeError("result must be the exact KerrFiniteThicknessAnnulusArea")
    metric = object.__getattribute__(result, "_metric")
    calibration = object.__getattribute__(result, "_calibration")
    policy = object.__getattribute__(result, "_policy")
    if type(policy) is not KerrFiniteThicknessAreaQuadraturePolicy:
        raise KerrFiniteThicknessAreaVerificationError(
            "stored quadrature policy has a non-exact type"
        )
    float_inputs = {
        name: _exact_finite_float(object.__getattribute__(result, name), name)
        for name in (
            "inner_radius_over_mass",
            "outer_radius_over_mass",
            "phi_ks_rad",
            "coordinate_time_m",
        )
    }
    face = _exact_str(object.__getattribute__(result, "face"), "face")
    try:
        expected = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=float_inputs["inner_radius_over_mass"],
            outer_radius_over_mass=float_inputs["outer_radius_over_mass"],
            face=face,
            policy=policy,
            phi_ks_rad=float_inputs["phi_ks_rad"],
            coordinate_time_m=float_inputs["coordinate_time_m"],
        )
    except (TypeError, ValueError, KerrFiniteThicknessAreaError) as error:
        raise KerrFiniteThicknessAreaVerificationError(
            "annulus area replay failed"
        ) from error

    int_fields = (
        "coarse_order",
        "fine_order",
        "point_evaluations",
        "maximum_point_evaluations",
    )
    for name in int_fields:
        actual = _exact_int(object.__getattribute__(result, name), name)
        if actual != object.__getattribute__(expected, name):
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match quadrature replay"
            )
    float_fields = (
        "inner_radius_over_mass",
        "outer_radius_over_mass",
        "phi_ks_rad",
        "coordinate_time_m",
        "coarse_area_over_mass_squared",
        "fine_area_over_mass_squared",
        "proper_area_over_mass_squared",
        "proper_area_m2",
        "estimated_absolute_error_over_mass_squared",
        "estimated_relative_error",
        "convergence_threshold_over_mass_squared",
        "minimum_sampled_density_over_mass_squared",
        "maximum_sampled_density_over_mass_squared",
        "maximum_tangency_residual_over_mass",
    )
    for name in float_fields:
        actual = _exact_finite_float(object.__getattribute__(result, name), name)
        expected_value = object.__getattribute__(expected, name)
        if actual.hex() != expected_value.hex():
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match quadrature replay"
            )
    for name in ("face", "_descriptor_json", "_descriptor_sha256"):
        actual = _exact_str(object.__getattribute__(result, name), name)
        if actual != object.__getattribute__(expected, name):
            raise KerrFiniteThicknessAreaVerificationError(
                f"stored {name} does not match quadrature replay"
            )


__all__ = (
    "IMPLEMENTATION_ID",
    "KerrFiniteThicknessAnnulusArea",
    "KerrFiniteThicknessAreaConvergenceError",
    "KerrFiniteThicknessAreaDensity",
    "KerrFiniteThicknessAreaError",
    "KerrFiniteThicknessAreaQuadraturePolicy",
    "KerrFiniteThicknessAreaVerificationError",
    "MAXIMUM_ABSOLUTE_CONVERGENCE_TOLERANCE_OVER_MASS_SQUARED",
    "MAXIMUM_GAUSS_LEGENDRE_ORDER",
    "MAXIMUM_POINT_EVALUATIONS",
    "MAXIMUM_RELATIVE_CONVERGENCE_TOLERANCE",
    "MINIMUM_GAUSS_LEGENDRE_ORDER",
    "SCIENTIFIC_STATUS",
    "integrate_kerr_finite_thickness_annulus_area",
    "kerr_finite_thickness_area_density",
    "verify_kerr_finite_thickness_annulus_area",
    "verify_kerr_finite_thickness_area_density",
)
