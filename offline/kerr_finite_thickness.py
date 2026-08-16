"""Literature-bound stationary Kerr finite-thickness calibration geometry.

This module implements only equations (6)--(7) of Zhou et al. (2020),
``arXiv:2004.12589v2``, following the fiducial prescription of Taylor &
Reynolds (2018), ``arXiv:1712.05418v2``.  In units ``G = c = M = 1``,

``H(rho) = (3 / (2 eta)) dot_m [1 - sqrt(r_ISCO / rho)]``

and the adopted photosphere is ``z(rho) = 2 H(rho)``.  Here
``eta = 1 - E_ISCO`` and ``dot_m`` is the paper's dimensionless parameter
``dot(M) / dot(M)_Edd``.

The cited papers do not give a physical definition of ``dot(M)_Edd`` in this
height equation.  Consequently this module accepts the dimensionless ratio
directly and intentionally provides no conversion from kg/s.  Guessing a
conversion would silently change the height normalization.

The prescription is a stationary analytic calibration surface.  Taylor &
Reynolds explicitly describe its profile as a strictly Newtonian fiducial
model and the choice ``z = 2 H`` as an assumed photosphere.  This module is
therefore not a hydrostatic vertical-structure solution, an atmosphere, GRMHD,
or returning-radiation transport.  It is independent of the exact-Kerr
zero-thickness v2 frame/product path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping

from offline.novikov_thorne import (
    Orientation,
    PROGRADE,
    RETROGRADE,
    circular_orbit_scalars,
    kerr_isco_radius_m,
)


PhotosphereFace = Literal["upper", "lower"]

UPPER: Final = "upper"
LOWER: Final = "lower"
VALID_FACES: Final = frozenset((UPPER, LOWER))

MODEL_IMPLEMENTATION_ID: Final = (
    "zhou-2020-taylor-reynolds-stationary-finite-thickness/v1"
)
PRIMARY_SOURCE_URL: Final = "https://arxiv.org/abs/2004.12589"
HEIGHT_SOURCE_URL: Final = "https://arxiv.org/abs/1712.05418"

# Zhou et al. tabulate dot(M)/dot(M)_Edd from 0 to 0.3 and explicitly show
# a*=0.998.  Values beyond those literature-calibration bounds are rejected.
MAXIMUM_CALIBRATED_EDDINGTON_RATIO: Final = 0.3
MAXIMUM_CALIBRATED_SPIN_MAGNITUDE: Final = 0.998
MAXIMUM_CALIBRATED_OUTER_RADIUS_OVER_MASS: Final = 1.0e6

# This is an implementation policy gate, not a claimed boundary derived by
# Zhou et al. or Taylor & Reynolds.  Callers may tighten it but not loosen it.
# It applies to the pressure scale height H/rho; the assumed photosphere has
# z/rho = 2 H/rho.
MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO: Final = 0.25

# Exact edge-on viewing makes first-visible-face topology especially
# ill-conditioned.  This geometry module does not trace rays; the helper below
# nevertheless prevents a caller from labelling such a setup calibrated.
EDGE_ON_COSINE_NUMERICAL_GUARD: Final = 1.0e-6

EDDINGTON_SCALING_DEFINITION: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "parameter": "dot(M) / dot(M)_Edd",
        "isDimensionless": True,
        "physicalDotMEddDefinitionProvidedBySource": False,
        "supportsKilogramsPerSecondConversion": False,
        "policy": (
            "The caller supplies the paper's dimensionless ratio directly; "
            "no SI Eddington-rate convention is inferred."
        ),
    }
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "stationary analytic Kerr finite-thickness calibration surface"
        ),
        "implementationId": MODEL_IMPLEMENTATION_ID,
        "primarySource": PRIMARY_SOURCE_URL,
        "heightSource": HEIGHT_SOURCE_URL,
        "heightPrescription": (
            "strictly Newtonian fiducial pressure-scale-height profile with "
            "an assumed photosphere at z = 2 H"
        ),
        "efficiencyDefinition": "eta = 1 - E_ISCO (Novikov-Thorne)",
        "eddingtonScaling": EDDINGTON_SCALING_DEFINITION,
        "upperAndLowerFaces": (
            "upper face follows the cited z=2H prescription; lower face is "
            "its explicit equatorial-reflection extension"
        ),
        "isHydrostaticVerticalStructureSolution": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "includesSolvedAtmosphere": False,
        "includesReturningRadiation": False,
        "includesRadialAdvection": False,
        "includesSelfOcclusionRayTracing": False,
        "providesSignedSelfOcclusionGeometry": True,
        "radialBoundary": (
            "photosphere faces exist only for rho in [r_ISCO, r_out]; no "
            "inner or outer vertical sidewall is supplied"
        ),
        "prohibitedClaim": (
            "Do not describe this phenomenological stationary height "
            "prescription as GRMHD, a hydrostatic solution, an atmosphere, "
            "returning radiation, or time-dependent accretion."
        ),
    }
)


class KerrFiniteThicknessError(RuntimeError):
    """Raised when an in-domain calibration calculation is not physical."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _face_sign(face: PhotosphereFace) -> float:
    if not isinstance(face, str) or face not in VALID_FACES:
        raise ValueError("face must be 'upper' or 'lower'")
    return 1.0 if face == UPPER else -1.0


def validate_calibration_observer_inclination_rad(value: float) -> float:
    """Validate the usual observer inclination ``0 <= i < pi/2``.

    ``i`` is the angle between the positive spin axis and the line of sight.
    Exact and numerically near-edge-on configurations fail closed because this
    module provides geometry but no certified first-visible-surface solver.
    The lower hemisphere is represented by reflecting the selected face, not
    by passing inclinations above ``pi/2``.
    """

    inclination = _finite_number(value, "observer_inclination_rad")
    if inclination < 0.0 or inclination > 0.5 * math.pi:
        raise ValueError("observer inclination must lie in [0, pi/2]")
    if math.cos(inclination) <= EDGE_ON_COSINE_NUMERICAL_GUARD:
        raise ValueError(
            "edge-on observer is outside the certified numerical geometry domain"
        )
    return inclination


@dataclass(frozen=True, slots=True)
class BoyerLindquistPhotospherePoint:
    """One point on a face in mass-normalized Boyer--Lindquist coordinates."""

    pseudo_cylindrical_radius_over_mass: float
    signed_height_over_mass: float
    radius_over_mass: float
    theta_rad: float
    face: PhotosphereFace

    def __post_init__(self) -> None:
        rho = _finite_number(
            self.pseudo_cylindrical_radius_over_mass,
            "pseudo_cylindrical_radius_over_mass",
        )
        height = _finite_number(
            self.signed_height_over_mass,
            "signed_height_over_mass",
        )
        radius = _finite_number(self.radius_over_mass, "radius_over_mass")
        theta = _finite_number(self.theta_rad, "theta_rad")
        sign = _face_sign(self.face)
        if rho <= 0.0 or radius <= 0.0:
            raise ValueError("photosphere radii must be positive")
        if theta <= 0.0 or theta >= math.pi:
            raise ValueError("photosphere theta must lie strictly inside (0, pi)")
        if sign * height < 0.0:
            raise ValueError("photosphere height sign is inconsistent with its face")
        if not math.isclose(radius, math.hypot(rho, height), rel_tol=2.0e-14):
            raise ValueError("photosphere BL radius is inconsistent")
        expected_theta = math.atan2(rho, height)
        if not math.isclose(theta, expected_theta, rel_tol=0.0, abs_tol=2.0e-14):
            raise ValueError("photosphere theta is inconsistent")
        object.__setattr__(self, "pseudo_cylindrical_radius_over_mass", rho)
        object.__setattr__(self, "signed_height_over_mass", height)
        object.__setattr__(self, "radius_over_mass", radius)
        object.__setattr__(self, "theta_rad", theta)


@dataclass(frozen=True, slots=True)
class StationaryKerrFiniteThicknessCalibration:
    """Zhou--Taylor--Reynolds height oracle on a bounded Kerr annulus.

    All lengths are divided by the black-hole mass ``M``.  The spin magnitude
    and orbit orientation follow :mod:`offline.novikov_thorne`.  The radial
    domain is the closed pseudo-cylindrical annulus ``[r_ISCO, r_out]``.

    ``thinness_gate_maximum_h_over_rho`` is an explicit caller claim gate for
    the pressure scale height, not for the assumed photosphere.  It may be
    tightened but cannot exceed the module policy maximum of 0.25.
    """

    dimensionless_spin: float
    eddington_scaled_mass_accretion_rate: float
    orientation: Orientation = PROGRADE
    outer_radius_over_mass: float = MAXIMUM_CALIBRATED_OUTER_RADIUS_OVER_MASS
    thinness_gate_maximum_h_over_rho: float = (
        MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO
    )

    def __post_init__(self) -> None:
        spin = _finite_number(self.dimensionless_spin, "dimensionless_spin")
        accretion_ratio = _finite_number(
            self.eddington_scaled_mass_accretion_rate,
            "eddington_scaled_mass_accretion_rate",
        )
        outer_radius = _finite_number(
            self.outer_radius_over_mass,
            "outer_radius_over_mass",
        )
        thinness_gate = _finite_number(
            self.thinness_gate_maximum_h_over_rho,
            "thinness_gate_maximum_h_over_rho",
        )
        if spin < 0.0 or spin > MAXIMUM_CALIBRATED_SPIN_MAGNITUDE:
            raise ValueError(
                "dimensionless_spin must lie in the literature-calibration "
                f"range [0, {MAXIMUM_CALIBRATED_SPIN_MAGNITUDE}]"
            )
        if self.orientation not in (PROGRADE, RETROGRADE):
            raise ValueError("orientation must be 'prograde' or 'retrograde'")
        if (
            accretion_ratio < 0.0
            or accretion_ratio > MAXIMUM_CALIBRATED_EDDINGTON_RATIO
        ):
            raise ValueError(
                "eddington-scaled mass accretion rate must lie in the "
                "literature grid range [0, 0.3]"
            )
        if (
            outer_radius <= 0.0
            or outer_radius > MAXIMUM_CALIBRATED_OUTER_RADIUS_OVER_MASS
        ):
            raise ValueError(
                "outer_radius_over_mass must lie in (0, 1e6]"
            )
        if (
            thinness_gate <= 0.0
            or thinness_gate > MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO
        ):
            raise ValueError(
                "thinness gate must lie in (0, 0.25] and cannot loosen the "
                "calibration policy"
            )

        isco = kerr_isco_radius_m(spin, self.orientation)
        if outer_radius <= isco:
            raise ValueError(
                "outer_radius_over_mass must be strictly outside the ISCO"
            )
        object.__setattr__(self, "dimensionless_spin", spin)
        object.__setattr__(
            self,
            "eddington_scaled_mass_accretion_rate",
            accretion_ratio,
        )
        object.__setattr__(self, "outer_radius_over_mass", outer_radius)
        object.__setattr__(
            self,
            "thinness_gate_maximum_h_over_rho",
            thinness_gate,
        )

        actual_aspect = self.maximum_pressure_scale_height_aspect_ratio
        if actual_aspect > thinness_gate * (1.0 + 32.0 * math.ulp(1.0)):
            raise ValueError(
                "finite-thickness profile violates the declared H/rho "
                f"thinness gate ({actual_aspect:.17g} > {thinness_gate:.17g})"
            )

    @property
    def isco_radius_over_mass(self) -> float:
        """Equatorial Kerr ISCO radius divided by ``M``."""

        return kerr_isco_radius_m(self.dimensionless_spin, self.orientation)

    @property
    def isco_specific_energy(self) -> float:
        """Specific circular-orbit energy at the selected ISCO."""

        return circular_orbit_scalars(
            self.isco_radius_over_mass,
            self.dimensionless_spin,
            self.orientation,
        ).specific_energy

    @property
    def novikov_thorne_radiative_efficiency(self) -> float:
        """Return the literature definition ``eta = 1 - E_ISCO``."""

        efficiency = 1.0 - self.isco_specific_energy
        if not math.isfinite(efficiency) or efficiency <= 0.0:
            raise KerrFiniteThicknessError(
                "Novikov-Thorne ISCO efficiency is not finite and positive"
            )
        return efficiency

    @property
    def asymptotic_pressure_scale_height_over_mass(self) -> float:
        """Return ``lim_(rho->infinity) H/M`` in the cited prescription."""

        return (
            1.5
            * self.eddington_scaled_mass_accretion_rate
            / self.novikov_thorne_radiative_efficiency
        )

    @property
    def asymptotic_photosphere_height_over_mass(self) -> float:
        """Return ``lim_(rho->infinity) z/M = 2 H/M``."""

        return 2.0 * self.asymptotic_pressure_scale_height_over_mass

    def _validated_pseudo_radius(self, value: float) -> float:
        radius = _finite_number(
            value,
            "pseudo_cylindrical_radius_over_mass",
        )
        if (
            radius < self.isco_radius_over_mass
            or radius > self.outer_radius_over_mass
        ):
            raise ValueError(
                "pseudo-cylindrical radius must lie in the closed disk "
                "annulus [r_ISCO, r_out]"
            )
        return radius

    def contains_pseudo_cylindrical_radius(self, value: float) -> bool:
        """Whether a finite radius lies inside the closed photosphere annulus."""

        radius = _finite_number(
            value,
            "pseudo_cylindrical_radius_over_mass",
        )
        return self.isco_radius_over_mass <= radius <= self.outer_radius_over_mass

    def pressure_scale_height_over_mass(self, value: float) -> float:
        """Evaluate Zhou et al. equation (6), ``H/M``."""

        radius = self._validated_pseudo_radius(value)
        # -expm1(log(sqrt(r_ISCO/rho))) retains the linear near-ISCO limit.
        height_factor = -math.expm1(
            0.5 * (math.log(self.isco_radius_over_mass) - math.log(radius))
        )
        height = self.asymptotic_pressure_scale_height_over_mass * height_factor
        if not math.isfinite(height) or height < 0.0:
            raise KerrFiniteThicknessError(
                "pressure scale height is not finite and non-negative"
            )
        return height

    def pressure_scale_height_derivative(self, value: float) -> float:
        """Return ``d(H/M) / d(rho/M)`` inside the closed annulus."""

        radius = self._validated_pseudo_radius(value)
        derivative = (
            0.5
            * self.asymptotic_pressure_scale_height_over_mass
            * math.sqrt(self.isco_radius_over_mass)
            / radius**1.5
        )
        if not math.isfinite(derivative) or derivative < 0.0:
            raise KerrFiniteThicknessError("height derivative is invalid")
        return derivative

    def photosphere_height_over_mass(self, value: float) -> float:
        """Evaluate the assumed upper photosphere height ``z/M = 2 H/M``."""

        return 2.0 * self.pressure_scale_height_over_mass(value)

    def photosphere_height_derivative(self, value: float) -> float:
        """Return ``d(z/M) / d(rho/M) = 2 d(H/M)/d(rho/M)``."""

        return 2.0 * self.pressure_scale_height_derivative(value)

    @property
    def maximum_pressure_scale_height_aspect_radius_over_mass(self) -> float:
        """Radius where ``H/rho`` is maximal on this bounded annulus."""

        unconstrained = 2.25 * self.isco_radius_over_mass
        return min(unconstrained, self.outer_radius_over_mass)

    @property
    def maximum_pressure_scale_height_aspect_ratio(self) -> float:
        """Exact maximum of ``H/rho`` on ``[r_ISCO, r_out]``."""

        radius = self.maximum_pressure_scale_height_aspect_radius_over_mass
        return self.pressure_scale_height_over_mass(radius) / radius

    @property
    def maximum_photosphere_height_aspect_ratio(self) -> float:
        """Exact maximum of ``z/rho = 2 H/rho`` on the annulus."""

        return 2.0 * self.maximum_pressure_scale_height_aspect_ratio

    def photosphere_point(
        self,
        pseudo_cylindrical_radius_over_mass: float,
        face: PhotosphereFace,
    ) -> BoyerLindquistPhotospherePoint:
        """Return one upper/lower photosphere point in BL ``(r, theta)``.

        Zhou et al. define ``rho = r sin(theta)``.  We complete this
        pseudo-cylindrical chart with ``z = r cos(theta)``.  The lower face is
        the explicitly declared equatorial reflection of the cited upper face.
        """

        rho = self._validated_pseudo_radius(
            pseudo_cylindrical_radius_over_mass
        )
        signed_height = _face_sign(face) * self.photosphere_height_over_mass(rho)
        return BoyerLindquistPhotospherePoint(
            pseudo_cylindrical_radius_over_mass=rho,
            signed_height_over_mass=signed_height,
            radius_over_mass=math.hypot(rho, signed_height),
            theta_rad=math.atan2(rho, signed_height),
            face=face,
        )

    def face_signed_surface_over_mass(
        self,
        *,
        radius_over_mass: float,
        theta_rad: float,
        face: PhotosphereFace,
    ) -> float:
        """Return an outward-positive signed face residual in units of ``M``.

        For face sign ``s=+1`` (upper) or ``s=-1`` (lower), the scalar is
        ``S_s = s r cos(theta) - z(r sin(theta))``.  ``S_s=0`` defines that
        face, and positive values lie beyond its outward side.  Evaluation
        outside the radial annulus fails closed; this is not a global SDF and
        does not add an unphysical radial sidewall.
        """

        radius = _finite_number(radius_over_mass, "radius_over_mass")
        theta = _finite_number(theta_rad, "theta_rad")
        sign = _face_sign(face)
        if radius <= 0.0:
            raise ValueError("radius_over_mass must be positive")
        if theta <= 0.0 or theta >= math.pi:
            raise ValueError("theta_rad must lie strictly inside (0, pi)")
        rho = self._validated_pseudo_radius(radius * math.sin(theta))
        return (
            sign * radius * math.cos(theta)
            - self.photosphere_height_over_mass(rho)
        )

    def face_surface_gradient_covector_bl(
        self,
        *,
        radius_over_mass: float,
        theta_rad: float,
        face: PhotosphereFace,
    ) -> tuple[float, float, float, float]:
        """Return the outward BL covector ``dS_s`` in the ``M=1`` chart."""

        radius = _finite_number(radius_over_mass, "radius_over_mass")
        theta = _finite_number(theta_rad, "theta_rad")
        sign = _face_sign(face)
        if radius <= 0.0:
            raise ValueError("radius_over_mass must be positive")
        if theta <= 0.0 or theta >= math.pi:
            raise ValueError("theta_rad must lie strictly inside (0, pi)")
        sine = math.sin(theta)
        cosine = math.cos(theta)
        rho = self._validated_pseudo_radius(radius * sine)
        slope = self.photosphere_height_derivative(rho)
        radial_component = sign * cosine - slope * sine
        polar_component = -sign * radius * sine - slope * radius * cosine
        covector = (0.0, radial_component, polar_component, 0.0)
        if not all(math.isfinite(component) for component in covector):
            raise KerrFiniteThicknessError("surface gradient is not finite")
        return covector

    def unit_face_normal_covector_bl(
        self,
        pseudo_cylindrical_radius_over_mass: float,
        face: PhotosphereFace,
    ) -> tuple[float, float, float, float]:
        """Return the unit spacelike outward face normal covector in BL basis."""

        point = self.photosphere_point(
            pseudo_cylindrical_radius_over_mass,
            face,
        )
        covector = self.face_surface_gradient_covector_bl(
            radius_over_mass=point.radius_over_mass,
            theta_rad=point.theta_rad,
            face=face,
        )
        radius = point.radius_over_mass
        cosine = math.cos(point.theta_rad)
        spin_squared = self.dimensionless_spin**2
        sigma = radius * radius + spin_squared * cosine * cosine
        delta = radius * radius - 2.0 * radius + spin_squared
        if not math.isfinite(delta) or delta <= 0.0 or sigma <= 0.0:
            raise KerrFiniteThicknessError(
                "photosphere normal is not outside the Kerr horizon"
            )
        norm_squared = (
            delta * covector[1] * covector[1]
            + covector[2] * covector[2]
        ) / sigma
        if not math.isfinite(norm_squared) or norm_squared <= 0.0:
            raise KerrFiniteThicknessError(
                "photosphere normal is not finite and spacelike"
            )
        inverse_norm = 1.0 / math.sqrt(norm_squared)
        return tuple(  # type: ignore[return-value]
            component * inverse_norm for component in covector
        )
