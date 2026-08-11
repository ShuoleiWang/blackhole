"""Stationary Novikov--Thorne thin-disk scalar oracle for Kerr spacetime.

The module is deliberately local and radial.  It evaluates equatorial,
geodesic circular-orbit scalars and the zero-torque Page--Thorne surface-flux
shape in exact stationary Kerr spacetime.  It does not trace photons, evolve a
fluid, model magnetic stress, solve vertical structure, or perform GRMHD.

Conventions
-----------

``G = c = M = 1``.  ``dimensionless_spin`` is the non-negative Kerr spin
magnitude ``a/M`` and ``orientation`` is ``"prograde"`` or ``"retrograde"``
relative to the hole's positive spin axis.  Consequently retrograde ``Omega``
and ``L_z`` are negative.  Radii and specific angular momenta are in units of
``M``; returned angular velocity is ``M Omega``.

``page_thorne_flux_shape`` returns ``4 pi M^2 F / dot(M)`` for the flux from
one disk face.  It evaluates the analytic Page--Thorne logarithmic form, with
the zero-stress boundary at the corresponding ISCO.  Its Newtonian far-field
limit is ``3 / (2 r^3)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping


Orientation = Literal["prograde", "retrograde"]

PROGRADE: Final = "prograde"
RETROGRADE: Final = "retrograde"
VALID_ORIENTATIONS: Final = frozenset((PROGRADE, RETROGRADE))

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "stationary analytic Novikov-Thorne thin-disk scalar oracle",
        "spacetime": "exact stationary Kerr in Boyer-Lindquist coordinates",
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "includesPhotonTransfer": False,
        "prohibitedClaim": (
            "Do not describe these local circular-orbit and radial flux scalars "
            "as GRMHD, plasma evolution, photon transfer, or a time-dependent disk."
        ),
    }
)

_NEAR_ISCO_ROOT_INTERVAL: Final = 1.0e-4
_SCHWARZSCHILD_SPIN_TOLERANCE: Final = 8.0 * math.ulp(1.0)

# Positive half of the eight-point Gauss--Legendre rule.  The fixed rule is
# used only for a very narrow interval above the ISCO, where the analytic
# logarithms cancel to second order.  It has bounded cost and preserves the
# analytic formula as the normal path.
_GAUSS_LEGENDRE_NODES: Final = (
    0.9602898564975363,
    0.7966664774136267,
    0.5255324099163290,
    0.1834346424956498,
)
_GAUSS_LEGENDRE_WEIGHTS: Final = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
)


class NovikovThorneError(RuntimeError):
    """Raised when a nominally valid reference calculation is not finite."""


@dataclass(frozen=True)
class CircularOrbitScalars:
    """Dimensionless equatorial circular-orbit invariants in Kerr."""

    radius_m: float
    dimensionless_spin: float
    orientation: Orientation
    omega_m: float
    specific_energy: float
    specific_angular_momentum_m: float


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validated_spin(value: Any) -> float:
    spin = _finite_number(value, "dimensionless_spin")
    if spin < 0.0 or spin >= 1.0:
        raise ValueError("dimensionless_spin must satisfy 0 <= a/M < 1")
    return spin


def _validated_radius(value: Any) -> float:
    radius = _finite_number(value, "radius_m")
    if radius <= 0.0:
        raise ValueError("radius_m must be positive")
    return radius


def _orientation_sign(value: Any) -> int:
    if not isinstance(value, str) or value not in VALID_ORIENTATIONS:
        raise ValueError("orientation must be 'prograde' or 'retrograde'")
    return 1 if value == PROGRADE else -1


def kerr_isco_radius_m(
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> float:
    """Return the equatorial Kerr ISCO radius in units of ``M``.

    This is the Bardeen--Press--Teukolsky algebraic root.  Exactly extremal
    spin is excluded because the prograde circular-orbit expressions become a
    removable but numerically singular ``0/0`` at ``r=M``.
    """

    spin = _validated_spin(dimensionless_spin)
    sign = _orientation_sign(orientation)
    z1 = 1.0 + (1.0 - spin * spin) ** (1.0 / 3.0) * (
        (1.0 + spin) ** (1.0 / 3.0)
        + (1.0 - spin) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * spin * spin + z1 * z1)
    radical = (3.0 - z1) * (3.0 + z1 + 2.0 * z2)
    # Roundoff can produce a tiny negative radical in the Schwarzschild limit.
    if radical < 0.0 and radical > -8.0 * math.ulp(1.0):
        radical = 0.0
    if radical < 0.0:
        raise NovikovThorneError("Kerr ISCO radical became negative")
    radius = 3.0 + z2 - sign * math.sqrt(radical)
    if not math.isfinite(radius) or radius <= 0.0:
        raise NovikovThorneError("Kerr ISCO radius is not finite and positive")
    return radius


def _raw_orbit_scalars(
    radius_m: float,
    spin: float,
    sign: int,
) -> tuple[float, float, float]:
    root_radius = math.sqrt(radius_m)
    radius_three_halves = radius_m * root_radius
    circular_radicand = (
        radius_three_halves - 3.0 * root_radius + 2.0 * sign * spin
    )
    if circular_radicand <= 0.0 or not math.isfinite(circular_radicand):
        raise NovikovThorneError(
            "equatorial timelike circular-orbit denominator is not positive"
        )
    denominator = radius_m ** 0.75 * math.sqrt(circular_radicand)
    omega = sign / (radius_three_halves + sign * spin)
    energy = (
        radius_three_halves - 2.0 * root_radius + sign * spin
    ) / denominator
    angular_momentum = sign * (
        radius_m * radius_m
        - 2.0 * sign * spin * root_radius
        + spin * spin
    ) / denominator
    if not all(
        math.isfinite(value) for value in (omega, energy, angular_momentum)
    ):
        raise NovikovThorneError("circular-orbit scalar is not finite")
    return omega, energy, angular_momentum


def circular_orbit_scalars(
    radius_m: float,
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> CircularOrbitScalars:
    """Return ``M Omega``, specific energy, and ``L_z/M`` on a stable orbit."""

    radius = _validated_radius(radius_m)
    spin = _validated_spin(dimensionless_spin)
    sign = _orientation_sign(orientation)
    isco = kerr_isco_radius_m(spin, orientation)
    if radius < isco:
        raise ValueError(
            f"radius_m must be at or outside the {orientation} ISCO ({isco:.17g} M)"
        )
    omega, energy, angular_momentum = _raw_orbit_scalars(radius, spin, sign)
    if energy <= 0.0 or energy >= 1.0:
        # Stable, finite-radius Kerr circular orbits are bound and future-timelike.
        raise NovikovThorneError("stable circular-orbit energy is outside 0 < E < 1")
    if sign * omega <= 0.0 or sign * angular_momentum <= 0.0:
        raise NovikovThorneError("orbit orientation is inconsistent with Omega or L_z")
    if energy - omega * angular_momentum <= 0.0:
        raise NovikovThorneError("E - Omega L_z must be positive")
    return CircularOrbitScalars(
        radius_m=radius,
        dimensionless_spin=spin,
        orientation=orientation,
        omega_m=omega,
        specific_energy=energy,
        specific_angular_momentum_m=angular_momentum,
    )


def orbital_angular_velocity_m(
    radius_m: float,
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> float:
    """Return the signed dimensionless angular velocity ``M Omega``."""

    return circular_orbit_scalars(
        radius_m,
        dimensionless_spin,
        orientation,
    ).omega_m


def specific_energy(
    radius_m: float,
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> float:
    """Return the conserved circular-orbit energy per unit rest mass."""

    return circular_orbit_scalars(
        radius_m,
        dimensionless_spin,
        orientation,
    ).specific_energy


def specific_angular_momentum_m(
    radius_m: float,
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> float:
    """Return the signed conserved axial angular momentum ``L_z/M``."""

    return circular_orbit_scalars(
        radius_m,
        dimensionless_spin,
        orientation,
    ).specific_angular_momentum_m


def _page_thorne_cubic_roots(signed_spin: float) -> tuple[float, float, float]:
    """Return the three real roots of ``x^3 - 3 x + 2 a``.

    Writing the two roots that coalesce at extremal spin as ``cos(q) +/-
    sqrt(3) sin(q)`` retains their separation when ``|a|`` is one ulp below
    unity.  Evaluating two shifted cosines loses substantially more precision
    in that limit.
    """

    angle = math.acos(signed_spin) / 3.0
    cosine = math.cos(angle)
    sine_term = math.sqrt(3.0) * math.sin(angle)
    positive_root = cosine + sine_term
    negative_root = -2.0 * cosine
    # The remaining root crosses zero with a.  Recovering it from the exact
    # cubic-root product avoids subtracting two nearly equal O(1) terms for a
    # slowly rotating hole.
    middle_root = (
        0.0
        if signed_spin == 0.0
        else -2.0 * signed_spin / (positive_root * negative_root)
    )
    return (
        positive_root,
        middle_root,
        negative_root,
    )


def _schwarzschild_radial_integral(
    root_radius: float,
    root_isco: float,
) -> float:
    delta = root_radius - root_isco
    root_three = math.sqrt(3.0)
    return math.fsum(
        (
            delta,
            -0.5
            * root_three
            * math.log1p(delta / (root_isco - root_three)),
            0.5
            * root_three
            * math.log1p(delta / (root_isco + root_three)),
        )
    )


def _near_isco_radial_integral(
    root_radius: float,
    root_isco: float,
    roots: tuple[float, float, float],
) -> float:
    """Integrate the regular ISCO-local form without log cancellation."""

    width = root_radius - root_isco
    circular_radicand_at_isco = math.prod(root_isco - root for root in roots)
    terms: list[float] = []
    for node, weight in zip(_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS):
        for fraction in (0.5 * (1.0 - node), 0.5 * (1.0 + node)):
            offset = width * fraction
            sample = root_isco + offset
            circular_radicand = math.prod(sample - root for root in roots)
            # The marginal-stability polynomial Q has its ISCO root removed:
            # Q(x0+u) = u [4 C(x0) + 6(x0^2-1)u + 4x0 u^2 + u^3].
            quotient = 4.0 * circular_radicand_at_isco + offset * (
                6.0 * (root_isco * root_isco - 1.0)
                + offset * (4.0 * root_isco + offset)
            )
            terms.append(
                0.5
                * weight
                * offset
                * quotient
                / (sample * circular_radicand)
            )
    return width * math.fsum(terms)


def _page_thorne_radial_integral(
    root_radius: float,
    root_isco: float,
    signed_spin: float,
    roots: tuple[float, float, float],
) -> float:
    """Return the closed-form Page--Thorne conservation integral."""

    delta = root_radius - root_isco
    nearest_pole_distance = min(root_isco - root for root in roots)
    local_interval = min(
        _NEAR_ISCO_ROOT_INTERVAL * max(root_isco, 1.0),
        2.0 * nearest_pole_distance,
    )
    if delta <= local_interval:
        return _near_isco_radial_integral(root_radius, root_isco, roots)
    if abs(signed_spin) <= _SCHWARZSCHILD_SPIN_TOLERANCE:
        # The root at zero merges with the explicit 1/x pole when a=0.  The
        # combined Schwarzschild limit avoids a removable 0/0 and is also the
        # correct floating-point limit for spins below machine resolution.
        return _schwarzschild_radial_integral(root_radius, root_isco)

    terms = [
        delta,
        -1.5 * signed_spin * math.log1p(delta / root_isco),
    ]
    for root in roots:
        denominator = root * (root - 1.0) * (root + 1.0)
        coefficient = -((root - signed_spin) ** 2) / denominator
        terms.append(
            coefficient * math.log1p(delta / (root_isco - root))
        )
    return math.fsum(terms)


def page_thorne_flux_shape(
    radius_m: float,
    dimensionless_spin: float,
    orientation: Orientation = PROGRADE,
) -> float:
    """Return the zero-torque relativistic thin-disk surface-flux shape.

    The returned scalar is ``4 pi M^2 F / dot(M)``.  For ``r <= r_ISCO`` it is
    exactly zero: the oracle declares no emitting circular thin disk inside its
    stress-free inner edge.  Outside the ISCO it evaluates

    ``-Omega_,r / (r (E - Omega L)^2)``
    ``* integral[r_ISCO:r] (E - Omega L) L_,r dr``.

    This is the Page--Thorne conservation-law expression specialized to the
    equatorial Kerr disk.  Its radial integral is evaluated through the
    equivalent three-root logarithmic closed form; only the cancellation-prone
    interval immediately above the ISCO uses a bounded fixed quadrature of the
    regular root-factored integrand.
    """

    radius = _validated_radius(radius_m)
    spin = _validated_spin(dimensionless_spin)
    sign = _orientation_sign(orientation)
    isco = kerr_isco_radius_m(spin, orientation)
    if radius <= isco:
        return 0.0

    root_isco = math.sqrt(isco)
    root_radius = math.sqrt(radius)
    if not root_radius > root_isco:
        return 0.0
    signed_spin = sign * spin
    if abs(signed_spin) <= _SCHWARZSCHILD_SPIN_TOLERANCE:
        root_three = math.sqrt(3.0)
        roots = (root_three, 0.0, -root_three)
    else:
        roots = _page_thorne_cubic_roots(signed_spin)
    integral = _page_thorne_radial_integral(
        root_radius,
        root_isco,
        signed_spin,
        roots,
    )
    if not math.isfinite(integral):
        raise NovikovThorneError("Page-Thorne radial integral is not finite")
    if integral < 0.0:
        if integral < -64.0 * math.ulp(max(root_radius - root_isco, 1.0)):
            raise NovikovThorneError("Page-Thorne radial integral became negative")
        integral = 0.0

    # C(x)/x^3 = product_i(1-x_i/x) is stable both near an almost-extremal
    # prograde ISCO and at enormous radii.  The equivalent r^-3 scaling avoids
    # overflowing x^4 C(x); a physically vanishing far-field flux may underflow
    # cleanly to zero at the largest finite IEEE-754 radii.
    scaled_circular_radicand = math.prod(
        (root_radius - root) / root_radius for root in roots
    )
    if (
        not math.isfinite(scaled_circular_radicand)
        or scaled_circular_radicand <= 0.0
    ):
        raise NovikovThorneError(
            "equatorial circular-orbit denominator is not finite and positive"
        )
    inverse_radius = 1.0 / radius
    flux = (
        1.5
        * (integral / root_radius)
        * (inverse_radius * inverse_radius * inverse_radius)
        / scaled_circular_radicand
    )
    if not math.isfinite(flux):
        raise NovikovThorneError("Page-Thorne flux shape is not finite")
    if flux < 0.0:
        roundoff_scale = abs(flux)
        if flux < -64.0 * math.ulp(max(roundoff_scale, 1.0)):
            raise NovikovThorneError("Page-Thorne flux shape became negative")
        return 0.0
    return flux


__all__ = [
    "CircularOrbitScalars",
    "NovikovThorneError",
    "PROGRADE",
    "RETROGRADE",
    "SCIENTIFIC_STATUS",
    "circular_orbit_scalars",
    "kerr_isco_radius_m",
    "orbital_angular_velocity_m",
    "page_thorne_flux_shape",
    "specific_angular_momentum_m",
    "specific_energy",
]
