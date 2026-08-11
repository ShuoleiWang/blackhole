"""Stationary Novikov--Thorne surface emission in exact Kerr spacetime.

This module joins two existing analytic reference layers without replacing
either of them:

* :mod:`offline.kerr` owns the exact ingoing Cartesian Kerr--Schild chart and
  Boyer--Lindquist-to-Cartesian vector transformation;
* :mod:`offline.novikov_thorne` owns the stable circular-orbit scalars, ISCO,
  and dimensionless Page--Thorne flux shape.

The result is a local, equatorial, stationary, zero-torque thin-disk surface
emitter.  It does not find ray/surface intersections, integrate photon paths,
model returning radiation or limb darkening, solve an atmosphere, evolve
magnetic fields, or provide GRMHD data.  Emission is an isotropic diluted
blackbody on either disk face with a declared colour-correction factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DecimalException,
    ROUND_HALF_EVEN,
    localcontext,
)
import math
import sys
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from offline.geodesic import (
    HamiltonianState,
    hamiltonian_null_residual,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_vector_to_ks_cartesian,
    kerr_constants_of_motion,
    kerr_oblate_event_to_ks_cartesian,
    stationary_axisymmetric_constants,
)
from offline.novikov_thorne import (
    Orientation,
    PROGRADE,
    RETROGRADE,
    circular_orbit_scalars,
    kerr_isco_radius_m,
    page_thorne_flux_shape,
)
from offline.spacetime import Vector4, bilinear, matrix_vector


# SI values used by the local surface calibration.  c, h and k_B are exact in
# the post-2019 SI; G is the CODATA conventional value used by this module.
GRAVITATIONAL_CONSTANT_M3_KG_S2: Final = 6.67430e-11
LIGHT_SPEED_M_S: Final = 299_792_458.0
PLANCK_CONSTANT_J_S: Final = 6.62607015e-34
BOLTZMANN_CONSTANT_J_K: Final = 1.380649e-23
COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID: Final = (
    "kerr-disk-colour-corrected-planck-binary64/v2"
)
STEFAN_BOLTZMANN_W_M2_K4: Final = (
    2.0
    * math.pi**5
    * BOLTZMANN_CONSTANT_J_K**4
    / (
        15.0
        * PLANCK_CONSTANT_J_S**3
        * LIGHT_SPEED_M_S**2
    )
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "stationary analytic Novikov-Thorne equatorial thin-disk surface emitter"
        ),
        "spacetime": "exact stationary Kerr in ingoing Cartesian Kerr-Schild coordinates",
        "emissionModel": "isotropic colour-corrected diluted blackbody from one disk face",
        "planckImplementationId": COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isTimeDependent": False,
        "includesReturningRadiation": False,
        "includesPhotonPathIntersection": False,
        "includesLimbDarkening": False,
        "includesPolarization": False,
        "prohibitedClaim": (
            "Do not describe this stationary zero-torque surface prescription as "
            "GRMHD, a radiating atmosphere, returning-radiation transport, or a "
            "time-dependent accretion flow."
        ),
    }
)

_MAXIMUM_FLOAT_LOG: Final = math.log(sys.float_info.max)
_MINIMUM_SUBNORMAL_LOG: Final = math.log(math.ulp(0.0))
_HALF_MINIMUM_SUBNORMAL_LOG: Final = (
    _MINIMUM_SUBNORMAL_LOG - math.log(2.0)
)
# A binary64 sum of several logarithms is not an interval proof.  Values this
# close to overflow or half-minimum-subnormal rounding therefore take a rare
# exact-float Decimal path.  2^-30 is deliberately much wider than the local
# accumulation of binary64 log ulps, while remaining negligible in practice.
_PLANCK_LOG_BOUNDARY_GUARD: Final = 2.0**-30
_PLANCK_DECIMAL_PRECISION: Final = 120
_PLANCK_DECIMAL_EMAX: Final = 999_999
_PLANCK_DECIMAL_EMIN: Final = -999_999
_FOUR_VELOCITY_TOLERANCE: Final = 4.0e-10
_ORBIT_INVARIANT_TOLERANCE: Final = 8.0e-10


class KerrDiskError(RuntimeError):
    """Raised when a nominally valid disk calculation cannot remain physical."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _decimal_expm1_positive(value: Decimal) -> Decimal:
    """Return ``exp(value)-1`` without cancellation for positive Decimal x."""

    if not value.is_finite() or value <= 0:
        raise KerrDiskError("high-precision Planck exponent is invalid")
    if value >= Decimal("0.5"):
        return value.exp() - Decimal(1)
    term = value
    total = value
    for divisor in range(2, 10_000):
        term = term * value / Decimal(divisor)
        updated = total + term
        if updated == total:
            return total
        total = updated
    raise KerrDiskError("high-precision Planck expm1 did not converge")


def _decimal_colour_corrected_planck_specific_intensity_nu(
    effective_temperature_k: float,
    colour_correction: float,
    emitted_frequency_hz: float,
) -> float:
    """Resolve a rare binary64 range boundary from exact float inputs."""

    isolated_context = Context(
        prec=_PLANCK_DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=_PLANCK_DECIMAL_EMIN,
        Emax=_PLANCK_DECIMAL_EMAX,
        capitals=1,
        clamp=0,
    )
    for signal in isolated_context.traps:
        isolated_context.traps[signal] = False
    isolated_context.clear_flags()
    try:
        with localcontext(isolated_context) as context:
            context.clear_flags()
            planck = Decimal.from_float(PLANCK_CONSTANT_J_S)
            boltzmann = Decimal.from_float(BOLTZMANN_CONSTANT_J_K)
            light_speed = Decimal.from_float(LIGHT_SPEED_M_S)
            temperature = Decimal.from_float(effective_temperature_k)
            correction = Decimal.from_float(colour_correction)
            frequency = Decimal.from_float(emitted_frequency_hz)
            exponent = (
                planck
                * frequency
                / (boltzmann * correction * temperature)
            )
            denominator = _decimal_expm1_positive(exponent)
            intensity = (
                Decimal(2)
                * planck
                * frequency**3
                / (light_speed**2 * denominator * correction**4)
            )
            if not intensity.is_finite() or intensity < 0:
                raise KerrDiskError(
                    "high-precision colour-corrected Planck intensity is invalid"
                )
            if intensity > Decimal.from_float(sys.float_info.max):
                raise KerrDiskError(
                    "colour-corrected Planck intensity overflowed binary64"
                )
            rounded = float(intensity)
    except KerrDiskError:
        raise
    except (DecimalException, OverflowError, ValueError) as error:
        raise KerrDiskError(
            "high-precision colour-corrected Planck boundary evaluation failed"
        ) from error
    if not math.isfinite(rounded) or rounded < 0.0:
        raise KerrDiskError(
            "colour-corrected Planck intensity is invalid after rounding"
        )
    return rounded


def _validated_colour_corrected_planck_specific_intensity_nu(
    effective_temperature_k: float,
    colour_correction: float,
    emitted_frequency_hz: float,
) -> float:
    """Fast Planck kernel for already validated exact binary64 values."""

    effective_temperature = effective_temperature_k
    correction = colour_correction
    frequency = emitted_frequency_hz
    if effective_temperature == 0.0:
        return 0.0

    colour_temperature = correction * effective_temperature
    if not math.isfinite(colour_temperature) or colour_temperature <= 0.0:
        raise KerrDiskError(
            "colour temperature overflowed or left the positive finite domain"
        )
    log_exponent = (
        math.log(PLANCK_CONSTANT_J_S)
        + math.log(frequency)
        - math.log(BOLTZMANN_CONSTANT_J_K)
        - math.log(colour_temperature)
    )
    if log_exponent > _MAXIMUM_FLOAT_LOG:
        return 0.0
    if log_exponent < _MINIMUM_SUBNORMAL_LOG:
        # Deep Rayleigh--Jeans: retain log(x) even when x itself is below the
        # minimum binary64 subnormal.
        log_denominator = log_exponent
    else:
        exponent = math.exp(log_exponent)
        log_denominator = (
            exponent if exponent > 50.0 else math.log(math.expm1(exponent))
        )
    log_intensity = (
        math.log(2.0 * PLANCK_CONSTANT_J_S)
        - 2.0 * math.log(LIGHT_SPEED_M_S)
        + 3.0 * math.log(frequency)
        - log_denominator
        - 4.0 * math.log(correction)
    )
    if not math.isfinite(log_intensity):
        raise KerrDiskError("colour-corrected Planck log intensity is invalid")
    if log_intensity > _MAXIMUM_FLOAT_LOG + _PLANCK_LOG_BOUNDARY_GUARD:
        raise KerrDiskError(
            "colour-corrected Planck intensity overflowed binary64"
        )
    if log_intensity >= _MAXIMUM_FLOAT_LOG - _PLANCK_LOG_BOUNDARY_GUARD:
        return _decimal_colour_corrected_planck_specific_intensity_nu(
            effective_temperature,
            correction,
            frequency,
        )
    if (
        log_intensity
        < _HALF_MINIMUM_SUBNORMAL_LOG - _PLANCK_LOG_BOUNDARY_GUARD
    ):
        return 0.0
    if (
        log_intensity
        <= _HALF_MINIMUM_SUBNORMAL_LOG + _PLANCK_LOG_BOUNDARY_GUARD
    ):
        return _decimal_colour_corrected_planck_specific_intensity_nu(
            effective_temperature,
            correction,
            frequency,
        )
    try:
        intensity = math.exp(log_intensity)
    except OverflowError as error:
        raise KerrDiskError(
            "colour-corrected Planck intensity overflowed binary64"
        ) from error
    if not math.isfinite(intensity) or intensity < 0.0:
        raise KerrDiskError("colour-corrected Planck intensity is invalid")
    return intensity


def colour_corrected_planck_specific_intensity_nu(
    effective_temperature_k: float,
    colour_correction: float,
    emitted_frequency_hz: float,
) -> float:
    """Return local diluted-blackbody ``I_nu`` in SI specific-intensity units.

    The result is ``B_nu(f_col T_eff) / f_col**4`` in
    ``W m^-2 sr^-1 Hz^-1``.  The common path uses stable binary64 log-space
    branches.  Only values near binary64 overflow or the half-minimum-
    subnormal rounding boundary use an exact-input Decimal calculation.
    """

    effective_temperature = _finite_number(
        effective_temperature_k,
        "effective_temperature_k",
    )
    correction = _finite_number(colour_correction, "colour_correction")
    frequency = _finite_number(emitted_frequency_hz, "emitted_frequency_hz")
    if effective_temperature < 0.0:
        raise ValueError("effective_temperature_k must be non-negative")
    if correction < 1.0:
        raise ValueError("colour_correction must be at least one")
    if frequency <= 0.0:
        raise ValueError("emitted_frequency_hz must be positive")
    return _validated_colour_corrected_planck_specific_intensity_nu(
        effective_temperature,
        correction,
        frequency,
    )


def _finite_vector4(value: Sequence[float], label: str) -> Vector4:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain four finite numbers")
    try:
        entries = tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must contain four finite numbers") from error
    if len(entries) != 4:
        raise ValueError(f"{label} must contain four finite numbers")
    return tuple(  # type: ignore[return-value]
        _finite_number(entry, f"{label}[{index}]")
        for index, entry in enumerate(entries)
    )


def _relative_close(
    first: float,
    second: float,
    tolerance: float,
    scale_floor: float = 1.0,
) -> bool:
    return abs(first - second) <= tolerance * max(
        scale_floor,
        abs(first),
        abs(second),
    )


@dataclass(frozen=True, slots=True)
class KerrDiskEmitter:
    """One stable equatorial circular emitter in Cartesian KS components."""

    event: Vector4
    four_velocity: Vector4
    kerr_mass_m: float
    kerr_spin_a_m: float
    radius_m: float
    radius_over_mass: float
    phi_ks_rad: float
    orientation: Orientation
    angular_velocity_inverse_m: float
    specific_energy: float
    specific_angular_momentum_m: float

    def __post_init__(self) -> None:
        event = _finite_vector4(self.event, "KerrDiskEmitter.event")
        velocity = _finite_vector4(
            self.four_velocity,
            "KerrDiskEmitter.four_velocity",
        )
        scalars = (
            self.kerr_mass_m,
            self.kerr_spin_a_m,
            self.radius_m,
            self.radius_over_mass,
            self.phi_ks_rad,
            self.angular_velocity_inverse_m,
            self.specific_energy,
            self.specific_angular_momentum_m,
        )
        normalized = tuple(
            _finite_number(value, f"KerrDiskEmitter scalar {index}")
            for index, value in enumerate(scalars)
        )
        if normalized[0] <= 0.0 or abs(normalized[1]) > normalized[0]:
            raise ValueError("KerrDiskEmitter Kerr parameters are invalid")
        if normalized[2] <= 0.0 or normalized[3] <= 0.0:
            raise ValueError("KerrDiskEmitter radii must be positive")
        if self.orientation not in (PROGRADE, RETROGRADE):
            raise ValueError("KerrDiskEmitter orientation is invalid")
        expected_radius_ratio = normalized[2] / normalized[0]
        if not _relative_close(
            normalized[3],
            expected_radius_ratio,
            2.0e-13,
        ):
            raise ValueError("KerrDiskEmitter radius scales are inconsistent")
        orbit = circular_orbit_scalars(
            expected_radius_ratio,
            abs(normalized[1] / normalized[0]),
            self.orientation,
        )
        spin_axis_sign = -1.0 if normalized[1] < 0.0 else 1.0
        expected_angular_velocity = (
            spin_axis_sign * orbit.omega_m / normalized[0]
        )
        expected_angular_momentum = (
            spin_axis_sign
            * orbit.specific_angular_momentum_m
            * normalized[0]
        )
        for actual, expected, label, scale_floor in (
            (
                normalized[5],
                expected_angular_velocity,
                "angular velocity",
                1.0 / normalized[0],
            ),
            (
                normalized[6],
                orbit.specific_energy,
                "specific energy",
                1.0,
            ),
            (
                normalized[7],
                expected_angular_momentum,
                "specific angular momentum",
                normalized[0],
            ),
        ):
            if not _relative_close(
                actual,
                expected,
                _ORBIT_INVARIANT_TOLERANCE,
                scale_floor,
            ):
                raise ValueError(f"KerrDiskEmitter {label} is inconsistent")
        expected_event = kerr_oblate_event_to_ks_cartesian(
            coordinate_time_m=event[0],
            radius_m=normalized[2],
            theta_rad=0.5 * math.pi,
            phi_ks_rad=normalized[4],
            spin_a_m=normalized[1],
        )
        if any(
            not _relative_close(
                actual,
                expected,
                2.0e-12,
                normalized[0],
            )
            for actual, expected in zip(event, expected_event)
        ):
            raise ValueError("KerrDiskEmitter event is inconsistent")
        redshift_factor = (
            orbit.specific_energy
            - orbit.omega_m * orbit.specific_angular_momentum_m
        )
        if not math.isfinite(redshift_factor) or redshift_factor <= 0.0:
            raise ValueError("KerrDiskEmitter circular redshift is invalid")
        expected_u_t = 1.0 / redshift_factor
        expected_velocity = kerr_bl_vector_to_ks_cartesian(
            (
                expected_u_t,
                0.0,
                0.0,
                expected_angular_velocity * expected_u_t,
            ),
            mass_m=normalized[0],
            spin_a_m=normalized[1],
            radius_m=normalized[2],
            theta_rad=0.5 * math.pi,
            phi_ks_rad=normalized[4],
        )
        if any(
            not _relative_close(actual, expected, 2.0e-12, 1.0e-14)
            for actual, expected in zip(velocity, expected_velocity)
        ):
            raise ValueError("KerrDiskEmitter four-velocity is inconsistent")
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "four_velocity", velocity)
        for name, value in zip(
            (
                "radius_m",
                "radius_over_mass",
                "phi_ks_rad",
                "angular_velocity_inverse_m",
                "specific_energy",
                "specific_angular_momentum_m",
            ),
            normalized[2:],
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "kerr_mass_m", normalized[0])
        object.__setattr__(self, "kerr_spin_a_m", normalized[1])


@dataclass(frozen=True, slots=True)
class KerrDiskThermalState:
    """One-face Page--Thorne surface flux and diluted-blackbody temperatures."""

    radius_m: float
    radius_over_mass: float
    page_thorne_flux_shape: float
    surface_flux_w_m2: float
    effective_temperature_k: float
    colour_temperature_k: float
    colour_correction: float

    def __post_init__(self) -> None:
        names = (
            "radius_m",
            "radius_over_mass",
            "page_thorne_flux_shape",
            "surface_flux_w_m2",
            "effective_temperature_k",
            "colour_temperature_k",
            "colour_correction",
        )
        values = tuple(
            _finite_number(getattr(self, name), f"KerrDiskThermalState.{name}")
            for name in names
        )
        if values[0] <= 0.0 or values[1] <= 0.0:
            raise ValueError("KerrDiskThermalState radii must be positive")
        if any(value < 0.0 for value in values[2:6]):
            raise ValueError("KerrDiskThermalState physical outputs must be non-negative")
        if values[6] < 1.0:
            raise ValueError("KerrDiskThermalState colour correction must be at least one")
        if not _relative_close(
            values[5],
            values[6] * values[4],
            8.0e-13,
        ):
            raise ValueError("KerrDiskThermalState colour temperature is inconsistent")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class StationaryNovikovThorneDisk:
    """Local stationary thin-disk surface model bound to one exact Kerr metric.

    ``metric.mass_m`` sets the coordinate length scale used by geodesics.
    ``black_hole_mass_kg`` separately binds that normalized geometric model to
    SI flux and temperature.  ``mass_accretion_rate_kg_s`` is the total rest-mass
    accretion rate; the returned Page--Thorne flux is from one disk face.
    """

    metric: KerrKerrSchildMetric
    black_hole_mass_kg: float
    mass_accretion_rate_kg_s: float
    orientation: Orientation = PROGRADE
    colour_correction: float = 1.7

    def __post_init__(self) -> None:
        if not isinstance(self.metric, KerrKerrSchildMetric):
            raise TypeError("metric must be an exact KerrKerrSchildMetric")
        mass = _finite_number(self.black_hole_mass_kg, "black_hole_mass_kg")
        accretion = _finite_number(
            self.mass_accretion_rate_kg_s,
            "mass_accretion_rate_kg_s",
        )
        correction = _finite_number(
            self.colour_correction,
            "colour_correction",
        )
        if mass <= 0.0:
            raise ValueError("black_hole_mass_kg must be positive")
        if accretion < 0.0:
            raise ValueError("mass_accretion_rate_kg_s must be non-negative")
        if correction < 1.0:
            raise ValueError("colour_correction must be at least one")

        # This delegates orientation and the strictly sub-extremal disk domain
        # to the scalar oracle rather than duplicating its branch policy here.
        kerr_isco_radius_m(
            abs(self.metric.dimensionless_spin),
            self.orientation,
        )
        object.__setattr__(self, "black_hole_mass_kg", mass)
        object.__setattr__(self, "mass_accretion_rate_kg_s", accretion)
        object.__setattr__(self, "colour_correction", correction)

    @property
    def dimensionless_spin_magnitude(self) -> float:
        return abs(self.metric.dimensionless_spin)

    @property
    def isco_radius_m(self) -> float:
        """Return the ISCO coordinate radius in the metric's length units."""

        return self.metric.mass_m * kerr_isco_radius_m(
            self.dimensionless_spin_magnitude,
            self.orientation,
        )

    def _radius_over_mass(self, radius_m: Any) -> float:
        radius = _finite_number(radius_m, "radius_m")
        if radius <= 0.0:
            raise ValueError("radius_m must be positive")
        radius_over_mass = radius / self.metric.mass_m
        if not math.isfinite(radius_over_mass) or radius_over_mass <= 0.0:
            raise ValueError("radius_m / metric.mass_m must be finite and positive")
        return radius_over_mass

    def emitter(
        self,
        radius_m: float,
        *,
        phi_ks_rad: float = 0.0,
        coordinate_time_m: float = 0.0,
    ) -> KerrDiskEmitter:
        """Return the future-timelike circular emitter at one surface event."""

        radius_over_mass = self._radius_over_mass(radius_m)
        radius = radius_over_mass * self.metric.mass_m
        phi = _finite_number(phi_ks_rad, "phi_ks_rad")
        time = _finite_number(coordinate_time_m, "coordinate_time_m")
        orbit = circular_orbit_scalars(
            radius_over_mass,
            self.dimensionless_spin_magnitude,
            self.orientation,
        )

        # The scalar oracle orients prograde/retrograde around a positive spin
        # axis.  A negative signed Kerr parameter reverses that axis in the
        # repository's fixed +z Cartesian chart.  Schwarzschild retains +z as
        # the coordinate convention for the otherwise degenerate orientation.
        spin_axis_sign = -1.0 if self.metric.spin_a_m < 0.0 else 1.0
        angular_velocity = (
            spin_axis_sign * orbit.omega_m / self.metric.mass_m
        )
        angular_momentum = (
            spin_axis_sign
            * orbit.specific_angular_momentum_m
            * self.metric.mass_m
        )
        redshift_factor = (
            orbit.specific_energy
            - orbit.omega_m * orbit.specific_angular_momentum_m
        )
        if not math.isfinite(redshift_factor) or redshift_factor <= 0.0:
            raise KerrDiskError("circular emitter redshift factor is invalid")
        u_t_contravariant = 1.0 / redshift_factor
        u_phi_contravariant = angular_velocity * u_t_contravariant

        # Reuse the Kerr module's exact coordinate map and
        # BL-to-ingoing-Cartesian-KS Jacobian.  No camera tetrad is involved in
        # the material four-velocity.
        event = kerr_oblate_event_to_ks_cartesian(
            coordinate_time_m=time,
            radius_m=radius,
            theta_rad=0.5 * math.pi,
            phi_ks_rad=phi,
            spin_a_m=self.metric.spin_a_m,
        )
        four_velocity = kerr_bl_vector_to_ks_cartesian(
            (u_t_contravariant, 0.0, 0.0, u_phi_contravariant),
            mass_m=self.metric.mass_m,
            spin_a_m=self.metric.spin_a_m,
            radius_m=radius,
            theta_rad=0.5 * math.pi,
            phi_ks_rad=phi,
        )
        sample = self.metric.sample(event)
        norm = bilinear(four_velocity, sample.covariant, four_velocity)
        if (
            four_velocity[0] <= 0.0
            or not math.isfinite(norm)
            or not _relative_close(norm, -1.0, _FOUR_VELOCITY_TOLERANCE)
        ):
            raise KerrDiskError("circular emitter four-velocity is not future unit-timelike")

        covector = matrix_vector(sample.covariant, four_velocity)
        energy, recovered_angular_momentum = stationary_axisymmetric_constants(
            HamiltonianState(event=event, covector=covector)
        )
        if not _relative_close(
            energy,
            orbit.specific_energy,
            _ORBIT_INVARIANT_TOLERANCE,
        ):
            raise KerrDiskError("Cartesian emitter does not preserve circular-orbit energy")
        if not _relative_close(
            recovered_angular_momentum,
            angular_momentum,
            _ORBIT_INVARIANT_TOLERANCE,
            self.metric.mass_m,
        ):
            raise KerrDiskError(
                "Cartesian emitter does not preserve circular-orbit angular momentum"
            )

        return KerrDiskEmitter(
            event=event,
            four_velocity=four_velocity,
            kerr_mass_m=self.metric.mass_m,
            kerr_spin_a_m=self.metric.spin_a_m,
            radius_m=radius,
            radius_over_mass=radius_over_mass,
            phi_ks_rad=phi,
            orientation=self.orientation,
            angular_velocity_inverse_m=angular_velocity,
            specific_energy=orbit.specific_energy,
            specific_angular_momentum_m=angular_momentum,
        )

    def thermal_state(self, radius_m: float) -> KerrDiskThermalState:
        """Convert the one-face Page--Thorne flux to SI effective temperature."""

        radius_over_mass = self._radius_over_mass(radius_m)
        radius = radius_over_mass * self.metric.mass_m
        flux_shape = page_thorne_flux_shape(
            radius_over_mass,
            self.dimensionless_spin_magnitude,
            self.orientation,
        )
        if not math.isfinite(flux_shape) or flux_shape < 0.0:
            raise KerrDiskError("Page-Thorne flux shape is invalid")

        if flux_shape == 0.0 or self.mass_accretion_rate_kg_s == 0.0:
            surface_flux = 0.0
            effective_temperature = 0.0
        else:
            # With f_PT = 4*pi*M_geo^2*F_geo/dot(M)_geo,
            # F_SI = c^6 dot(M)_SI f_PT / (4*pi*G^2 M_SI^2).
            log_surface_flux = (
                6.0 * math.log(LIGHT_SPEED_M_S)
                + math.log(self.mass_accretion_rate_kg_s)
                + math.log(flux_shape)
                - math.log(4.0 * math.pi)
                - 2.0 * math.log(GRAVITATIONAL_CONSTANT_M3_KG_S2)
                - 2.0 * math.log(self.black_hole_mass_kg)
            )
            if log_surface_flux > _MAXIMUM_FLOAT_LOG:
                raise KerrDiskError("SI Page-Thorne surface flux overflowed binary64")
            if log_surface_flux < _MINIMUM_SUBNORMAL_LOG:
                raise KerrDiskError("SI Page-Thorne surface flux underflowed binary64")
            surface_flux = math.exp(log_surface_flux)
            effective_temperature = math.exp(
                0.25
                * (
                    log_surface_flux
                    - math.log(STEFAN_BOLTZMANN_W_M2_K4)
                )
            )
            if (
                not math.isfinite(surface_flux)
                or surface_flux <= 0.0
                or not math.isfinite(effective_temperature)
                or effective_temperature <= 0.0
            ):
                raise KerrDiskError("SI disk flux or effective temperature is invalid")

        colour_temperature = self.colour_correction * effective_temperature
        if not math.isfinite(colour_temperature) or colour_temperature < 0.0:
            raise KerrDiskError("disk colour temperature is invalid")
        return KerrDiskThermalState(
            radius_m=radius,
            radius_over_mass=radius_over_mass,
            page_thorne_flux_shape=flux_shape,
            surface_flux_w_m2=surface_flux,
            effective_temperature_k=effective_temperature,
            colour_temperature_k=colour_temperature,
            colour_correction=self.colour_correction,
        )

    def emitted_specific_intensity_nu(
        self,
        radius_m: float,
        emitted_frequency_hz: float,
    ) -> float:
        """Return local emitted ``I_nu`` in ``W m^-2 sr^-1 Hz^-1``.

        The intensity is ``B_nu(f_col T_eff) / f_col^4`` and is isotropic over
        the outward hemisphere of one disk face.  No transfer factor ``g^3`` is
        applied here; callers must evaluate this function at the emitter-frame
        frequency belonging to their independently traced photon.
        """

        frequency = _finite_number(emitted_frequency_hz, "emitted_frequency_hz")
        if frequency <= 0.0:
            raise ValueError("emitted_frequency_hz must be positive")
        thermal = self.thermal_state(radius_m)
        return colour_corrected_planck_specific_intensity_nu(
            thermal.effective_temperature_k,
            self.colour_correction,
            frequency,
        )


def _validated_future_unit_velocity(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
    four_velocity: Sequence[float],
    label: str,
) -> Vector4:
    velocity = _finite_vector4(four_velocity, label)
    norm = bilinear(
        velocity,
        metric.sample(state.event).covariant,
        velocity,
    )
    if velocity[0] <= 0.0 or not _relative_close(
        norm,
        -1.0,
        _FOUR_VELOCITY_TOLERANCE,
    ):
        raise ValueError(f"{label} must be future unit-timelike")
    return velocity


def observer_to_emitter_frequency_shift_g(
    metric: KerrKerrSchildMetric,
    observer_state: HamiltonianState,
    observer_four_velocity: Sequence[float],
    emitter_state: HamiltonianState,
    emitter: KerrDiskEmitter,
    *,
    null_residual_limit: float = 1.0e-7,
    conserved_quantity_tolerance: float = 1.0e-7,
    emitter_event_tolerance_m: float | None = None,
) -> float:
    """Return ``g = nu_observer / nu_emitter`` for one past-directed ray.

    Both Hamiltonian states must belong to the same traced photon and use the
    repository's past-directed covector convention.  The function validates
    the null residual plus stationary ``E``, axial ``L_z``, and Carter ``K``
    conservation.  It consumes an already identified emitter event; it
    performs no disk crossing or geodesic integration.
    """

    if not isinstance(metric, KerrKerrSchildMetric):
        raise TypeError("metric must be an exact KerrKerrSchildMetric")
    if not isinstance(observer_state, HamiltonianState):
        raise TypeError("observer_state must be a HamiltonianState")
    if not isinstance(emitter_state, HamiltonianState):
        raise TypeError("emitter_state must be a HamiltonianState")
    if not isinstance(emitter, KerrDiskEmitter):
        raise TypeError("emitter must be a KerrDiskEmitter")
    if (
        not _relative_close(emitter.kerr_mass_m, metric.mass_m, 2.0e-13)
        or not _relative_close(
            emitter.kerr_spin_a_m,
            metric.spin_a_m,
            2.0e-13,
            metric.mass_m,
        )
    ):
        raise ValueError("emitter was constructed for a different Kerr metric")
    residual_limit = _finite_number(null_residual_limit, "null_residual_limit")
    constant_tolerance = _finite_number(
        conserved_quantity_tolerance,
        "conserved_quantity_tolerance",
    )
    if residual_limit <= 0.0 or constant_tolerance <= 0.0:
        raise ValueError("frequency-shift tolerances must be positive")
    event_tolerance = (
        1.0e-8 * metric.mass_m
        if emitter_event_tolerance_m is None
        else _finite_number(
            emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
    )
    if event_tolerance < 0.0:
        raise ValueError("emitter_event_tolerance_m must be non-negative")
    if any(
        abs(emitter_state.event[index] - emitter.event[index]) > event_tolerance
        for index in range(4)
    ):
        raise ValueError("emitter state does not match the declared disk event")

    momentum_scale = max(
        *(abs(value) for value in observer_state.covector),
        *(abs(value) for value in emitter_state.covector),
    )
    if not math.isfinite(momentum_scale) or momentum_scale <= 0.0:
        raise ValueError("photon covectors must have a finite common scale")
    normalized_observer_state = HamiltonianState(
        observer_state.event,
        tuple(value / momentum_scale for value in observer_state.covector),
    )
    normalized_emitter_state = HamiltonianState(
        emitter_state.event,
        tuple(value / momentum_scale for value in emitter_state.covector),
    )

    observer_velocity = _validated_future_unit_velocity(
        metric,
        observer_state,
        observer_four_velocity,
        "observer_four_velocity",
    )
    emitter_velocity = _validated_future_unit_velocity(
        metric,
        emitter_state,
        emitter.four_velocity,
        "emitter.four_velocity",
    )
    for label, state in (
        ("observer", normalized_observer_state),
        ("emitter", normalized_emitter_state),
    ):
        residual = hamiltonian_null_residual(metric, state)
        if not math.isfinite(residual) or residual > residual_limit:
            raise ValueError(
                f"{label} photon covector exceeds the declared null residual limit"
            )

    observer_energy, observer_angular_momentum = stationary_axisymmetric_constants(
        normalized_observer_state
    )
    emitter_energy, emitter_angular_momentum = stationary_axisymmetric_constants(
        normalized_emitter_state
    )
    if not _relative_close(
        observer_energy,
        emitter_energy,
        constant_tolerance,
    ):
        raise ValueError("photon energy is not conserved between observer and emitter")
    if not _relative_close(
        observer_angular_momentum,
        emitter_angular_momentum,
        constant_tolerance,
        metric.mass_m,
    ):
        raise ValueError(
            "photon axial angular momentum is not conserved between observer and emitter"
        )
    observer_constants = kerr_constants_of_motion(metric, normalized_observer_state)
    emitter_constants = kerr_constants_of_motion(metric, normalized_emitter_state)
    if not _relative_close(
        observer_constants.carter_k,
        emitter_constants.carter_k,
        constant_tolerance,
        metric.mass_m * metric.mass_m,
    ):
        raise ValueError(
            "photon Carter constant is not conserved between observer and emitter"
        )

    observer_frequency = math.fsum(
        observer_velocity[index] * normalized_observer_state.covector[index]
        for index in range(4)
    )
    emitter_frequency = math.fsum(
        emitter_velocity[index] * normalized_emitter_state.covector[index]
        for index in range(4)
    )
    if not math.isfinite(observer_frequency) or observer_frequency <= 0.0:
        raise ValueError("observer photon frequency must be positive for a past-directed ray")
    if not math.isfinite(emitter_frequency) or emitter_frequency <= 0.0:
        raise ValueError("emitter photon frequency must be positive for a past-directed ray")
    shift = observer_frequency / emitter_frequency
    if not math.isfinite(shift) or shift <= 0.0:
        raise KerrDiskError("observer/emitter frequency shift is invalid")
    return shift


__all__ = (
    "BOLTZMANN_CONSTANT_J_K",
    "COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID",
    "GRAVITATIONAL_CONSTANT_M3_KG_S2",
    "KerrDiskEmitter",
    "KerrDiskError",
    "KerrDiskThermalState",
    "LIGHT_SPEED_M_S",
    "PLANCK_CONSTANT_J_S",
    "SCIENTIFIC_STATUS",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "StationaryNovikovThorneDisk",
    "colour_corrected_planck_specific_intensity_nu",
    "observer_to_emitter_frequency_shift_g",
)
