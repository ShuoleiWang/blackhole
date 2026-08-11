"""Angular emission laws and local Kerr thin-disk emission angles.

This module is intentionally smaller than a disk-atmosphere solver.  It owns
only the angle between one already traced photon and the local equatorial disk
normal, plus explicitly labelled angular intensity prescriptions.  The
default flux-conserving linear law is the KERRBB D20 approximation
``I(mu) proportional to 1/2 + 3/4 mu``.  A separately declared coefficient
may represent another linear proxy, including the often-used Laor-style
``1 + 2.06 mu`` law.  Neither claims frequency-dependent atmosphere transfer,
polarization, or self-irradiation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, runtime_checkable

from offline.geodesic import HamiltonianState, hamiltonian_null_residual
from offline.kerr import KerrKerrSchildMetric, kerr_constants_of_motion
from offline.kerr_disk import KerrDiskEmitter


SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "local analytic thin-disk angular emission law",
        "normal": (
            "equatorial unit normal evaluated from the separated Kerr Carter "
            "momentum, algebraically equivalent to the projected z=0 gradient"
        ),
        "linearLaw": (
            "flux-conserving linear electron-scattering proxy; default is KERRBB D20"
        ),
        "isAtmosphereSolution": False,
        "isFrequencyDependent": False,
        "includesPolarization": False,
        "includesReturningRadiation": False,
        "prohibitedClaim": (
            "Do not describe the linear angular law as a solved disk atmosphere, "
            "GRMHD radiation field, polarization model, or returning-radiation model."
        ),
    }
)

_UNIT_TOLERANCE: Final = 4.0e-10


class DiskAtmosphereError(RuntimeError):
    """Raised when a local angle or angular intensity is not physical."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _direction_cosine(value: Any) -> float:
    cosine = _finite_number(value, "emission_angle_cosine")
    if cosine < 0.0 or cosine > 1.0:
        raise ValueError("emission_angle_cosine must lie in [0, 1]")
    return cosine


@runtime_checkable
class AngularEmissionLaw(Protocol):
    """One dimensionless, bolometric-flux-normalized angular law."""

    def intensity_multiplier(self, emission_angle_cosine: float) -> float:
        """Return the local specific-intensity multiplier at ``mu``."""

        ...

    def descriptor(self) -> Mapping[str, Any]:
        """Return deterministic finite identity metadata."""

        ...


@dataclass(frozen=True, slots=True)
class IsotropicAngularEmission:
    """Locally isotropic outward-hemisphere intensity."""

    implementation_id: str = field(
        default="isotropic-angular-emission/v1",
        init=False,
    )

    def intensity_multiplier(self, emission_angle_cosine: float) -> float:
        _direction_cosine(emission_angle_cosine)
        return 1.0

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "fluxNormalization": "2 integral_0^1 mu f(mu) dmu = 1",
            "implementationId": self.implementation_id,
            "kind": "isotropic",
        }


@dataclass(frozen=True, slots=True)
class FluxConservingLinearLimbDarkening:
    """Normalized ``1 + b mu`` electron-scattering limb-darkening proxy.

    The normalization ``1 / (1 + 2 b / 3)`` keeps
    ``2 integral_0^1 mu f(mu) dmu = 1``.  Therefore applying this angular law
    redistributes a one-face Novikov--Thorne flux without changing that flux.
    The default ``b=1.5`` is exactly the KERRBB D20 law
    ``1/2 + 3/4 mu``.  A caller may explicitly choose another finite
    coefficient, such as the separately used ``1 + 2.06 mu`` ionized-surface
    proxy, but the descriptor preserves that choice.
    """

    coefficient: float = 1.5
    implementation_id: str = field(
        default="flux-conserving-linear-limb-darkening/v1",
        init=False,
    )

    def __post_init__(self) -> None:
        coefficient = _finite_number(self.coefficient, "coefficient")
        if coefficient < 0.0:
            raise ValueError("limb-darkening coefficient must be non-negative")
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def normalization(self) -> float:
        if self.coefficient <= 1.0:
            return 1.0 / (1.0 + 2.0 * self.coefficient / 3.0)
        inverse_coefficient = 1.0 / self.coefficient
        return inverse_coefficient / (inverse_coefficient + 2.0 / 3.0)

    def intensity_multiplier(self, emission_angle_cosine: float) -> float:
        cosine = _direction_cosine(emission_angle_cosine)
        if self.coefficient <= 1.0:
            multiplier = self.normalization * (
                1.0 + self.coefficient * cosine
            )
        else:
            inverse_coefficient = 1.0 / self.coefficient
            multiplier = (inverse_coefficient + cosine) / (
                inverse_coefficient + 2.0 / 3.0
            )
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise DiskAtmosphereError("limb-darkening multiplier is invalid")
        return multiplier

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "coefficient": self.coefficient,
            "fluxNormalization": "2 integral_0^1 mu f(mu) dmu = 1",
            "implementationId": self.implementation_id,
            "kind": "linear-electron-scattering-proxy",
            "normalization": self.normalization,
        }


def apply_angular_emission(
    isotropic_specific_intensity_nu: float,
    emission_angle_cosine: float,
    law: AngularEmissionLaw,
) -> float:
    """Apply one declared dimensionless angular law to local ``I_nu``."""

    intensity = _finite_number(
        isotropic_specific_intensity_nu,
        "isotropic_specific_intensity_nu",
    )
    if intensity < 0.0:
        raise ValueError("isotropic_specific_intensity_nu must be non-negative")
    if not isinstance(law, AngularEmissionLaw):
        raise TypeError("law must implement AngularEmissionLaw")
    multiplier = _finite_number(
        law.intensity_multiplier(emission_angle_cosine),
        "angular intensity multiplier",
    )
    if multiplier < 0.0:
        raise DiskAtmosphereError("angular intensity multiplier must be non-negative")
    result = intensity * multiplier
    if not math.isfinite(result) or result < 0.0:
        raise DiskAtmosphereError("angular specific intensity overflowed")
    return result


def equatorial_emission_angle_cosine(
    metric: KerrKerrSchildMetric,
    photon_state: HamiltonianState,
    emitter: KerrDiskEmitter,
    *,
    null_residual_limit: float = 1.0e-7,
    emitter_event_tolerance_m: float | None = None,
) -> float:
    """Return ``mu=|cos(theta_emit)|`` for one past-directed disk photon.

    The surface-normal contraction is evaluated from the exact Kerr Carter
    momentum at the equatorial plane.  This is algebraically equivalent to
    projecting the ``z=0`` gradient into the circular emitter's rest space,
    but remains well conditioned near the extremal prograde ISCO.  With the
    repository's past-directed covector convention, the positive circular-
    emitter frequency is ``u^t(-E + Omega Lz)`` and
    ``sqrt(Q) / (r nu_emit)`` is the required direction cosine.  The absolute
    normal momentum selects whichever infinitesimal disk face the observer
    sees.
    """

    if not isinstance(metric, KerrKerrSchildMetric):
        raise TypeError("metric must be an exact KerrKerrSchildMetric")
    if not isinstance(photon_state, HamiltonianState):
        raise TypeError("photon_state must be a HamiltonianState")
    if not isinstance(emitter, KerrDiskEmitter):
        raise TypeError("emitter must be a KerrDiskEmitter")
    if not math.isclose(
        emitter.kerr_mass_m,
        metric.mass_m,
        rel_tol=2.0e-13,
        abs_tol=0.0,
    ) or not math.isclose(
        emitter.kerr_spin_a_m,
        metric.spin_a_m,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13 * metric.mass_m,
    ):
        raise ValueError("emitter was constructed for a different Kerr metric")
    residual_limit = _finite_number(null_residual_limit, "null_residual_limit")
    if residual_limit <= 0.0:
        raise ValueError("null_residual_limit must be positive")
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
        abs(photon_state.event[index] - emitter.event[index]) > event_tolerance
        for index in range(4)
    ):
        raise ValueError("photon state does not match the declared disk event")
    residual = hamiltonian_null_residual(metric, photon_state)
    if not math.isfinite(residual) or residual > residual_limit:
        raise ValueError("photon covector exceeds the declared null residual limit")

    # Evaluate the local angle through Kerr invariants rather than subtracting
    # large Cartesian Kerr--Schild components.  Close to an extremal prograde
    # ISCO, u^t and the KS spatial components are O(10^3) or larger while the
    # locally measured photon frequency remains O(1); a direct coordinate
    # contraction can therefore lose the very digits needed to distinguish
    # mu=1 from an unphysical mu>1.  At the equatorial plane Carter Q=p_theta^2
    # and the unit-normal projection is |p_theta|/r.  The circular-emitter
    # frequency is u^t(-E+Omega Lz), with
    # u^t=1/(E_emitter-Omega L_emitter).  These separated invariants are the
    # same exact Kerr quantities used to audit the Hamiltonian ray.
    covector_scale = max(abs(value) for value in photon_state.covector)
    if not math.isfinite(covector_scale) or covector_scale <= 0.0:
        raise ValueError("photon covector must have a finite non-zero scale")
    normalized_state = HamiltonianState(
        photon_state.event,
        tuple(value / covector_scale for value in photon_state.covector),
    )
    constants = kerr_constants_of_motion(metric, normalized_state)
    angular_velocity = emitter.angular_velocity_inverse_m
    emitter_redshift_denominator = math.fsum(
        (
            emitter.specific_energy,
            -angular_velocity * emitter.specific_angular_momentum_m,
        )
    )
    corotating_photon_energy = math.fsum(
        (
            -constants.energy,
            angular_velocity * constants.angular_momentum_z,
        )
    )
    if (
        not math.isfinite(emitter_redshift_denominator)
        or emitter_redshift_denominator <= 0.0
        or not math.isfinite(corotating_photon_energy)
        or corotating_photon_energy <= 0.0
    ):
        raise ValueError("emitter photon frequency must be positive")
    emitter_frequency = corotating_photon_energy / emitter_redshift_denominator

    carter_scale = max(
        abs(constants.carter_k),
        abs(
            constants.angular_momentum_z
            - metric.spin_a_m * constants.energy
        )
        ** 2,
        (emitter.radius_m * constants.energy) ** 2,
        1.0e-300,
    )
    carter_q = constants.carter_q
    if carter_q < -_UNIT_TOLERANCE * carter_scale:
        raise DiskAtmosphereError(
            "equatorial photon has a negative Carter normal momentum"
        )
    normal_frequency = math.sqrt(max(0.0, carter_q)) / emitter.radius_m
    if not math.isfinite(emitter_frequency) or emitter_frequency <= 0.0:
        raise ValueError("emitter photon frequency must be positive")
    cosine = abs(normal_frequency) / emitter_frequency
    if not math.isfinite(cosine) or cosine < 0.0:
        raise DiskAtmosphereError("emission angle cosine is invalid")
    if cosine > 1.0 + _UNIT_TOLERANCE:
        raise DiskAtmosphereError("emission angle cosine exceeds the local light cone")
    return min(1.0, cosine)


__all__ = (
    "AngularEmissionLaw",
    "DiskAtmosphereError",
    "FluxConservingLinearLimbDarkening",
    "IsotropicAngularEmission",
    "SCIENTIFIC_STATUS",
    "apply_angular_emission",
    "equatorial_emission_angle_cosine",
)
