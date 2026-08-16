"""First-visible geometry adapter for the finite-thickness Kerr calibration.

This module connects :mod:`offline.kerr_finite_thickness` to the generic
accepted-step multi-surface event layer.  Upper and lower photospheres retain
independent stable ids and signed scalars.  Only crossings whose
pseudo-cylindrical radius lies in ``[r_ISCO, r_out]`` are opaque candidates;
crossings of the auxiliary signed continuations are explicitly transparent.

The continuation is only a numerical device needed to evaluate finite signed
fields along a ray.  It adds no radial sidewall and has no emission or fluid
meaning.  A zero-accretion/zero-thickness calibration is rejected because its
upper and lower faces coincide; callers must use the existing equatorial
zero-thickness path instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping

from offline.geodesic import (
    HamiltonianState,
    InteriorSurfaceDecision,
    RecordedSurfaceCrossing,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_ks_event_to_oblate_meridional,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)


UPPER_SURFACE_ID: Final = "kerr-finite-thickness-upper-photosphere"
LOWER_SURFACE_ID: Final = "kerr-finite-thickness-lower-photosphere"
FINITE_THICKNESS_SURFACE_IDS: Final = (
    LOWER_SURFACE_ID,
    UPPER_SURFACE_ID,
)
OPAQUE_OUTCOME: Final = "opaque-finite-thickness-disk-hit"
UPPER_TARGET_ID: Final = "kerr-finite-thickness-opaque-upper-face"
LOWER_TARGET_ID: Final = "kerr-finite-thickness-opaque-lower-face"

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "stationary Kerr finite-thickness first-visible geometry adapter"
        ),
        "surfaceRepresentation": (
            "independent stable-id upper and lower signed photospheres"
        ),
        "opaqueRadialDomain": "rho in [r_ISCO, r_out]",
        "auxiliaryContinuation": (
            "transparent signed face continuations outside the opaque annulus"
        ),
        "includesRadialSidewall": False,
        "acceptsZeroThickness": False,
        "includesSpectrum": False,
        "includesReturningRadiation": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "prohibitedClaim": (
            "This is first-visible event geometry only, not a complete "
            "finite-thickness renderer, atmosphere, returning-radiation "
            "transport, hydrostatic disk, or GRMHD simulation."
        ),
    }
)


def _surface_face(surface_id: str) -> str:
    if surface_id == UPPER_SURFACE_ID:
        return UPPER
    if surface_id == LOWER_SURFACE_ID:
        return LOWER
    raise ValueError("finite-thickness surface id is not declared")


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessMultiSurface:
    """Independent upper/lower signed faces for accepted-step ray events."""

    metric: KerrKerrSchildMetric
    calibration: StationaryKerrFiniteThicknessCalibration
    surface_ids: tuple[str, str] = field(
        default=FINITE_THICKNESS_SURFACE_IDS,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.metric, KerrKerrSchildMetric):
            raise TypeError("metric must be an exact KerrKerrSchildMetric")
        if not isinstance(
            self.calibration,
            StationaryKerrFiniteThicknessCalibration,
        ):
            raise TypeError(
                "calibration must be a StationaryKerrFiniteThicknessCalibration"
            )
        metric_spin = abs(self.metric.dimensionless_spin)
        calibration_spin = self.calibration.dimensionless_spin
        if not math.isclose(
            metric_spin,
            calibration_spin,
            rel_tol=0.0,
            abs_tol=8.0 * math.ulp(max(1.0, calibration_spin)),
        ):
            raise ValueError("metric and finite-thickness spin are inconsistent")
        if self.calibration.eddington_scaled_mass_accretion_rate == 0.0:
            raise ValueError(
                "zero-thickness upper/lower faces coincide; use the existing "
                "equatorial zero-thickness path"
            )

    def _oblate_coordinates_over_mass(
        self,
        state: HamiltonianState,
    ) -> tuple[float, float, float]:
        if not isinstance(state, HamiltonianState):
            raise TypeError("finite-thickness surface state must be HamiltonianState")
        oblate = kerr_ks_event_to_oblate_meridional(self.metric, state.event)
        radius = oblate.radius_m / self.metric.mass_m
        rho = radius * math.sin(oblate.theta_rad)
        height = radius * math.cos(oblate.theta_rad)
        if not all(math.isfinite(value) for value in (radius, rho, height)):
            raise ValueError("finite-thickness oblate coordinates are not finite")
        if radius <= 0.0 or rho < 0.0:
            raise ValueError("finite-thickness oblate coordinates are invalid")
        return rho, height, oblate.theta_rad

    def auxiliary_photosphere_height_over_mass(self, rho: float) -> float:
        """Return a positive continuous face height for signed evaluation.

        Inside the ISCO, a V-shaped transparent continuation keeps the two
        faces distinct except at their physical inner-edge endpoint.  Outside
        ``r_out`` the last physical height is continued horizontally.  These
        extensions are not opaque material and do not create a radial wall.
        """

        if (
            isinstance(rho, bool)
            or not isinstance(rho, (int, float))
            or not math.isfinite(float(rho))
            or rho < 0.0
        ):
            raise ValueError(
                "pseudo-cylindrical radius must be finite and non-negative"
            )
        radius = float(rho)
        inner = self.calibration.isco_radius_over_mass
        outer = self.calibration.outer_radius_over_mass
        if radius < inner:
            inner_slope = self.calibration.photosphere_height_derivative(inner)
            return inner_slope * (inner - radius)
        if radius > outer:
            return self.calibration.photosphere_height_over_mass(outer)
        return self.calibration.photosphere_height_over_mass(radius)

    def value(
        self,
        surface_id: str,
        state: HamiltonianState,
    ) -> float:
        """Return the member's dimensionless outward-positive signed field."""

        face = _surface_face(surface_id)
        rho, signed_height, _theta = self._oblate_coordinates_over_mass(state)
        face_height = self.auxiliary_photosphere_height_over_mass(rho)
        value = (
            signed_height - face_height
            if face == UPPER
            else -signed_height - face_height
        )
        if not math.isfinite(value):
            raise ValueError("finite-thickness signed surface is not finite")
        return value

    def classify(
        self,
        surface_id: str,
        crossing: RecordedSurfaceCrossing,
    ) -> InteriorSurfaceDecision:
        """Make only the physical annular part of each face opaque."""

        face = _surface_face(surface_id)
        if not isinstance(crossing, RecordedSurfaceCrossing):
            raise TypeError("crossing must be a RecordedSurfaceCrossing")
        rho, _height, _theta = self._oblate_coordinates_over_mass(crossing.state)
        inner = self.calibration.isco_radius_over_mass
        outer = self.calibration.outer_radius_over_mass
        if rho < inner:
            return InteriorSurfaceDecision("inside-isco-transparent")
        if rho > outer:
            return InteriorSurfaceDecision("outside-outer-radius-transparent")
        return InteriorSurfaceDecision(
            f"opaque-{face}-photosphere",
            OPAQUE_OUTCOME,
            UPPER_TARGET_ID if face == UPPER else LOWER_TARGET_ID,
        )
