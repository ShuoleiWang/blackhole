"""Exact stationary Kerr calibration objects for the offline renderer.

The metric uses ingoing Cartesian Kerr--Schild coordinates with signature
``-+++`` and spin aligned with coordinate ``+z``.  It is an exact analytic
vacuum solution, not numerical-relativity or GRMHD data.  The module also owns
the matching Boyer--Lindquist ZAMO camera transform and constant-oblate-radius
terminal surfaces so coordinate conventions cannot drift between producers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from offline.geodesic import (
    HamiltonianState,
    TerminationCrossing,
)
from offline.spacetime import (
    MINKOWSKI_COVARIANT,
    ZERO_MATRIX4,
    Matrix4,
    MetricSample,
    Vector4,
    bilinear,
    matrix_vector,
)


_TETRAD_GRAM_TOLERANCE = 2.0e-10
_CAMERA_NULL_ABSOLUTE_TOLERANCE = 2.0e-10


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _canonical_exact_float(value: object, label: str) -> float:
    """Return one built-in binary64 value without invoking numeric subclasses."""

    if type(value) not in (int, float):
        raise TypeError(f"{label} must be an exact int or float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} cannot be represented as binary64") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def kerr_oblate_radius_m(
    x_m: float,
    y_m: float,
    z_m: float,
    spin_a_m: float,
) -> float:
    """Return the non-negative Kerr oblate radial coordinate ``r``.

    It is the outer-sheet solution of
    ``(x^2+y^2)/(r^2+a^2) + z^2/r^2 = 1``.  The function deliberately does
    not regularize the Kerr ring; metric providers must reject that region.
    """

    values = (x_m, y_m, z_m, spin_a_m)
    if not _finite(values):
        raise ValueError("Kerr coordinates and spin must be finite")
    rho_squared = math.fsum(value * value for value in (x_m, y_m, z_m))
    spin_squared = spin_a_m * spin_a_m
    difference = rho_squared - spin_squared
    spin_position = spin_a_m * z_m
    discriminant_root = math.hypot(difference, 2.0 * spin_position)
    if not math.isfinite(discriminant_root):
        raise ValueError("Kerr oblate-radius discriminant is invalid")
    if difference >= 0.0:
        radius_squared = 0.5 * (difference + discriminant_root)
    elif spin_position != 0.0:
        # Rationalized form avoids cancellation inside the equatorial ring.
        radius_squared = (
            2.0 * spin_position * spin_position
            / (discriminant_root - difference)
        )
    else:
        radius_squared = 0.0
    # Roundoff can only make the analytic non-negative root slightly negative.
    scale = max(1.0, rho_squared, spin_squared)
    if radius_squared < -16.0 * math.ulp(scale):
        raise ValueError("Kerr oblate radius squared is negative")
    return math.sqrt(max(0.0, radius_squared))


@dataclass(frozen=True)
class KerrKerrSchildMetric:
    """Exact Kerr metric in ingoing Cartesian Kerr--Schild coordinates.

    ``spin_a_m`` is the signed dimensional Kerr parameter.  Positive spin is
    along coordinate ``+z``.  No near-ring smoothing or cap is permitted in
    this scientific provider: guarded samples fail closed instead.
    """

    mass_m: float = 1.0
    spin_a_m: float = 0.0
    singularity_guard_m: float = 1.0e-9
    source_id: str = "analytic-kerr-kerr-schild"
    time_dependent: bool = False

    def __post_init__(self) -> None:
        mass_m = _canonical_exact_float(self.mass_m, "Kerr mass")
        spin_a_m = _canonical_exact_float(self.spin_a_m, "Kerr spin")
        guard_m = _canonical_exact_float(
            self.singularity_guard_m,
            "Kerr singularity guard",
        )
        if type(self.source_id) is not str or (
            self.source_id != "analytic-kerr-kerr-schild"
        ):
            raise ValueError("Kerr source_id must name the exact analytic provider")
        if type(self.time_dependent) is not bool or self.time_dependent is not False:
            raise ValueError("analytic Kerr provider must be exactly stationary")
        if mass_m <= 0.0:
            raise ValueError("Kerr mass must be finite and positive")
        if abs(spin_a_m) > mass_m:
            raise ValueError("Kerr spin must be finite and satisfy |a| <= M")
        if guard_m <= 0.0:
            raise ValueError("Kerr singularity guard must be finite and positive")
        object.__setattr__(self, "mass_m", mass_m)
        object.__setattr__(self, "spin_a_m", spin_a_m)
        object.__setattr__(self, "singularity_guard_m", guard_m)

    @property
    def dimensionless_spin(self) -> float:
        return self.spin_a_m / self.mass_m

    @property
    def outer_horizon_radius_m(self) -> float:
        spin_magnitude = abs(self.spin_a_m)
        return self.mass_m + math.sqrt(
            (self.mass_m - spin_magnitude) * (self.mass_m + spin_magnitude)
        )

    def oblate_radius_m(self, event: Vector4) -> float:
        if not _finite(event):
            raise ValueError("Kerr metric event must be finite")
        return kerr_oblate_radius_m(*event[1:], self.spin_a_m)

    def sample(self, event: Vector4) -> MetricSample:
        if not _finite(event):
            raise ValueError("Kerr metric event must be finite")
        _time, x_m, y_m, z_m = event
        spin = self.spin_a_m
        spin_squared = spin * spin
        rho_squared = math.fsum(value * value for value in (x_m, y_m, z_m))
        difference = rho_squared - spin_squared
        spin_position = spin * z_m
        discriminant_root = math.hypot(difference, 2.0 * spin_position)
        if not math.isfinite(discriminant_root):
            raise ValueError("Kerr oblate-radius discriminant is invalid")
        if difference >= 0.0:
            radius_squared = 0.5 * (difference + discriminant_root)
        elif spin_position != 0.0:
            radius_squared = (
                2.0 * spin_position * spin_position
                / (discriminant_root - difference)
            )
        else:
            radius_squared = 0.0
        radius = math.sqrt(max(0.0, radius_squared))
        if (
            radius <= self.singularity_guard_m
            or discriminant_root <= self.singularity_guard_m**2
        ):
            raise ValueError("Kerr metric sampled at guarded ring/interior disk")

        radius_derivatives = (
            radius_squared * x_m / (radius * discriminant_root),
            radius_squared * y_m / (radius * discriminant_root),
            (radius_squared + spin_squared)
            * z_m
            / (radius * discriminant_root),
        )

        radius_fourth = radius_squared * radius_squared
        h_denominator = radius_fourth + spin_squared * z_m * z_m
        if (
            not math.isfinite(h_denominator)
            or h_denominator <= self.singularity_guard_m**4
        ):
            raise ValueError("Kerr-Schild scalar sampled at guarded singularity")
        h = self.mass_m * radius * radius_squared / h_denominator

        spatial_denominator = radius_squared + spin_squared
        numerator_x = radius * x_m + spin * y_m
        numerator_y = radius * y_m - spin * x_m
        spatial_null = (
            numerator_x / spatial_denominator,
            numerator_y / spatial_denominator,
            z_m / radius,
        )
        l_covariant: Vector4 = (1.0, *spatial_null)
        l_contravariant: Vector4 = (-1.0, *spatial_null)

        covariant = tuple(  # type: ignore[assignment]
            tuple(
                MINKOWSKI_COVARIANT[row][column]
                + 2.0 * h * l_covariant[row] * l_covariant[column]
                for column in range(4)
            )
            for row in range(4)
        )
        inverse = tuple(  # type: ignore[assignment]
            tuple(
                MINKOWSKI_COVARIANT[row][column]
                - 2.0 * h * l_contravariant[row] * l_contravariant[column]
                for column in range(4)
            )
            for row in range(4)
        )

        derivatives: list[Matrix4] = [ZERO_MATRIX4]
        for axis in range(3):
            derivative_radius = radius_derivatives[axis]
            derivative_h_denominator = (
                4.0 * radius * radius_squared * derivative_radius
                + (2.0 * spin_squared * z_m if axis == 2 else 0.0)
            )
            derivative_h = h * (
                3.0 * derivative_radius / radius
                - derivative_h_denominator / h_denominator
            )
            derivative_spatial_denominator = 2.0 * radius * derivative_radius
            derivative_numerator_x = (
                derivative_radius * x_m
                + (radius if axis == 0 else 0.0)
                + (spin if axis == 1 else 0.0)
            )
            derivative_numerator_y = (
                derivative_radius * y_m
                + (radius if axis == 1 else 0.0)
                - (spin if axis == 0 else 0.0)
            )
            derivative_l: Vector4 = (
                0.0,
                (
                    derivative_numerator_x * spatial_denominator
                    - numerator_x * derivative_spatial_denominator
                )
                / (spatial_denominator * spatial_denominator),
                (
                    derivative_numerator_y * spatial_denominator
                    - numerator_y * derivative_spatial_denominator
                )
                / (spatial_denominator * spatial_denominator),
                (1.0 if axis == 2 else 0.0) / radius
                - z_m * derivative_radius / radius_squared,
            )
            derivative = tuple(  # type: ignore[assignment]
                tuple(
                    -2.0
                    * (
                        derivative_h
                        * l_contravariant[row]
                        * l_contravariant[column]
                        + h
                        * derivative_l[row]
                        * l_contravariant[column]
                        + h
                        * l_contravariant[row]
                        * derivative_l[column]
                    )
                    for column in range(4)
                )
                for row in range(4)
            )
            derivatives.append(derivative)

        return MetricSample(
            covariant=covariant,
            inverse=inverse,
            inverse_derivatives=tuple(derivatives),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class KerrOblateTermination:
    """Capture and escape worldtubes at constant Kerr oblate radius."""

    spin_a_m: float
    capture_radius_m: float
    escape_radius_m: float
    capture_target_id: str = "analytic-kerr-stretched-horizon"
    escape_target_id: str = "analytic-kerr-escape-worldtube"

    def __post_init__(self) -> None:
        spin_a_m = _canonical_exact_float(
            self.spin_a_m,
            "Kerr terminal spin",
        )
        capture_radius_m = _canonical_exact_float(
            self.capture_radius_m,
            "Kerr capture radius",
        )
        escape_radius_m = _canonical_exact_float(
            self.escape_radius_m,
            "Kerr escape radius",
        )
        if capture_radius_m <= 0.0 or escape_radius_m <= capture_radius_m:
            raise ValueError("Kerr surfaces require 0 < capture < escape")
        if (
            type(self.capture_target_id) is not str
            or not self.capture_target_id
            or type(self.escape_target_id) is not str
            or not self.escape_target_id
        ):
            raise ValueError("Kerr terminal surfaces need non-empty target ids")
        object.__setattr__(self, "spin_a_m", spin_a_m)
        object.__setattr__(self, "capture_radius_m", capture_radius_m)
        object.__setattr__(self, "escape_radius_m", escape_radius_m)

    @classmethod
    def horizon_worldtube(
        cls,
        metric: KerrKerrSchildMetric,
        *,
        escape_radius_m: float,
        offset_m: float = 0.02,
    ) -> "KerrOblateTermination":
        if not math.isfinite(offset_m) or offset_m < 0.0:
            raise ValueError("horizon offset must be finite and non-negative")
        target_id = (
            "analytic-kerr-event-horizon"
            if offset_m == 0.0
            else "analytic-kerr-stretched-horizon"
        )
        return cls(
            spin_a_m=metric.spin_a_m,
            capture_radius_m=metric.outer_horizon_radius_m + offset_m,
            escape_radius_m=escape_radius_m,
            capture_target_id=target_id,
        )

    def radius(self, state: HamiltonianState) -> float:
        return kerr_oblate_radius_m(*state.event[1:], self.spin_a_m)

    def classify_initial(self, state: HamiltonianState) -> tuple[str, str] | None:
        radius = self.radius(state)
        if radius <= self.capture_radius_m:
            return ("captured", self.capture_target_id)
        if radius >= self.escape_radius_m:
            return ("escaped", self.escape_target_id)
        return None

    def value(self, state: HamiltonianState, crossing: TerminationCrossing) -> float:
        radius = self.radius(state)
        if crossing.outcome == "captured":
            return radius - self.capture_radius_m
        if crossing.outcome == "escaped":
            return radius - self.escape_radius_m
        raise ValueError(f"unsupported Kerr terminal outcome {crossing.outcome!r}")

    def crossing(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> TerminationCrossing | None:
        previous_radius = self.radius(previous)
        current_radius = self.radius(current)
        capture_before = previous_radius - self.capture_radius_m
        capture_after = current_radius - self.capture_radius_m
        if capture_before > 0.0 and capture_after <= 0.0:
            return TerminationCrossing(
                "captured",
                self.capture_target_id,
                capture_before,
                capture_after,
            )
        escape_before = previous_radius - self.escape_radius_m
        escape_after = current_radius - self.escape_radius_m
        if escape_before < 0.0 and escape_after >= 0.0:
            return TerminationCrossing(
                "escaped",
                self.escape_target_id,
                escape_before,
                escape_after,
            )
        return None

    def needs_refinement(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> bool:
        """Detect an accepted chord that enters and exits the capture spheroid."""

        if (
            self.radius(previous) <= self.capture_radius_m
            or self.radius(current) <= self.capture_radius_m
        ):
            return False
        radius_squared = self.capture_radius_m * self.capture_radius_m
        equatorial_squared = radius_squared + self.spin_a_m * self.spin_a_m
        start = previous.event[1:]
        delta = tuple(
            current.event[index + 1] - previous.event[index + 1]
            for index in range(3)
        )
        weights = (
            1.0 / equatorial_squared,
            1.0 / equatorial_squared,
            1.0 / radius_squared,
        )
        quadratic = math.fsum(
            weights[index] * delta[index] * delta[index] for index in range(3)
        )
        if quadratic == 0.0:
            return False
        linear_half = math.fsum(
            weights[index] * start[index] * delta[index] for index in range(3)
        )
        fraction = min(1.0, max(0.0, -linear_half / quadratic))
        minimum_level = math.fsum(
            weights[index]
            * (start[index] + fraction * delta[index]) ** 2
            for index in range(3)
        )
        return minimum_level <= 1.0


@dataclass(frozen=True)
class KerrZamoTetrad:
    """A finite-radius BL-ZAMO tetrad expressed in Cartesian KS components."""

    event: Vector4
    four_velocity: Vector4
    right: Vector4
    up: Vector4
    forward: Vector4


@dataclass(frozen=True)
class KerrConstantsOfMotion:
    """Stationary Kerr photon constants in the signed past/future convention."""

    energy: float
    angular_momentum_z: float
    carter_q: float
    carter_k: float


@dataclass(frozen=True)
class KerrOblateEvent:
    """One Kerr-Schild event expressed in oblate spatial coordinates.

    ``phi_ks_rad`` is the ingoing Kerr-Schild azimuth, not the Euclidean
    ``atan2(y, x)`` angle.  Keeping that distinction in a public value object
    prevents disk and camera producers from silently rotating by
    ``atan2(a, r)``.
    """

    coordinate_time_m: float
    radius_m: float
    theta_rad: float
    phi_ks_rad: float


@dataclass(frozen=True)
class KerrOblateMeridionalEvent:
    """Kerr-Schild event reduced to the globally defined ``(t, r, theta)``.

    Unlike :class:`KerrOblateEvent`, this value is well-defined on the spin
    axis where every azimuth labels the same event.  Surface geometry that
    does not need ``phi_KS`` should use this narrower inverse transform.
    """

    coordinate_time_m: float
    radius_m: float
    theta_rad: float


def _stable_kerr_delta(mass_m: float, spin_a_m: float, radius_m: float) -> float:
    spin_magnitude = abs(spin_a_m)
    horizon_separation = math.sqrt(
        (mass_m - spin_magnitude) * (mass_m + spin_magnitude)
    )
    outer_horizon = mass_m + horizon_separation
    inner_horizon = mass_m - horizon_separation
    delta = (radius_m - outer_horizon) * (radius_m - inner_horizon)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("Boyer-Lindquist event must lie outside the outer horizon")
    return delta


def kerr_oblate_event_to_ks_cartesian(
    *,
    coordinate_time_m: float,
    radius_m: float,
    theta_rad: float,
    phi_ks_rad: float,
    spin_a_m: float,
) -> Vector4:
    """Map oblate Kerr-Schild coordinates to a Cartesian KS event."""

    values = (
        coordinate_time_m,
        radius_m,
        theta_rad,
        phi_ks_rad,
        spin_a_m,
    )
    if not _finite(values):
        raise ValueError("Kerr oblate event coordinates must be finite")
    if radius_m <= 0.0:
        raise ValueError("Kerr oblate event radius must be positive")
    if theta_rad < 0.0 or theta_rad > math.pi:
        raise ValueError("Kerr oblate event polar angle must lie in [0, pi]")
    sine = math.sin(theta_rad)
    cosine = math.cos(theta_rad)
    cosine_phi = math.cos(phi_ks_rad)
    sine_phi = math.sin(phi_ks_rad)
    return (
        coordinate_time_m,
        (radius_m * cosine_phi - spin_a_m * sine_phi) * sine,
        (radius_m * sine_phi + spin_a_m * cosine_phi) * sine,
        radius_m * cosine,
    )


def kerr_ks_event_to_oblate(
    metric: KerrKerrSchildMetric,
    event: Vector4,
) -> KerrOblateEvent:
    """Recover ``(t, r, theta, phi_KS)`` from a Cartesian KS event.

    The azimuth is undefined on the spin axis and therefore fails closed.
    Scientific disk callers only use this inverse on equatorial crossings.
    """

    meridional = kerr_ks_event_to_oblate_meridional(metric, event)
    _coordinate_time_m, x_m, y_m, _z_m = event
    radius_m = meridional.radius_m
    cosine_numerator = radius_m * x_m + metric.spin_a_m * y_m
    sine_numerator = radius_m * y_m - metric.spin_a_m * x_m
    if cosine_numerator == 0.0 and sine_numerator == 0.0:
        raise ValueError("Kerr-Schild azimuth is undefined on the spin axis")
    phi_ks_rad = math.atan2(sine_numerator, cosine_numerator)
    return KerrOblateEvent(
        coordinate_time_m=meridional.coordinate_time_m,
        radius_m=radius_m,
        theta_rad=meridional.theta_rad,
        phi_ks_rad=phi_ks_rad,
    )


def kerr_ks_event_to_oblate_meridional(
    metric: KerrKerrSchildMetric,
    event: Vector4,
) -> KerrOblateMeridionalEvent:
    """Recover globally defined oblate ``(t, r, theta)`` from Cartesian KS.

    The polar angle uses the same analytic oblate relations as
    :func:`kerr_ks_event_to_oblate`, but deliberately does not request an
    azimuth.  It therefore remains valid at ``x=y=0`` while preserving the
    guarded-ring rejection owned by the exact Kerr metric.
    """

    if not isinstance(metric, KerrKerrSchildMetric):
        raise TypeError("metric must be an exact KerrKerrSchildMetric")
    if not _finite(event):
        raise ValueError("Kerr-Schild event must be finite")
    coordinate_time_m, x_m, y_m, z_m = event
    radius_m = metric.oblate_radius_m(event)
    if radius_m <= metric.singularity_guard_m:
        raise ValueError("Kerr-Schild event is inside the guarded ring region")
    equatorial_scale = math.hypot(radius_m, metric.spin_a_m)
    sine_theta = math.hypot(x_m, y_m) / equatorial_scale
    cosine_theta = z_m / radius_m
    theta_rad = math.atan2(sine_theta, cosine_theta)
    if not math.isfinite(theta_rad):
        raise ValueError("Kerr oblate polar angle is not finite")
    return KerrOblateMeridionalEvent(
        coordinate_time_m=coordinate_time_m,
        radius_m=radius_m,
        theta_rad=theta_rad,
    )


def kerr_bl_vector_to_ks_cartesian(
    vector: Vector4,
    *,
    mass_m: float,
    spin_a_m: float,
    radius_m: float,
    theta_rad: float,
    phi_ks_rad: float,
) -> Vector4:
    """Transform BL vector components at an oblate event to Cartesian KS."""

    if not _finite((*vector, mass_m, spin_a_m, radius_m, theta_rad, phi_ks_rad)):
        raise ValueError("Kerr BL vector transform inputs must be finite")
    if mass_m <= 0.0 or abs(spin_a_m) > mass_m:
        raise ValueError("Kerr BL vector transform parameters are invalid")
    if theta_rad < 0.0 or theta_rad > math.pi:
        raise ValueError("Kerr BL vector polar angle must lie in [0, pi]")
    sine = math.sin(theta_rad)
    cosine = math.cos(theta_rad)
    cosine_phi = math.cos(phi_ks_rad)
    sine_phi = math.sin(phi_ks_rad)
    delta = _stable_kerr_delta(mass_m, spin_a_m, radius_m)
    dt_dr = 2.0 * mass_m * radius_m / delta
    dphi_dr = spin_a_m / delta
    x_m = (radius_m * cosine_phi - spin_a_m * sine_phi) * sine
    y_m = (radius_m * sine_phi + spin_a_m * cosine_phi) * sine
    dx_dr = (
        cosine_phi
        - (radius_m * sine_phi + spin_a_m * cosine_phi) * dphi_dr
    ) * sine
    dy_dr = (
        sine_phi
        + (radius_m * cosine_phi - spin_a_m * sine_phi) * dphi_dr
    ) * sine
    dx_dtheta = (radius_m * cosine_phi - spin_a_m * sine_phi) * cosine
    dy_dtheta = (radius_m * sine_phi + spin_a_m * cosine_phi) * cosine
    dz_dtheta = -radius_m * sine
    v_t, v_r, v_theta, v_phi = vector
    return (
        v_t + dt_dr * v_r,
        dx_dr * v_r + dx_dtheta * v_theta - y_m * v_phi,
        dy_dr * v_r + dy_dtheta * v_theta + x_m * v_phi,
        cosine * v_r + dz_dtheta * v_theta,
    )

def kerr_bl_zamo_tetrad(
    metric: KerrKerrSchildMetric,
    *,
    observer_radius_m: float,
    theta_rad: float = 0.5 * math.pi,
    phi_ks_rad: float | None = None,
    coordinate_time_m: float = 0.0,
) -> KerrZamoTetrad:
    """Construct a BL-ZAMO camera frame in the provider's KS chart."""

    values = (observer_radius_m, theta_rad, coordinate_time_m)
    if not _finite(values):
        raise ValueError("Kerr ZAMO camera parameters must be finite")
    if observer_radius_m <= metric.outer_horizon_radius_m:
        raise ValueError("Kerr ZAMO observer must be outside the outer horizon")
    if theta_rad <= 0.0 or theta_rad >= math.pi or abs(math.sin(theta_rad)) < 1e-12:
        raise ValueError("Kerr ZAMO azimuthal basis is singular on the spin axis")
    if phi_ks_rad is None:
        phi_ks_rad = -math.atan2(metric.spin_a_m, observer_radius_m)
    if not math.isfinite(phi_ks_rad):
        raise ValueError("Kerr ZAMO azimuth must be finite")

    radius = observer_radius_m
    theta = theta_rad
    spin = metric.spin_a_m
    mass = metric.mass_m
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = _stable_kerr_delta(mass, spin, radius)
    big_a = (radius * radius + spin * spin) ** 2 - spin * spin * delta * sine * sine
    lapse = math.sqrt(sigma * delta / big_a)
    omega = 2.0 * mass * spin * radius / big_a

    cosine_phi = math.cos(phi_ks_rad)
    sine_phi = math.sin(phi_ks_rad)
    event = kerr_oblate_event_to_ks_cartesian(
        coordinate_time_m=coordinate_time_m,
        radius_m=radius,
        theta_rad=theta,
        phi_ks_rad=phi_ks_rad,
        spin_a_m=spin,
    )
    transform = lambda vector: kerr_bl_vector_to_ks_cartesian(  # noqa: E731
        vector,
        mass_m=mass,
        spin_a_m=spin,
        radius_m=radius,
        theta_rad=theta,
        phi_ks_rad=phi_ks_rad,  # type: ignore[arg-type]
    )
    four_velocity = transform((1.0 / lapse, 0.0, 0.0, omega / lapse))
    radial = transform((0.0, math.sqrt(delta / sigma), 0.0, 0.0))
    polar = transform((0.0, 0.0, 1.0 / math.sqrt(sigma), 0.0))
    azimuthal = transform((0.0, 0.0, 0.0, math.sqrt(sigma / big_a) / sine))
    negate = lambda vector: tuple(-value for value in vector)  # noqa: E731
    result = KerrZamoTetrad(
        event=event,
        four_velocity=four_velocity,
        right=negate(azimuthal),  # type: ignore[arg-type]
        up=negate(polar),  # type: ignore[arg-type]
        forward=negate(radial),  # type: ignore[arg-type]
    )
    sample = metric.sample(event)
    basis = (
        result.four_velocity,
        result.right,
        result.up,
        result.forward,
    )
    maximum_gram_error = max(
        abs(
            bilinear(basis[first], sample.covariant, basis[second])
            - (-1.0 if first == second == 0 else float(first == second))
        )
        for first in range(4)
        for second in range(4)
    )
    if (
        not math.isfinite(maximum_gram_error)
        or maximum_gram_error > _TETRAD_GRAM_TOLERANCE
    ):
        raise ValueError(
            "Boyer-Lindquist ZAMO transform is ill-conditioned at this event"
        )
    return result


def kerr_zamo_camera_ray(
    metric: KerrKerrSchildMetric,
    *,
    observer_radius_m: float,
    screen_x: float,
    screen_y: float,
    theta_rad: float = 0.5 * math.pi,
    phi_ks_rad: float | None = None,
    coordinate_time_m: float = 0.0,
) -> HamiltonianState:
    """Create a unit-local-frequency past-directed ZAMO pinhole ray."""

    if not _finite((screen_x, screen_y)):
        raise ValueError("Kerr camera screen coordinates must be finite")
    tetrad = kerr_bl_zamo_tetrad(
        metric,
        observer_radius_m=observer_radius_m,
        theta_rad=theta_rad,
        phi_ks_rad=phi_ks_rad,
        coordinate_time_m=coordinate_time_m,
    )
    inverse_norm = 1.0 / math.sqrt(1.0 + screen_x * screen_x + screen_y * screen_y)
    contravariant: Vector4 = tuple(  # type: ignore[assignment]
        -tetrad.four_velocity[index]
        + inverse_norm
        * (
            screen_x * tetrad.right[index]
            + screen_y * tetrad.up[index]
            + tetrad.forward[index]
        )
        for index in range(4)
    )
    covector = matrix_vector(metric.sample(tetrad.event).covariant, contravariant)
    state = HamiltonianState(event=tetrad.event, covector=covector)
    sample = metric.sample(state.event)
    raw_null = abs(bilinear(state.covector, sample.inverse, state.covector))
    local_frequency = math.fsum(
        state.covector[index] * tetrad.four_velocity[index]
        for index in range(4)
    )
    if (
        not math.isfinite(raw_null)
        or raw_null > _CAMERA_NULL_ABSOLUTE_TOLERANCE
        or not math.isfinite(local_frequency)
        or abs(local_frequency - 1.0) > _TETRAD_GRAM_TOLERANCE
    ):
        raise ValueError("Kerr ZAMO camera ray failed its local null normalization")
    return state


def stationary_axisymmetric_constants(
    state: HamiltonianState,
) -> tuple[float, float]:
    """Return ``(E=-p_t, Lz=x*p_y-y*p_x)`` in Cartesian KS coordinates."""

    _time, x_m, y_m, _z_m = state.event
    p_t, p_x, p_y, _p_z = state.covector
    return -p_t, x_m * p_y - y_m * p_x


def kerr_constants_of_motion(
    metric: KerrKerrSchildMetric,
    state: HamiltonianState,
) -> KerrConstantsOfMotion:
    """Recover ``E``, ``Lz``, Carter ``Q`` and ``K`` from a KS state.

    The BL-to-KS time and azimuth shifts depend only on ``r``.  Consequently
    ``p_t``, ``p_phi`` and ``p_theta`` at fixed ``r`` are unchanged, which lets
    the separated Kerr invariants serve as an independent audit of the generic
    Cartesian Hamiltonian integrator.
    """

    energy, angular_momentum = stationary_axisymmetric_constants(state)
    _time, x_m, y_m, z_m = state.event
    radius = metric.oblate_radius_m(state.event)
    if radius <= metric.singularity_guard_m:
        raise ValueError("Kerr constants are undefined at the guarded ring")
    cosine = z_m / radius
    if abs(cosine) > 1.0 + 2.0e-12:
        raise ValueError("Kerr polar coordinate lies outside its analytic range")
    cosine = min(1.0, max(-1.0, cosine))
    sine_squared = (x_m * x_m + y_m * y_m) / (
        radius * radius + metric.spin_a_m * metric.spin_a_m
    )
    if not math.isfinite(sine_squared) or sine_squared <= 0.0:
        raise ValueError("Kerr Carter constant is singular on the coordinate axis")
    if sine_squared > 1.0 + 2.0e-12:
        raise ValueError("Kerr polar sine lies outside its analytic range")
    sine_squared = min(1.0, sine_squared)
    sine = math.sqrt(sine_squared)
    phi_ks = math.atan2(y_m, x_m) - math.atan2(metric.spin_a_m, radius)
    cosine_phi = math.cos(phi_ks)
    sine_phi = math.sin(phi_ks)
    derivative_x_theta = (
        radius * cosine_phi - metric.spin_a_m * sine_phi
    ) * cosine
    derivative_y_theta = (
        radius * sine_phi + metric.spin_a_m * cosine_phi
    ) * cosine
    derivative_z_theta = -radius * sine
    p_theta = (
        state.covector[1] * derivative_x_theta
        + state.covector[2] * derivative_y_theta
        + state.covector[3] * derivative_z_theta
    )
    carter_q = p_theta * p_theta + cosine * cosine * (
        angular_momentum * angular_momentum / sine_squared
        - metric.spin_a_m * metric.spin_a_m * energy * energy
    )
    carter_k = carter_q + (
        angular_momentum - metric.spin_a_m * energy
    ) ** 2
    if not _finite((energy, angular_momentum, carter_q, carter_k)):
        raise ValueError("Kerr constants of motion must be finite")
    return KerrConstantsOfMotion(
        energy=energy,
        angular_momentum_z=angular_momentum,
        carter_q=carter_q,
        carter_k=carter_k,
    )
