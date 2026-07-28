#!/usr/bin/env python3
"""Independently verify the stationary Kerr remnant transfer map.

The generic contract validator checks schema, provenance, frames, hashes and
the 32-byte ABI.  This verifier adds Kerr-specific physics without importing
the production generator or reusing its adaptive integrator:

* a finite-distance BL-ZAMO shadow oracle from spherical photon orbits;
* full-image capture-mask and equatorial-reflection checks;
* an independent fixed-step RK4 integration of selected complete rays;
* fixed-step refinement, E/Lz/Q separation-constraint and infinity-tail checks;
* spin asymmetry, spin-reversal mirror and camera/ICRS axis checks.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

try:
    from scripts.verify_nr_contract import validate_contract
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from verify_nr_contract import validate_contract


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST: Final = (
    ROOT
    / "assets"
    / "transfer-maps"
    / "kerr-remnant-reference-v1"
    / "manifest.json"
)
RECORD: Final = struct.Struct("<7fBBH")

EXPECTED_SPIN: Final = 0.686461676493
EXPECTED_OBSERVER_RADIUS_M: Final = 40.0
EXPECTED_ESCAPE_RADIUS_M: Final = 1_000.0
EXPECTED_CAPTURE_RADIUS_M: Final = 1.747165982913406
EXPECTED_SHADOW_X_MIN: Final = -0.15952329
EXPECTED_SHADOW_X_MAX: Final = 0.08797226
EXPECTED_SHADOW_TOP_X: Final = -0.0342456
EXPECTED_SHADOW_TOP_Y: Final = 0.12771175

OUTCOME_ESCAPED: Final = 0
OUTCOME_CAPTURED: Final = 1
OUTCOME_UNRESOLVED: Final = 2
CAPTURE_BH: Final = 0
CAPTURE_NONE: Final = 255

SHADOW_ORACLE_SAMPLES: Final = 200_000
SHADOW_COORDINATE_TOLERANCE: Final = 2.0e-6
DIRECTION_TOLERANCE_RAD: Final = 2.0e-6
REFINEMENT_TOLERANCE_RAD: Final = 8.0e-7
FREQUENCY_TOLERANCE: Final = 2.0e-6
LOOKBACK_TOLERANCE_M: Final = 5.0e-4
AXIS_TOLERANCE: Final = 1.0e-12
NULL_RESIDUAL_LIMIT: Final = 1.0e-8
PROJECTION_ERROR_LIMIT_PX: Final = 0.25
FIXED_RK4_STEP: Final = 1.0e-5
COARSE_RK4_STEP: Final = 2.0e-5


@dataclass(frozen=True)
class PhysicsReport:
    width: int
    height: int
    records: int
    escaped: int
    captured: int
    unresolved: int
    shadow_x_min: float
    shadow_x_max: float
    shadow_top_x: float
    shadow_top_y: float
    analytic_mask_mismatches: int
    max_direction_norm_error: float
    max_vertical_symmetry_error: float
    max_independent_direction_error_rad: float
    max_independent_frequency_error: float
    max_independent_lookback_error_m: float
    max_fixed_step_refinement_rad: float
    max_independent_separation_residual: float
    max_boundary_richardson_error_rad: float
    direction_probe_count: int
    max_axis_mapping_error: float
    max_kerr_observer_identity_error: float
    remnant_spin_source_magnitude_error: float
    max_null_residual: float
    max_projection_error_px: float


@dataclass(frozen=True)
class _ShadowCurve:
    x: tuple[float, ...]
    upper_y: tuple[float, ...]
    x_min: float
    x_max: float
    top_x: float
    top_y: float


@dataclass(frozen=True)
class _OracleConstants:
    energy: float
    angular_momentum: float
    carter_q: float
    carter_k: float
    spin: float


@dataclass(frozen=True)
class _OracleRay:
    outcome: int
    direction_icrs: tuple[float, float, float]
    frequency_shift_g: float
    lookback_time_m: float
    max_separation_residual: float
    boundary_richardson_error_rad: float


def _fail(message: str) -> None:
    raise AssertionError(message)


def _screen_coordinates(
    x: int,
    y: int,
    width: int,
    height: int,
    vertical_fov_rad: float,
) -> tuple[float, float]:
    tangent = math.tan(0.5 * vertical_fov_rad)
    return (
        ((x + 0.5) / width * 2.0 - 1.0)
        * (width / height)
        * tangent,
        (1.0 - (y + 0.5) / height * 2.0) * tangent,
    )


def _bl_zamo_quantities(
    radius: float,
    theta: float,
    spin: float,
) -> tuple[float, float, float, float, float]:
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * radius + spin * spin
    big_a = (
        (radius * radius + spin * spin) ** 2
        - spin * spin * delta * sine * sine
    )
    lapse = math.sqrt(sigma * delta / big_a)
    omega = 2.0 * spin * radius / big_a
    return sigma, delta, big_a, lapse, omega


def _spherical_photon_constants(
    radius: float,
    spin: float,
) -> tuple[float, float]:
    """Return xi=Lz/E and eta=Q/E^2 for a spherical Kerr photon orbit."""
    xi = (
        radius * radius * (radius - 3.0)
        + spin * spin * (radius + 1.0)
    ) / (spin * (1.0 - radius))
    delta = radius * radius - 2.0 * radius + spin * spin
    eta = (
        4.0 * radius * radius * delta / (radius - 1.0) ** 2
        - (xi - spin) ** 2
    )
    return xi, eta


def _finite_zamo_screen_from_xi_eta(
    xi: float,
    eta: float,
    *,
    observer_radius: float,
    spin: float,
) -> tuple[float, float] | None:
    _sigma, _delta, big_a, lapse, omega = _bl_zamo_quantities(
        observer_radius,
        0.5 * math.pi,
        spin,
    )
    energy = -lapse / (1.0 - omega * xi)
    local_right = -xi * energy / math.sqrt(big_a / (observer_radius**2))
    local_up = math.sqrt(max(0.0, eta)) * abs(energy) / observer_radius
    forward_squared = 1.0 - local_right * local_right - local_up * local_up
    if forward_squared <= 0.0:
        return None
    local_forward = math.sqrt(forward_squared)
    return local_right / local_forward, local_up / local_forward


def _analytic_shadow_curve(
    spin: float,
    *,
    observer_radius: float = EXPECTED_OBSERVER_RADIUS_M,
    samples: int = SHADOW_ORACLE_SAMPLES,
) -> _ShadowCurve:
    if abs(spin) < 1.0e-14:
        critical_b = 3.0 * math.sqrt(3.0)
        alpha = math.asin(
            critical_b
            * math.sqrt(1.0 - 2.0 / observer_radius)
            / observer_radius
        )
        screen_radius = math.tan(alpha)
        points = [
            (
                -screen_radius + 2.0 * screen_radius * index / samples,
                math.sqrt(
                    max(
                        0.0,
                        screen_radius * screen_radius
                        - (
                            -screen_radius
                            + 2.0 * screen_radius * index / samples
                        )
                        ** 2,
                    )
                ),
            )
            for index in range(samples + 1)
        ]
    else:
        # Find both eta=0 equatorial photon-orbit endpoints first, rather than
        # allowing a sampling grid to bias the reported horizontal extrema.
        search_lower = 1.0 + math.sqrt(1.0 - spin * spin) + 1.0e-10
        search_upper = 4.0 + 2.0 * abs(spin)
        brackets: list[tuple[float, float]] = []
        previous_radius = search_lower
        _previous_xi, previous_eta = _spherical_photon_constants(
            previous_radius, spin
        )
        for index in range(1, 20_001):
            radius = (
                search_lower
                + (search_upper - search_lower) * index / 20_000
            )
            _xi, eta = _spherical_photon_constants(radius, spin)
            if (eta >= 0.0) != (previous_eta >= 0.0):
                brackets.append((previous_radius, radius))
            previous_radius = radius
            previous_eta = eta
        if len(brackets) != 2:
            _fail(
                "could not bracket both equatorial spherical-photon radii"
            )

        roots: list[float] = []
        for first, second in brackets:
            _xi, first_eta = _spherical_photon_constants(first, spin)
            for _iteration in range(100):
                middle = 0.5 * (first + second)
                _xi, middle_eta = _spherical_photon_constants(middle, spin)
                if (middle_eta >= 0.0) == (first_eta >= 0.0):
                    first = middle
                    first_eta = middle_eta
                else:
                    second = middle
            roots.append(0.5 * (first + second))
        lower, upper = sorted(roots)
        points: list[tuple[float, float]] = []
        for index in range(samples + 1):
            radius = lower + (upper - lower) * index / samples
            xi, eta = _spherical_photon_constants(radius, spin)
            if eta < 0.0:
                continue
            screen = _finite_zamo_screen_from_xi_eta(
                xi,
                eta,
                observer_radius=observer_radius,
                spin=spin,
            )
            if screen is not None:
                points.append(screen)
    if len(points) < 100:
        _fail("analytic Kerr shadow oracle produced too few boundary points")
    points.sort(key=lambda value: value[0])
    # x is monotonic on the physical upper branch.  Collapse any numerical
    # duplicates by keeping the larger upper envelope.
    collapsed_x: list[float] = []
    collapsed_y: list[float] = []
    for x_value, y_value in points:
        if collapsed_x and abs(x_value - collapsed_x[-1]) < 1.0e-14:
            collapsed_y[-1] = max(collapsed_y[-1], y_value)
        else:
            collapsed_x.append(x_value)
            collapsed_y.append(y_value)
    top_index = max(range(len(collapsed_y)), key=collapsed_y.__getitem__)
    x_min = collapsed_x[0]
    x_max = collapsed_x[-1]
    top_x = collapsed_x[top_index]
    top_y = collapsed_y[top_index]
    if abs(spin) >= 1.0e-14:
        # Golden-section maximize the finite-observer vertical coordinate.
        left = lower
        right = upper
        inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0

        def screen_at(radius: float) -> tuple[float, float]:
            xi, eta = _spherical_photon_constants(radius, spin)
            value = _finite_zamo_screen_from_xi_eta(
                xi,
                max(0.0, eta),
                observer_radius=observer_radius,
                spin=spin,
            )
            if value is None:
                _fail("top-shadow optimizer left the physical local sky")
            return value

        first = right - inverse_phi * (right - left)
        second = left + inverse_phi * (right - left)
        first_value = screen_at(first)
        second_value = screen_at(second)
        for _iteration in range(100):
            if first_value[1] < second_value[1]:
                left = first
                first = second
                first_value = second_value
                second = left + inverse_phi * (right - left)
                second_value = screen_at(second)
            else:
                right = second
                second = first
                second_value = first_value
                first = right - inverse_phi * (right - left)
                first_value = screen_at(first)
        top_x, top_y = screen_at(0.5 * (left + right))
    return _ShadowCurve(
        x=tuple(collapsed_x),
        upper_y=tuple(collapsed_y),
        x_min=x_min,
        x_max=x_max,
        top_x=top_x,
        top_y=top_y,
    )


def _shadow_upper_y(curve: _ShadowCurve, screen_x: float) -> float:
    if screen_x < curve.x_min or screen_x > curve.x_max:
        return -1.0
    upper = bisect.bisect_left(curve.x, screen_x)
    if upper <= 0:
        return curve.upper_y[0]
    if upper >= len(curve.x):
        return curve.upper_y[-1]
    lower = upper - 1
    span = curve.x[upper] - curve.x[lower]
    weight = (screen_x - curve.x[lower]) / span
    return (
        curve.upper_y[lower] * (1.0 - weight)
        + curve.upper_y[upper] * weight
    )


def _load_records(
    manifest_path: Path,
    manifest: dict[str, object],
) -> list[
    tuple[float, float, float, float, float, float, float, int, int, int]
]:
    records: list[
        tuple[float, float, float, float, float, float, float, int, int, int]
    ] = []
    for chunk in manifest["chunks"]:  # type: ignore[index]
        chunk_path = manifest_path.parent / chunk["uri"]  # type: ignore[index]
        payload = chunk_path.read_bytes()
        records.extend(RECORD.iter_unpack(payload))
    return records


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(first, second))


def _angular_separation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    return math.atan2(
        math.sqrt(_dot(cross, cross)),
        max(-1.0, min(1.0, _dot(first, second))),
    )


def _oracle_initial_state(
    screen_x: float,
    screen_y: float,
    spin: float,
) -> tuple[_OracleConstants, tuple[float, float, float, float, float, float]]:
    radius = EXPECTED_OBSERVER_RADIUS_M
    _sigma, delta, big_a, lapse, omega = _bl_zamo_quantities(
        radius,
        0.5 * math.pi,
        spin,
    )
    inverse_norm = 1.0 / math.sqrt(1.0 + screen_x**2 + screen_y**2)
    local_right = screen_x * inverse_norm
    local_up = screen_y * inverse_norm
    local_forward = inverse_norm

    # Independent local-frame derivation: for the past-directed ray,
    # Lz=-n_right*sqrt(g_phiphi) and -E+omega*Lz=alpha.
    angular_momentum = -local_right * math.sqrt(big_a / (radius * radius))
    energy = omega * angular_momentum - lapse
    p_theta = -radius * local_up
    carter_q = p_theta * p_theta
    carter_k = carter_q + (angular_momentum - spin * energy) ** 2
    velocity_u = local_forward * math.sqrt(delta) / radius
    velocity_theta = p_theta
    return (
        _OracleConstants(
            energy=energy,
            angular_momentum=angular_momentum,
            carter_q=carter_q,
            carter_k=carter_k,
            spin=spin,
        ),
        (
            1.0 / radius,
            velocity_u,
            0.5 * math.pi,
            velocity_theta,
            -math.atan2(spin, radius),
            0.0,
        ),
    )


def _oracle_derivative(
    state: Sequence[float],
    constants: _OracleConstants,
) -> tuple[float, float, float, float, float, float]:
    u, velocity_u, theta, velocity_theta, _phi, _time = state
    energy = constants.energy
    angular_momentum = constants.angular_momentum
    carter_k = constants.carter_k
    spin = constants.spin
    c_term = spin * (spin * energy - angular_momentum)
    denominator = 1.0 - 2.0 * u + spin * spin * u * u
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sine_safe = math.copysign(max(abs(sine), 1.0e-15), sine)
    acceleration_u = (
        (2.0 * energy * c_term - carter_k) * u
        + 3.0 * carter_k * u * u
        + 2.0 * (c_term * c_term - spin * spin * carter_k) * u**3
    )
    acceleration_theta = (
        -spin * spin * energy * energy * cosine * sine
        + angular_momentum * angular_momentum * cosine / sine_safe**3
    )
    p_scaled = energy + c_term * u * u
    phi_rate = (
        angular_momentum / (sine_safe * sine_safe)
        - spin * energy
        + spin * p_scaled / denominator
        - spin * velocity_u / denominator
    )
    if u <= 0.0:
        time_rate = 0.0
    else:
        radius = 1.0 / u
        time_bl = (
            spin
            * (angular_momentum - spin * energy * sine * sine)
            + (radius * radius + spin * spin)
            * p_scaled
            / denominator
        )
        time_rate = time_bl - 2.0 * velocity_u / (u * denominator)
    return (
        velocity_u,
        acceleration_u,
        velocity_theta,
        acceleration_theta,
        phi_rate,
        time_rate,
    )


def _rk4_step(
    state: Sequence[float],
    step: float,
    constants: _OracleConstants,
) -> tuple[float, ...]:
    k1 = _oracle_derivative(state, constants)
    second = tuple(
        state[index] + 0.5 * step * k1[index] for index in range(6)
    )
    k2 = _oracle_derivative(second, constants)
    third = tuple(
        state[index] + 0.5 * step * k2[index] for index in range(6)
    )
    k3 = _oracle_derivative(third, constants)
    fourth = tuple(
        state[index] + step * k3[index] for index in range(6)
    )
    k4 = _oracle_derivative(fourth, constants)
    return tuple(
        state[index]
        + step
        * (
            k1[index]
            + 2.0 * k2[index]
            + 2.0 * k3[index]
            + k4[index]
        )
        / 6.0
        for index in range(6)
    )


def _locate_event_rk4(
    start: Sequence[float],
    full_step_state: Sequence[float],
    target_u: float,
    step: float,
    constants: _OracleConstants,
) -> tuple[float, ...]:
    """Locate a radial event with partial RK4 steps, not linear t blending."""
    lower = 0.0
    upper = step
    start_sign = start[0] - target_u
    candidate = tuple(full_step_state)
    for _iteration in range(52):
        middle = 0.5 * (lower + upper)
        candidate = _rk4_step(start, middle, constants)
        candidate_sign = candidate[0] - target_u
        if candidate_sign == 0.0:
            break
        if (candidate_sign > 0.0) == (start_sign > 0.0):
            lower = middle
        else:
            upper = middle
    mutable = list(candidate)
    mutable[0] = target_u
    return tuple(mutable)


def _separation_residual(
    state: Sequence[float],
    constants: _OracleConstants,
) -> float:
    u, velocity_u, theta, velocity_theta, _phi, _time = state
    energy = constants.energy
    angular_momentum = constants.angular_momentum
    spin = constants.spin
    c_term = spin * (spin * energy - angular_momentum)
    radial = (
        energy * energy
        + (2.0 * energy * c_term - constants.carter_k) * u * u
        + 2.0 * constants.carter_k * u**3
        + (
            c_term * c_term - spin * spin * constants.carter_k
        )
        * u**4
    )
    sine = math.sin(theta)
    cosine = math.cos(theta)
    polar = (
        constants.carter_q
        + spin * spin * energy * energy * cosine * cosine
        - angular_momentum
        * angular_momentum
        * (cosine / max(abs(sine), 1.0e-15)) ** 2
    )
    return max(
        abs(velocity_u * velocity_u - radial),
        abs(velocity_theta * velocity_theta - polar),
    )


def _world_direction(theta: float, phi: float) -> tuple[float, float, float]:
    sine = math.sin(theta)
    return sine * math.cos(phi), sine * math.sin(phi), math.cos(theta)


def _icrs_direction(theta: float, phi: float) -> tuple[float, float, float]:
    world = _world_direction(theta, phi)
    # Kerr camera alignment: ICRS=(-world.y, world.x, world.z).
    direction = (-world[1], world[0], world[2])
    inverse_norm = 1.0 / math.sqrt(_dot(direction, direction))
    return tuple(value * inverse_norm for value in direction)


def _oracle_integrate(
    screen_x: float,
    screen_y: float,
    *,
    spin: float,
    step: float,
) -> _OracleRay:
    constants, state = _oracle_initial_state(screen_x, screen_y, spin)
    capture_u = 1.0 / (1.0 + math.sqrt(1.0 - spin * spin) + 0.02)
    escape_u = 1.0 / EXPECTED_ESCAPE_RADIUS_M
    half_escape_u = 0.5 * escape_u
    maximum_residual = _separation_residual(state, constants)
    boundary_state: tuple[float, ...] | None = None
    half_state: tuple[float, ...] | None = None
    turned = False

    for _iteration in range(200_000):
        candidate = _rk4_step(state, step, constants)
        maximum_residual = max(
            maximum_residual,
            _separation_residual(candidate, constants),
        )
        if candidate[1] < 0.0:
            turned = True
        if state[0] < capture_u <= candidate[0]:
            terminal = _locate_event_rk4(
                state, candidate, capture_u, step, constants
            )
            return _OracleRay(
                outcome=OUTCOME_CAPTURED,
                direction_icrs=(0.0, 0.0, 0.0),
                frequency_shift_g=0.0,
                lookback_time_m=max(0.0, -terminal[5]),
                max_separation_residual=maximum_residual,
                boundary_richardson_error_rad=0.0,
            )
        if (
            boundary_state is None
            and turned
            and state[0] > escape_u >= candidate[0]
        ):
            boundary_state = _locate_event_rk4(
                state, candidate, escape_u, step, constants
            )
        if (
            boundary_state is not None
            and half_state is None
            and state[0] > half_escape_u >= candidate[0]
        ):
            half_state = _locate_event_rk4(
                state, candidate, half_escape_u, step, constants
            )
        if half_state is not None and state[0] > 0.0 >= candidate[0]:
            infinity = _locate_event_rk4(
                state, candidate, 0.0, step, constants
            )
            direction = _icrs_direction(infinity[2], infinity[4])
            radius = EXPECTED_ESCAPE_RADIUS_M
            _sigma, _delta, _big_a, lapse, omega = _bl_zamo_quantities(
                radius,
                boundary_state[2],  # type: ignore[index]
                spin,
            )
            boundary_energy = (
                -constants.energy + omega * constants.angular_momentum
            ) / lapse
            delta_phi = math.remainder(
                half_state[4] - boundary_state[4],  # type: ignore[index]
                2.0 * math.pi,
            )
            richardson = _icrs_direction(
                2.0 * half_state[2] - boundary_state[2],  # type: ignore[index]
                half_state[4] + delta_phi,
            )
            return _OracleRay(
                outcome=OUTCOME_ESCAPED,
                direction_icrs=direction,
                frequency_shift_g=1.0 / boundary_energy,
                lookback_time_m=max(0.0, -boundary_state[5]),  # type: ignore[index]
                max_separation_residual=maximum_residual,
                boundary_richardson_error_rad=_angular_separation(
                    richardson,
                    direction,
                ),
            )
        state = candidate
    _fail("independent fixed-step Kerr ray exhausted iteration budget")


def _record_direction(
    record: Sequence[float | int],
) -> tuple[float, float, float]:
    return float(record[0]), float(record[1]), float(record[2])


def _validate_axis_mapping(manifest: dict[str, object]) -> float:
    sky = manifest["coordinates"]["sky"]  # type: ignore[index]
    matrix_values = sky["worldToIcrs"]  # type: ignore[index]
    matrix = [
        [float(value) for value in matrix_values[index : index + 3]]
        for index in range(0, 9, 3)
    ]

    def transform(vector: Sequence[float]) -> tuple[float, float, float]:
        return tuple(_dot(row, vector) for row in matrix)

    expected = (
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    )
    maximum = 0.0
    for world, icrs in expected:
        mapped = transform(world)
        maximum = max(
            maximum,
            max(abs(mapped[index] - icrs[index]) for index in range(3)),
        )
    if maximum > AXIS_TOLERANCE:
        _fail(f"camera-aligned ICRS axis mapping error {maximum:.3e}")
    return maximum


def _kerr_radius_from_cartesian(
    x: float,
    y: float,
    z: float,
    spin: float,
) -> float:
    rho_squared = x * x + y * y + z * z
    difference = rho_squared - spin * spin
    radius_squared = 0.5 * (
        difference
        + math.sqrt(difference * difference + 4.0 * spin * spin * z * z)
    )
    return math.sqrt(max(0.0, radius_squared))


def _validate_kerr_observer_identity(
    manifest: dict[str, object],
    spin: float,
) -> float:
    """Prove the declared sample is the stated KS Kerr metric and BL ZAMO."""
    sample = manifest["observer"]["samples"][0]  # type: ignore[index]
    event = [float(value) for value in sample["eventNr"]]  # type: ignore[index]
    _time, x, y, z = event
    radius = _kerr_radius_from_cartesian(x, y, z, spin)
    if abs(radius - EXPECTED_OBSERVER_RADIUS_M) > 1.0e-11:
        _fail(f"observer Kerr radius is {radius:.15g}M, not 40M")
    if abs(float(manifest["escapeBoundary"]["surface"]["radiusM"]) - 1_000.0) > 0:
        _fail("escape boundary Kerr radius must be exactly 1000M")

    sigma = radius * radius + spin * spin * (z / radius) ** 2
    h = radius / sigma
    l_spatial = (
        (radius * x + spin * y) / (radius * radius + spin * spin),
        (radius * y - spin * x) / (radius * radius + spin * spin),
        z / radius,
    )
    expected_covariant = [[0.0] * 4 for _ in range(4)]
    expected_covariant[0][0] = -1.0 + 2.0 * h
    expected_contravariant = [[0.0] * 4 for _ in range(4)]
    expected_contravariant[0][0] = -1.0 - 2.0 * h
    for first in range(3):
        expected_covariant[0][first + 1] = 2.0 * h * l_spatial[first]
        expected_covariant[first + 1][0] = 2.0 * h * l_spatial[first]
        expected_contravariant[0][first + 1] = (
            2.0 * h * l_spatial[first]
        )
        expected_contravariant[first + 1][0] = (
            2.0 * h * l_spatial[first]
        )
        for second in range(3):
            identity = 1.0 if first == second else 0.0
            expected_covariant[first + 1][second + 1] = (
                identity
                + 2.0 * h * l_spatial[first] * l_spatial[second]
            )
            expected_contravariant[first + 1][second + 1] = (
                identity
                - 2.0 * h * l_spatial[first] * l_spatial[second]
            )

    stored_covariant = [
        float(value) for value in sample["metricCovariantNr"]  # type: ignore[index]
    ]
    stored_contravariant = [
        float(value)
        for value in sample["metricContravariantNr"]  # type: ignore[index]
    ]
    maximum = 0.0
    for row in range(4):
        for column in range(4):
            maximum = max(
                maximum,
                abs(
                    stored_covariant[4 * row + column]
                    - expected_covariant[row][column]
                ),
                abs(
                    stored_contravariant[4 * row + column]
                    - expected_contravariant[row][column]
                ),
            )

    # Reconstruct the BL ZAMO and orthonormal axes, then independently apply
    # the BL -> ingoing Cartesian-KS Jacobian at the sample event.
    theta = math.acos(z / radius)
    phi_ks = math.atan2(y, x) - math.atan2(spin, radius)
    sigma_bl, delta, big_a, lapse, omega = _bl_zamo_quantities(
        radius, theta, spin
    )
    sine = math.sin(theta)
    cosine = math.cos(theta)
    cos_phi = math.cos(phi_ks)
    sin_phi = math.sin(phi_ks)
    reconstructed_x = (radius * cos_phi - spin * sin_phi) * sine
    reconstructed_y = (radius * sin_phi + spin * cos_phi) * sine
    dt_dr = 2.0 * radius / delta
    dphi_dr = spin / delta
    dx_dr = (
        cos_phi - (radius * sin_phi + spin * cos_phi) * dphi_dr
    ) * sine
    dy_dr = (
        sin_phi + (radius * cos_phi - spin * sin_phi) * dphi_dr
    ) * sine
    dx_dtheta = (radius * cos_phi - spin * sin_phi) * cosine
    dy_dtheta = (radius * sin_phi + spin * cos_phi) * cosine
    dz_dtheta = -radius * sine

    def transform_bl(vector: Sequence[float]) -> list[float]:
        v_t, v_r, v_theta, v_phi = vector
        return [
            v_t + dt_dr * v_r,
            dx_dr * v_r + dx_dtheta * v_theta - reconstructed_y * v_phi,
            dy_dr * v_r + dy_dtheta * v_theta + reconstructed_x * v_phi,
            cosine * v_r + dz_dtheta * v_theta,
        ]

    u = transform_bl((1.0 / lapse, 0.0, 0.0, omega / lapse))
    e_r = transform_bl((0.0, math.sqrt(delta / sigma_bl), 0.0, 0.0))
    e_theta = transform_bl((0.0, 0.0, 1.0 / math.sqrt(sigma_bl), 0.0))
    e_phi = transform_bl(
        (0.0, 0.0, 0.0, math.sqrt(sigma_bl / big_a) / sine)
    )
    expected_tetrad = [
        u,
        [-value for value in e_phi],
        [-value for value in e_theta],
        [-value for value in e_r],
    ]
    stored_tetrad = sample["tetradContravariantNr"]  # type: ignore[index]
    stored_u = sample["fourVelocityContravariantNr"]  # type: ignore[index]
    for index in range(4):
        maximum = max(maximum, abs(float(stored_u[index]) - u[index]))
        for component in range(4):
            maximum = max(
                maximum,
                abs(
                    float(stored_tetrad[index][component])  # type: ignore[index]
                    - expected_tetrad[index][component]
                ),
            )

    camera_to_world = [
        float(value) for value in manifest["camera"]["cameraToWorld"]  # type: ignore[index]
    ]
    expected_camera = [
        0.0, 0.0, -1.0, x,
        -1.0, 0.0, 0.0, y,
        0.0, 1.0, 0.0, z,
        0.0, 0.0, 0.0, 1.0,
    ]
    maximum = max(
        maximum,
        max(
            abs(actual - expected)
            for actual, expected in zip(camera_to_world, expected_camera)
        ),
    )
    if maximum > 1.0e-11:
        _fail(
            "declared Cartesian-KS metric/BL-ZAMO/camera identity error "
            f"{maximum:.3e}"
        )
    return maximum


def validate_kerr_physics(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    oracle_probe_limit: int = 8,
) -> PhysicsReport:
    contract = validate_contract(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["datasetKind"] != "stationary-reference-transfer-map":
        _fail("Kerr verifier requires a stationary reference transfer map")
    spin_vector = manifest["physicalSystem"]["dimensionlessSpins"][0]["vector"]
    spin = float(spin_vector[2])
    if abs(spin - EXPECTED_SPIN) > 5.0e-13:
        _fail(f"unexpected Kerr spin {spin:.15g}")
    if manifest["coordinates"]["nrChart"]["coordinates"] != (
        "Cartesian Kerr-Schild (t_KS,x,y,z)"
    ):
        _fail("manifest must declare the Cartesian Kerr-Schild chart")
    if manifest["escapeBoundary"]["surface"]["kind"] != (
        "constant-Kerr-r-oblate-worldtube"
    ):
        _fail("escape boundary must use the Kerr-r oblate-worldtube kind")
    if manifest["escapeBoundary"]["referenceObserver"]["kind"] != (
        "Boyer-Lindquist-ZAMO"
    ):
        _fail("frequency reference must be the Boyer-Lindquist ZAMO")
    spin_artifacts = [
        artifact
        for artifact in manifest["provenance"]["sourceArtifacts"]
        if artifact["role"] == "remnant-spin-source"
    ]
    if len(spin_artifacts) != 1:
        _fail("manifest needs exactly one remnant-spin-source artifact")
    spin_uri = Path(spin_artifacts[0]["uri"])
    spin_path = (ROOT / spin_uri).resolve()
    if not spin_path.is_relative_to(ROOT.resolve()) or not spin_path.is_file():
        _fail("remnant-spin-source URI is not a bundled repository file")
    spin_document = json.loads(spin_path.read_text(encoding="utf-8"))
    source_vector = [
        float(value)
        for value in spin_document["physicalSystem"]["remnant"][
            "dimensionlessSpin"
        ]
    ]
    if len(source_vector) != 3 or any(
        not math.isfinite(value) for value in source_vector
    ):
        _fail("remnant-spin-source does not contain a finite vector3")
    source_magnitude = math.sqrt(_dot(source_vector, source_vector))
    spin_source_error = abs(source_magnitude - spin)
    if spin_source_error > 5.0e-13:
        _fail(
            "manifest Kerr magnitude is not bound to the pinned remnant vector"
        )
    description = manifest["physicalSystem"]["description"]
    if "rigidly rotated (not component-truncated) onto world +Z" not in description:
        _fail("manifest does not disclose remnant-spin frame alignment")
    if source_vector[0] == 0.0 or source_vector[1] == 0.0:
        _fail("source-spin audit expected the pinned tiny transverse components")
    capture_description = manifest["captureTargets"][0]["description"]
    if f"{EXPECTED_CAPTURE_RADIUS_M:.15g}" not in capture_description:
        _fail("stretched-horizon Kerr radius is absent from capture description")

    width = int(manifest["projection"]["widthPixels"])
    height = int(manifest["projection"]["heightPixels"])
    vertical_fov = float(manifest["projection"]["verticalFieldOfViewRad"])
    records = _load_records(manifest_path, manifest)
    if len(records) != width * height:
        _fail("record count does not match projection dimensions")

    curve = _analytic_shadow_curve(spin)
    for actual, expected, name in (
        (curve.x_min, EXPECTED_SHADOW_X_MIN, "left"),
        (curve.x_max, EXPECTED_SHADOW_X_MAX, "right"),
        (curve.top_x, EXPECTED_SHADOW_TOP_X, "top-x"),
        (curve.top_y, EXPECTED_SHADOW_TOP_Y, "top-y"),
    ):
        if abs(actual - expected) > SHADOW_COORDINATE_TOLERANCE:
            _fail(
                f"finite-ZAMO analytic shadow {name}={actual:.9g}, "
                f"expected {expected:.9g}"
            )
    mirror = _analytic_shadow_curve(-spin)
    if (
        abs(mirror.x_min + curve.x_max) > SHADOW_COORDINATE_TOLERANCE
        or abs(mirror.x_max + curve.x_min) > SHADOW_COORDINATE_TOLERANCE
        or abs(mirror.top_x + curve.top_x) > SHADOW_COORDINATE_TOLERANCE
        or abs(mirror.top_y - curve.top_y) > SHADOW_COORDINATE_TOLERANCE
    ):
        _fail("analytic a->-a shadow does not mirror across screen X")

    escaped = captured = unresolved = 0
    analytic_mismatches = 0
    maximum_norm_error = 0.0
    maximum_null = 0.0
    maximum_projection = 0.0
    for index, record in enumerate(records):
        outcome = int(record[7])
        if outcome == OUTCOME_ESCAPED:
            escaped += 1
            direction = _record_direction(record)
            maximum_norm_error = max(
                maximum_norm_error,
                abs(math.sqrt(_dot(direction, direction)) - 1.0),
            )
        elif outcome == OUTCOME_CAPTURED:
            captured += 1
            if int(record[8]) != CAPTURE_BH:
                _fail("captured Kerr ray has the wrong target code")
        else:
            unresolved += 1
        maximum_null = max(maximum_null, float(record[5]))
        maximum_projection = max(maximum_projection, float(record[6]))

        x = index % width
        y = index // width
        screen_x, screen_y = _screen_coordinates(
            x, y, width, height, vertical_fov
        )
        expected_captured = abs(screen_y) < _shadow_upper_y(curve, screen_x)
        if (outcome == OUTCOME_CAPTURED) != expected_captured:
            analytic_mismatches += 1
    if analytic_mismatches:
        _fail(
            f"{analytic_mismatches} texels disagree with finite-ZAMO "
            "spherical-photon shadow oracle"
        )
    if unresolved:
        _fail(f"bundled Kerr reference contains {unresolved} unresolved rays")
    if maximum_norm_error > 1.0e-6:
        _fail(f"maximum stored direction norm error {maximum_norm_error:.3e}")
    if maximum_null > NULL_RESIDUAL_LIMIT:
        _fail(f"stored null residual {maximum_null:.3e} exceeds 1e-8")
    if maximum_projection > PROJECTION_ERROR_LIMIT_PX:
        _fail(
            f"stored projection error {maximum_projection:.3e} exceeds 0.25px"
        )

    maximum_symmetry = 0.0
    for y in range(height // 2):
        mirror_y = height - 1 - y
        for x in range(width):
            upper = records[y * width + x]
            lower = records[mirror_y * width + x]
            if int(upper[7]) != int(lower[7]):
                _fail("equatorial reflection changed a ray outcome")
            if int(upper[7]) == OUTCOME_ESCAPED:
                expected = (float(upper[0]), float(upper[1]), -float(upper[2]))
                maximum_symmetry = max(
                    maximum_symmetry,
                    _angular_separation(expected, _record_direction(lower)),
                )
                maximum_symmetry = max(
                    maximum_symmetry,
                    abs(float(upper[3]) - float(lower[3])),
                )
    if maximum_symmetry > 2.0e-6:
        _fail(f"equatorial reflection error {maximum_symmetry:.3e}")

    candidate_pixels = [
        (round(width * x_fraction - 0.5), round(height * y_fraction - 0.5))
        for y_fraction in (0.18, 0.5, 0.82)
        # Avoid the BL polar-coordinate axis Lz≈0 in the independent
        # fixed-step oracle.  The production solver handles it, but an
        # axis-crossing probe would test coordinate cancellation rather than
        # provide a clean E/Lz/Q conservation oracle.
        for x_fraction in (0.15, 0.35, 0.65, 0.85)
    ]
    probes: list[tuple[int, int]] = []
    for x, y in candidate_pixels:
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        if records[y * width + x][7] == OUTCOME_ESCAPED:
            probes.append((x, y))
        if len(probes) >= oracle_probe_limit:
            break
    if len(probes) < min(6, oracle_probe_limit):
        _fail("not enough escaped pixels for independent geodesic probes")

    max_direction_error = 0.0
    max_frequency_error = 0.0
    max_lookback_error = 0.0
    max_refinement = 0.0
    max_separation = 0.0
    max_boundary_richardson = 0.0
    for x, y in probes:
        screen_x, screen_y = _screen_coordinates(
            x, y, width, height, vertical_fov
        )
        fine = _oracle_integrate(
            screen_x,
            screen_y,
            spin=spin,
            step=FIXED_RK4_STEP,
        )
        coarse = _oracle_integrate(
            screen_x,
            screen_y,
            spin=spin,
            step=COARSE_RK4_STEP,
        )
        if fine.outcome != OUTCOME_ESCAPED or coarse.outcome != OUTCOME_ESCAPED:
            _fail("independent probe outcome was not escaped")
        record = records[y * width + x]
        max_direction_error = max(
            max_direction_error,
            _angular_separation(
                fine.direction_icrs,
                _record_direction(record),
            ),
        )
        max_frequency_error = max(
            max_frequency_error,
            abs(fine.frequency_shift_g - float(record[3])),
        )
        max_lookback_error = max(
            max_lookback_error,
            abs(fine.lookback_time_m - float(record[4])),
        )
        max_refinement = max(
            max_refinement,
            _angular_separation(
                fine.direction_icrs,
                coarse.direction_icrs,
            ),
        )
        max_separation = max(
            max_separation,
            fine.max_separation_residual,
        )
        max_boundary_richardson = max(
            max_boundary_richardson,
            fine.boundary_richardson_error_rad,
        )
    if max_direction_error > DIRECTION_TOLERANCE_RAD:
        _fail(
            f"independent direction error {max_direction_error:.3e} rad"
        )
    if max_frequency_error > FREQUENCY_TOLERANCE:
        _fail(f"independent frequency error {max_frequency_error:.3e}")
    if max_lookback_error > LOOKBACK_TOLERANCE_M:
        _fail(f"independent KS lookback error {max_lookback_error:.3e}M")
    if max_refinement > REFINEMENT_TOLERANCE_RAD:
        _fail(f"fixed-step refinement error {max_refinement:.3e} rad")
    if max_separation > 1.0e-8:
        _fail(f"independent E/Lz/Q separation residual {max_separation:.3e}")
    if max_boundary_richardson > 1.0e-4:
        _fail(
            "infinity continuation boundary-doubling error "
            f"{max_boundary_richardson:.3e} rad"
        )

    maximum_axis = _validate_axis_mapping(manifest)
    maximum_kerr_identity = _validate_kerr_observer_identity(manifest, spin)
    declared = manifest["accuracy"]["outcomeFractions"]
    for name, count in (
        ("escaped", escaped),
        ("captured", captured),
        ("unresolved", unresolved),
    ):
        if abs(float(declared[name]) - count / len(records)) > 1.0e-15:
            _fail(f"declared {name} fraction does not match binary records")
    if contract["records"] != len(records):
        _fail("contract report and Kerr verifier disagree on record count")

    return PhysicsReport(
        width=width,
        height=height,
        records=len(records),
        escaped=escaped,
        captured=captured,
        unresolved=unresolved,
        shadow_x_min=curve.x_min,
        shadow_x_max=curve.x_max,
        shadow_top_x=curve.top_x,
        shadow_top_y=curve.top_y,
        analytic_mask_mismatches=analytic_mismatches,
        max_direction_norm_error=maximum_norm_error,
        max_vertical_symmetry_error=maximum_symmetry,
        max_independent_direction_error_rad=max_direction_error,
        max_independent_frequency_error=max_frequency_error,
        max_independent_lookback_error_m=max_lookback_error,
        max_fixed_step_refinement_rad=max_refinement,
        max_independent_separation_residual=max_separation,
        max_boundary_richardson_error_rad=max_boundary_richardson,
        direction_probe_count=len(probes),
        max_axis_mapping_error=maximum_axis,
        max_kerr_observer_identity_error=maximum_kerr_identity,
        remnant_spin_source_magnitude_error=spin_source_error,
        max_null_residual=maximum_null,
        max_projection_error_px=maximum_projection,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--probe-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = validate_kerr_physics(
        arguments.manifest.resolve(),
        oracle_probe_limit=arguments.probe_limit,
    )
    print("Kerr stationary physics checks passed")
    print(
        f"  resolution={report.width}x{report.height}, records={report.records}; "
        f"outcomes=escaped:{report.escaped}, captured:{report.captured}, "
        f"unresolved:{report.unresolved}"
    )
    print(
        "  finite-ZAMO shadow screen extrema="
        f"[{report.shadow_x_min:.9f}, {report.shadow_x_max:.9f}], "
        f"top=({report.shadow_top_x:.9f}, {report.shadow_top_y:.9f})"
    )
    print(
        f"  analytic mask mismatches={report.analytic_mask_mismatches}; "
        f"max direction norm error={report.max_direction_norm_error:.3e}"
    )
    print(
        f"  independent probes={report.direction_probe_count}; "
        f"direction={report.max_independent_direction_error_rad:.3e} rad, "
        f"frequency={report.max_independent_frequency_error:.3e}, "
        f"lookback={report.max_independent_lookback_error_m:.3e}M"
    )
    print(
        f"  fixed-step refinement={report.max_fixed_step_refinement_rad:.3e} rad; "
        "E/Lz/Q separation residual="
        f"{report.max_independent_separation_residual:.3e}; "
        "boundary Richardson="
        f"{report.max_boundary_richardson_error_rad:.3e} rad"
    )
    print(
        f"  stored max null/projection={report.max_null_residual:.3e}/"
        f"{report.max_projection_error_px:.3e}px; "
        f"axis error={report.max_axis_mapping_error:.3e}; "
        "KS metric/ZAMO identity="
        f"{report.max_kerr_observer_identity_error:.3e}; "
        "remnant-spin magnitude binding="
        f"{report.remnant_spin_source_magnitude_error:.3e}"
    )


if __name__ == "__main__":
    main()
