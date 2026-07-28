#!/usr/bin/env python3
"""Generate the stationary Kerr remnant reference transfer map.

The source spacetime is the exact vacuum Kerr solution with ``M=1`` and
``a/M=0.686461676493``.  Rays are initialized in the orthonormal tetrad of a
zero-angular-momentum observer (ZAMO) at Boyer-Lindquist ``r=40M`` in the
equatorial plane.  The camera looks radially inward.

The geodesic production path is deliberately independent of the browser:

* conserved ``E=-p_t``, ``Lz=p_phi`` and Carter ``Q`` are formed in float64;
* the separated Kerr Hamilton-Jacobi equations are integrated in Mino time
  with an adaptive Dormand-Prince 5(4) method;
* capture means crossing the constant-Kerr-r stretched horizon
  ``r=r_+ + 0.02M``;
* escape time and frequency shift are measured at ``r=1000M`` by a ZAMO;
* escaped rays are then continued through the exact Kerr equations to
  ``r=infinity`` before their ICRS direction is stored.

The manifest chart is ingoing Cartesian Kerr-Schild.  Observer and tetrad
components are transformed from the physical Boyer-Lindquist ZAMO frame into
that chart.  This is an analytic stationary reference, not NR or GRMHD data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

try:
    from scripts.generate_nr_contract_fixture import manifest as fixture_manifest
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from generate_nr_contract_fixture import manifest as fixture_manifest


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA_PATH: Final = ROOT / "schemas" / "nr-transfer-map-v1.schema.json"
HELPER_PATH: Final = ROOT / "scripts" / "generate_nr_contract_fixture.py"
REMNANT_SPIN_SOURCE_PATH: Final = (
    ROOT / "assets" / "scenes" / "binary-sxs-bbh-0001-v2.json"
)
DEFAULT_OUTPUT_DIR: Final = (
    ROOT / "assets" / "transfer-maps" / "kerr-remnant-reference-v1"
)

MASS_M: Final = 1.0
SPIN_A_M: Final = 0.686461676493
OBSERVER_RADIUS_M: Final = 40.0
ESCAPE_RADIUS_M: Final = 1_000.0
STRETCHED_HORIZON_OFFSET_M: Final = 0.02
VERTICAL_FOV_RAD: Final = math.radians(40.0)
WIDTH: Final = 1024
HEIGHT: Final = 576
TILE_HEIGHT: Final = 64

FINE_ABSOLUTE_TOLERANCE: Final = 2.0e-12
FINE_RELATIVE_TOLERANCE: Final = 2.0e-11
COARSE_ABSOLUTE_TOLERANCE: Final = 2.0e-8
COARSE_RELATIVE_TOLERANCE: Final = 2.0e-8
MAX_ACCEPTED_STEPS: Final = 20_000
FLOAT32_DIRECTION_ERROR_RAD: Final = 2.0e-7
PROJECTION_CONVERGENCE_GATE_PX: Final = 0.25

RECORD: Final = struct.Struct("<7fBBH")
RECORD_BYTES: Final = RECORD.size
OUTCOME_ESCAPED: Final = 0
OUTCOME_CAPTURED: Final = 1
OUTCOME_UNRESOLVED: Final = 2
CAPTURE_BH: Final = 0
CAPTURE_NONE: Final = 255
VALID_DIRECTION: Final = 1 << 0
VALID_FREQUENCY_SHIFT: Final = 1 << 1
VALID_COORDINATE_LOOKBACK_TIME: Final = 1 << 2
VALID_NULL_RESIDUAL: Final = 1 << 3
VALID_PROJECTION_ERROR: Final = 1 << 4
VALID_ALL: Final = (
    VALID_DIRECTION
    | VALID_FREQUENCY_SHIFT
    | VALID_COORDINATE_LOOKBACK_TIME
    | VALID_NULL_RESIDUAL
    | VALID_PROJECTION_ERROR
)
VALID_CAPTURED: Final = (
    VALID_COORDINATE_LOOKBACK_TIME
    | VALID_NULL_RESIDUAL
    | VALID_PROJECTION_ERROR
)
VALID_UNRESOLVED: Final = VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR

# Camera-aligned ICRS convention matching the Schwarzschild product: optical
# centre is RA=270 deg, camera-right is ICRS +X, and camera-up is ICRS north.
# For the Kerr camera right=-world Y, up=+world Z, forward=-world X.
WORLD_TO_ICRS: Final = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)
ICRS_TO_WORLD: Final = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class Constants:
    energy: float
    angular_momentum_z: float
    carter_q: float
    carter_k: float
    spin: float


@dataclass(frozen=True)
class RaySolution:
    outcome: int
    escape_direction_icrs: tuple[float, float, float]
    frequency_shift_g: float
    coordinate_lookback_time_m: float
    null_residual: float
    projection_error_px: float
    boundary_continuation_error_rad: float
    accepted_steps: int


@dataclass(frozen=True)
class GenerationReport:
    width: int
    height: int
    chunks: int
    records: int
    escaped: int
    captured: int
    unresolved: int
    max_null_residual: float
    p95_projection_error_px: float
    max_projection_error_px: float
    p95_boundary_continuation_error_rad: float
    max_boundary_continuation_error_rad: float
    max_accepted_steps: int
    elapsed_seconds: float


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_artifact(role: str, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "role": role,
        "storage": "bundled",
        "uri": path.relative_to(ROOT).as_posix(),
        "byteLength": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _remnant_spin_source_vector() -> tuple[float, float, float]:
    document = json.loads(REMNANT_SPIN_SOURCE_PATH.read_text(encoding="utf-8"))
    values = document["physicalSystem"]["remnant"]["dimensionlessSpin"]
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError("remnant spin source does not contain a finite vector3")
    return tuple(float(value) for value in values)


def horizon_radius_m(spin: float = SPIN_A_M) -> float:
    if abs(spin) > MASS_M:
        raise ValueError("Kerr spin must satisfy |a| <= M")
    return MASS_M + math.sqrt(MASS_M * MASS_M - spin * spin)


def capture_radius_m(spin: float = SPIN_A_M) -> float:
    return horizon_radius_m(spin) + STRETCHED_HORIZON_OFFSET_M


def equatorial_shadow_intercepts(
    spin: float = SPIN_A_M,
    observer_radius_m: float = OBSERVER_RADIUS_M,
) -> tuple[float, float]:
    """Exact finite-ZAMO horizontal shadow intercepts in screen coordinates."""
    if abs(spin) < 1.0e-14:
        critical_impact = 3.0 * math.sqrt(3.0)
        alpha = math.asin(
            critical_impact
            * math.sqrt(1.0 - 2.0 / observer_radius_m)
            / observer_radius_m
        )
        radius = math.tan(alpha)
        return -radius, radius

    _sigma, _delta, big_a, lapse, omega = _kerr_bl_quantities(
        observer_radius_m,
        0.5 * math.pi,
        spin,
    )
    radii = (
        2.0
        * (
            1.0
            + math.cos(
                (2.0 / 3.0) * math.acos(-spin)
            )
        ),
        2.0
        * (
            1.0
            + math.cos(
                (2.0 / 3.0) * math.acos(spin)
            )
        ),
    )
    intercepts: list[float] = []
    for photon_radius in radii:
        xi = (
            photon_radius * photon_radius * (photon_radius - 3.0)
            + spin * spin * (photon_radius + 1.0)
        ) / (spin * (1.0 - photon_radius))
        energy = -lapse / (1.0 - omega * xi)
        local_right = -xi * energy / math.sqrt(
            big_a / (observer_radius_m * observer_radius_m)
        )
        local_forward = math.sqrt(max(0.0, 1.0 - local_right * local_right))
        intercepts.append(local_right / local_forward)
    return min(intercepts), max(intercepts)


def _is_equatorial_separatrix(
    screen_x: float,
    screen_y: float,
    spin: float,
) -> bool:
    if screen_y != 0.0:
        return False
    for intercept in equatorial_shadow_intercepts(spin):
        tolerance = 8.0 * max(math.ulp(intercept), math.ulp(screen_x))
        if abs(screen_x - intercept) <= tolerance:
            return True
    return False


def _kerr_bl_quantities(
    radius: float,
    theta: float,
    spin: float,
) -> tuple[float, float, float, float, float]:
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * MASS_M * radius + spin * spin
    big_a = (
        (radius * radius + spin * spin) ** 2
        - spin * spin * delta * sine * sine
    )
    lapse = math.sqrt(sigma * delta / big_a)
    omega = 2.0 * MASS_M * spin * radius / big_a
    return sigma, delta, big_a, lapse, omega


def _initial_constants_and_state(
    screen_x: float,
    screen_y: float,
    spin: float,
) -> tuple[Constants, tuple[float, float, float, float, float, float]]:
    """Return constants and ``(u,u',theta,theta',phi_KS,t_KS)``.

    Local components follow the manifest convention
    ``k^(a)=(-1, normalize(screenX,screenY,1))`` in the basis
    ``(time,right,up,forward)``.  At the +X equatorial observer those axes are
    ``right=-e_phi``, ``up=-e_theta=+Z`` and ``forward=-e_r``.
    """
    radius = OBSERVER_RADIUS_M
    theta = 0.5 * math.pi
    sigma, delta, big_a, lapse, omega = _kerr_bl_quantities(
        radius, theta, spin
    )
    inverse_norm = 1.0 / math.sqrt(
        1.0 + screen_x * screen_x + screen_y * screen_y
    )
    local_right = screen_x * inverse_norm
    local_up = screen_y * inverse_norm
    local_forward = inverse_norm

    e_r = math.sqrt(delta / sigma)
    e_theta = 1.0 / math.sqrt(sigma)
    e_phi = math.sqrt(sigma / big_a)

    k_t_contra = -1.0 / lapse
    k_r_contra = -local_forward * e_r
    k_theta_contra = -local_up * e_theta
    k_phi_contra = -omega / lapse - local_right * e_phi

    sine2 = 1.0
    g_tt = -(1.0 - 2.0 * radius / sigma)
    g_t_phi = -2.0 * spin * radius * sine2 / sigma
    g_phi_phi = big_a * sine2 / sigma
    p_t = g_tt * k_t_contra + g_t_phi * k_phi_contra
    p_phi = g_t_phi * k_t_contra + g_phi_phi * k_phi_contra
    p_theta = sigma * k_theta_contra
    energy = -p_t
    angular_momentum = p_phi
    carter_q = p_theta * p_theta
    carter_k = carter_q + (angular_momentum - spin * energy) ** 2

    # Mino derivatives: dx/dgamma = Sigma dx/dlambda.
    radial_mino = sigma * k_r_contra
    inverse_radius = 1.0 / radius
    inverse_radius_mino = -radial_mino * inverse_radius * inverse_radius
    theta_mino = sigma * k_theta_contra

    # Ingoing Kerr-Schild azimuth.  Choose the integration constant so the
    # observer lies exactly on the positive Cartesian X axis.
    phi_ks = -math.atan2(spin, radius)
    state = (
        inverse_radius,
        inverse_radius_mino,
        theta,
        theta_mino,
        phi_ks,
        0.0,
    )
    constants = Constants(
        energy=energy,
        angular_momentum_z=angular_momentum,
        carter_q=carter_q,
        carter_k=carter_k,
        spin=spin,
    )
    return constants, state


def _radial_potential_u(u: float, constants: Constants) -> float:
    energy = constants.energy
    angular_momentum = constants.angular_momentum_z
    spin = constants.spin
    carter_k = constants.carter_k
    c_term = spin * (spin * energy - angular_momentum)
    return (
        energy * energy
        + (2.0 * energy * c_term - carter_k) * u * u
        + 2.0 * carter_k * u * u * u
        + (c_term * c_term - spin * spin * carter_k) * u**4
    )


def _theta_potential(theta: float, constants: Constants) -> float:
    sine = math.sin(theta)
    cosine = math.cos(theta)
    if abs(sine) < 1.0e-15:
        if constants.angular_momentum_z != 0.0:
            return -math.inf
        return constants.carter_q + (
            constants.spin * constants.energy * cosine
        ) ** 2
    return (
        constants.carter_q
        + (constants.spin * constants.energy * cosine) ** 2
        - constants.angular_momentum_z**2 * (cosine / sine) ** 2
    )


def _derivatives(
    state: Sequence[float],
    constants: Constants,
    *,
    accumulate_time: bool,
) -> tuple[float, float, float, float, float, float]:
    u, velocity_u, theta, velocity_theta, _phi_ks, _time_ks = state
    energy = constants.energy
    angular_momentum = constants.angular_momentum_z
    spin = constants.spin
    carter_k = constants.carter_k
    c_term = spin * (spin * energy - angular_momentum)

    acceleration_u = (
        (2.0 * energy * c_term - carter_k) * u
        + 3.0 * carter_k * u * u
        + 2.0 * (c_term * c_term - spin * spin * carter_k) * u**3
    )
    sine = math.sin(theta)
    cosine = math.cos(theta)
    sine_safe = math.copysign(max(abs(sine), 1.0e-15), sine)
    acceleration_theta = (
        -spin * spin * energy * energy * cosine * sine
        + angular_momentum * angular_momentum * cosine / sine_safe**3
    )

    denominator = 1.0 - 2.0 * u + spin * spin * u * u
    p_scaled = energy + c_term * u * u
    phi_bl_rate = (
        angular_momentum / (sine_safe * sine_safe)
        - spin * energy
        + spin * p_scaled / denominator
    )
    # d(phi_KS)/dgamma=d(phi_BL)/dgamma+(a/Delta)dr/dgamma.
    # With u=1/r, the correction is exactly -a*u'/D and remains regular at
    # u=0; avoid reconstructing the individually divergent dr/dgamma there.
    phi_ks_rate = phi_bl_rate - spin * velocity_u / denominator

    if accumulate_time:
        radius = 1.0 / u
        radial_rate = -velocity_u / (u * u)
        time_bl_rate = (
            spin
            * (
                angular_momentum
                - spin * energy * sine * sine
            )
            + (radius * radius + spin * spin)
            * p_scaled
            / denominator
        )
        time_ks_rate = (
            time_bl_rate
            + 2.0 * radius * radial_rate * u * u / denominator
        )
    else:
        time_ks_rate = 0.0

    return (
        velocity_u,
        acceleration_u,
        velocity_theta,
        acceleration_theta,
        phi_ks_rate,
        time_ks_rate,
    )


def _linear_combination(
    base: Sequence[float],
    step: float,
    terms: Iterable[tuple[float, Sequence[float]]],
) -> tuple[float, ...]:
    weighted = list(terms)
    return tuple(
        base[index]
        + step
        * math.fsum(
            coefficient * vector[index] for coefficient, vector in weighted
        )
        for index in range(len(base))
    )


def _dormand_prince_step(
    state: Sequence[float],
    step: float,
    constants: Constants,
    *,
    accumulate_time: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """One Dormand-Prince 5(4) step; return fifth-order state and error."""
    derivative = lambda value: _derivatives(  # noqa: E731
        value, constants, accumulate_time=accumulate_time
    )
    k1 = derivative(state)
    k2 = derivative(_linear_combination(state, step, [(1 / 5, k1)]))
    k3 = derivative(
        _linear_combination(state, step, [(3 / 40, k1), (9 / 40, k2)])
    )
    k4 = derivative(
        _linear_combination(
            state,
            step,
            [(44 / 45, k1), (-56 / 15, k2), (32 / 9, k3)],
        )
    )
    k5 = derivative(
        _linear_combination(
            state,
            step,
            [
                (19372 / 6561, k1),
                (-25360 / 2187, k2),
                (64448 / 6561, k3),
                (-212 / 729, k4),
            ],
        )
    )
    k6 = derivative(
        _linear_combination(
            state,
            step,
            [
                (9017 / 3168, k1),
                (-355 / 33, k2),
                (46732 / 5247, k3),
                (49 / 176, k4),
                (-5103 / 18656, k5),
            ],
        )
    )
    fifth = _linear_combination(
        state,
        step,
        [
            (35 / 384, k1),
            (500 / 1113, k3),
            (125 / 192, k4),
            (-2187 / 6784, k5),
            (11 / 84, k6),
        ],
    )
    k7 = derivative(fifth)
    fourth = _linear_combination(
        state,
        step,
        [
            (5179 / 57600, k1),
            (7571 / 16695, k3),
            (393 / 640, k4),
            (-92097 / 339200, k5),
            (187 / 2100, k6),
            (1 / 40, k7),
        ],
    )
    error = tuple(
        fifth[index] - fourth[index] for index in range(len(fifth))
    )
    return fifth, error


def _normalize_polar_chart(
    state: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    u, velocity_u, theta, velocity_theta, phi, coordinate_time = state
    while theta < 0.0 or theta > math.pi:
        if theta < 0.0:
            theta = -theta
            velocity_theta = -velocity_theta
            phi += math.pi
        elif theta > math.pi:
            theta = 2.0 * math.pi - theta
            velocity_theta = -velocity_theta
            phi += math.pi
    return u, velocity_u, theta, velocity_theta, phi, coordinate_time


def _null_residual(state: Sequence[float], constants: Constants) -> float:
    """Reconstruct ``|g^{mu nu}p_mu p_nu|`` from separated BL variables."""
    u, velocity_u, theta, velocity_theta, _phi, _time = state
    if u <= 0.0:
        return 0.0
    radius = 1.0 / u
    spin = constants.spin
    energy = constants.energy
    angular_momentum = constants.angular_momentum_z
    sigma, delta, big_a, _lapse, _omega = _kerr_bl_quantities(
        radius, theta, spin
    )
    sine = math.sin(theta)
    sine2 = max(sine * sine, 1.0e-30)
    radial_mino = -velocity_u / (u * u)
    p_r = radial_mino / delta
    p_theta = velocity_theta
    g_tt_inverse = -big_a / (sigma * delta)
    g_t_phi_inverse = -2.0 * spin * radius / (sigma * delta)
    g_phi_phi_inverse = (
        delta - spin * spin * sine2
    ) / (sigma * delta * sine2)
    return abs(
        g_tt_inverse * energy * energy
        + 2.0 * g_t_phi_inverse * (-energy) * angular_momentum
        + g_phi_phi_inverse * angular_momentum * angular_momentum
        + delta / sigma * p_r * p_r
        + p_theta * p_theta / sigma
    )


def _angular_separation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    dot = max(
        -1.0,
        min(1.0, math.fsum(a * b for a, b in zip(first, second))),
    )
    cross_x = first[1] * second[2] - first[2] * second[1]
    cross_y = first[2] * second[0] - first[0] * second[2]
    cross_z = first[0] * second[1] - first[1] * second[0]
    cross_norm = math.sqrt(
        cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    )
    return math.atan2(cross_norm, dot)


def _world_direction(theta: float, phi_ks: float) -> tuple[float, float, float]:
    sine = math.sin(theta)
    return (
        sine * math.cos(phi_ks),
        sine * math.sin(phi_ks),
        math.cos(theta),
    )


def _to_icrs(world: Sequence[float]) -> tuple[float, float, float]:
    transformed = tuple(
        math.fsum(row[index] * world[index] for index in range(3))
        for row in WORLD_TO_ICRS
    )
    inverse_norm = 1.0 / math.sqrt(math.fsum(value * value for value in transformed))
    return tuple(value * inverse_norm for value in transformed)


def _locate_u_event(
    start: Sequence[float],
    step: float,
    target_u: float,
    constants: Constants,
    *,
    accumulate_time: bool,
) -> tuple[float, ...]:
    lower = 0.0
    upper = step
    start_sign = start[0] - target_u
    candidate = tuple(start)
    for _ in range(54):
        middle = 0.5 * (lower + upper)
        candidate, _error = _dormand_prince_step(
            start,
            middle,
            constants,
            accumulate_time=accumulate_time,
        )
        sign = candidate[0] - target_u
        if sign == 0.0:
            break
        if (sign > 0.0) == (start_sign > 0.0):
            lower = middle
        else:
            upper = middle
    mutable = list(candidate)
    mutable[0] = target_u
    return _normalize_polar_chart(mutable)


def _integrate_to_radial_event(
    initial_state: Sequence[float],
    constants: Constants,
    *,
    target_u: float,
    crossing: str,
    accumulate_time: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
    initial_step: float = 2.0e-4,
) -> tuple[tuple[float, ...], int, float, float]:
    """Integrate until u crosses target; return state, steps, error, null max."""
    state = tuple(initial_state)
    step = initial_step
    accepted = 0
    angular_error_sum = 0.0
    maximum_null = _null_residual(state, constants)
    absolute_scales = (
        absolute_tolerance,
        5.0 * absolute_tolerance,
        absolute_tolerance,
        5.0 * absolute_tolerance,
        absolute_tolerance,
        max(1.0e-8, 100.0 * absolute_tolerance),
    )

    while accepted < MAX_ACCEPTED_STEPS:
        candidate, error = _dormand_prince_step(
            state,
            step,
            constants,
            accumulate_time=accumulate_time,
        )
        if any(not math.isfinite(value) for value in candidate):
            step *= 0.2
            if step < 1.0e-14:
                raise ArithmeticError("non-finite Kerr integrator state")
            continue
        normalized_error = max(
            abs(error[index])
            / (
                absolute_scales[index]
                + relative_tolerance
                * max(abs(state[index]), abs(candidate[index]))
            )
            for index in range(len(state))
        )
        if normalized_error > 1.0:
            step *= max(0.1, 0.9 * normalized_error ** (-0.2))
            if step < 1.0e-14:
                raise ArithmeticError("Kerr integrator minimum step exhausted")
            continue

        candidate = _normalize_polar_chart(candidate)
        crossed = (
            crossing == "up"
            and state[0] < target_u <= candidate[0]
        ) or (
            crossing == "down"
            and state[0] > target_u >= candidate[0]
        )
        accepted += 1
        angular_error_sum += math.hypot(error[2], error[4])
        maximum_null = max(
            maximum_null,
            _null_residual(candidate, constants),
        )
        if crossed:
            event_state = _locate_u_event(
                state,
                step,
                target_u,
                constants,
                accumulate_time=accumulate_time,
            )
            maximum_null = max(
                maximum_null,
                _null_residual(event_state, constants),
            )
            return (
                event_state,
                accepted,
                angular_error_sum,
                maximum_null,
            )
        state = candidate
        if normalized_error == 0.0:
            growth = 4.0
        else:
            growth = min(4.0, max(0.2, 0.9 * normalized_error ** (-0.2)))
        step *= growth
        step = min(step, 2.0e-2)

    raise ArithmeticError("Kerr integrator accepted-step budget exhausted")


def _integrate_primary(
    initial_state: Sequence[float],
    constants: Constants,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[str, tuple[float, ...], int, float, float]:
    """Integrate until capture or outward escape boundary crossing."""
    state = tuple(initial_state)
    capture_u = 1.0 / capture_radius_m(constants.spin)
    escape_u = 1.0 / ESCAPE_RADIUS_M
    step = 2.0e-4
    accepted = 0
    angular_error_sum = 0.0
    maximum_null = _null_residual(state, constants)
    absolute_scales = (
        absolute_tolerance,
        5.0 * absolute_tolerance,
        absolute_tolerance,
        5.0 * absolute_tolerance,
        absolute_tolerance,
        max(1.0e-8, 100.0 * absolute_tolerance),
    )

    while accepted < MAX_ACCEPTED_STEPS:
        candidate, error = _dormand_prince_step(
            state,
            step,
            constants,
            accumulate_time=True,
        )
        if any(not math.isfinite(value) for value in candidate):
            step *= 0.2
            if step < 1.0e-14:
                raise ArithmeticError("non-finite primary Kerr state")
            continue
        normalized_error = max(
            abs(error[index])
            / (
                absolute_scales[index]
                + relative_tolerance
                * max(abs(state[index]), abs(candidate[index]))
            )
            for index in range(len(state))
        )
        if normalized_error > 1.0:
            step *= max(0.1, 0.9 * normalized_error ** (-0.2))
            continue
        candidate = _normalize_polar_chart(candidate)
        accepted += 1
        angular_error_sum += math.hypot(error[2], error[4])
        maximum_null = max(maximum_null, _null_residual(candidate, constants))

        if state[0] < capture_u <= candidate[0]:
            terminal = _locate_u_event(
                state,
                step,
                capture_u,
                constants,
                accumulate_time=True,
            )
            maximum_null = max(maximum_null, _null_residual(terminal, constants))
            return (
                "captured",
                terminal,
                accepted,
                angular_error_sum,
                maximum_null,
            )
        if (
            state[0] > escape_u >= candidate[0]
            and candidate[1] < 0.0
        ):
            terminal = _locate_u_event(
                state,
                step,
                escape_u,
                constants,
                accumulate_time=True,
            )
            maximum_null = max(maximum_null, _null_residual(terminal, constants))
            return (
                "escaped",
                terminal,
                accepted,
                angular_error_sum,
                maximum_null,
            )

        state = candidate
        if normalized_error == 0.0:
            growth = 4.0
        else:
            growth = min(4.0, max(0.2, 0.9 * normalized_error ** (-0.2)))
        step = min(2.0e-2, step * growth)

    return "unresolved", state, accepted, angular_error_sum, maximum_null


def _solve_ray_once(
    screen_x: float,
    screen_y: float,
    focal_pixels_per_radian: float,
    *,
    spin: float = SPIN_A_M,
    absolute_tolerance: float = FINE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = FINE_RELATIVE_TOLERANCE,
) -> RaySolution:
    constants, initial_state = _initial_constants_and_state(
        screen_x, screen_y, spin
    )
    outcome, terminal, primary_steps, angular_error, null_residual = (
        _integrate_primary(
            initial_state,
            constants,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )

    coordinate_lookback = max(0.0, -terminal[5])
    base_projection_error = (
        angular_error + FLOAT32_DIRECTION_ERROR_RAD
    ) * focal_pixels_per_radian
    if outcome == "captured":
        return RaySolution(
            outcome=OUTCOME_CAPTURED,
            escape_direction_icrs=(0.0, 0.0, 0.0),
            frequency_shift_g=0.0,
            coordinate_lookback_time_m=coordinate_lookback,
            null_residual=null_residual,
            projection_error_px=base_projection_error,
            boundary_continuation_error_rad=0.0,
            accepted_steps=primary_steps,
        )
    if outcome != "escaped":
        return RaySolution(
            outcome=OUTCOME_UNRESOLVED,
            escape_direction_icrs=(0.0, 0.0, 0.0),
            frequency_shift_g=0.0,
            coordinate_lookback_time_m=0.0,
            null_residual=null_residual,
            projection_error_px=base_projection_error,
            boundary_continuation_error_rad=0.0,
            accepted_steps=primary_steps,
        )

    boundary_theta = terminal[2]
    boundary_phi = terminal[4]
    half_state, half_steps, half_error, half_null = _integrate_to_radial_event(
        terminal,
        constants,
        target_u=0.5 / ESCAPE_RADIUS_M,
        crossing="down",
        accumulate_time=False,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    infinity_state, infinity_steps, infinity_error, infinity_null = (
        _integrate_to_radial_event(
            half_state,
            constants,
            target_u=0.0,
            crossing="down",
            accumulate_time=False,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )
    world_direction = _world_direction(infinity_state[2], infinity_state[4])
    direction_icrs = _to_icrs(world_direction)

    # A boundary-doubling audit: linearly Richardson-extrapolate angular
    # coordinates from R and 2R, then compare with the exact u=0 continuation.
    delta_phi = math.remainder(
        half_state[4] - boundary_phi,
        2.0 * math.pi,
    )
    richardson_theta = 2.0 * half_state[2] - boundary_theta
    richardson_phi = half_state[4] + delta_phi
    boundary_error = _angular_separation(
        _world_direction(richardson_theta, richardson_phi),
        world_direction,
    )

    radius = ESCAPE_RADIUS_M
    theta = terminal[2]
    _sigma, _delta, _big_a, boundary_lapse, boundary_omega = (
        _kerr_bl_quantities(radius, theta, spin)
    )
    boundary_energy = (
        -constants.energy
        + boundary_omega * constants.angular_momentum_z
    ) / boundary_lapse
    frequency_shift = 1.0 / boundary_energy
    total_error = angular_error + half_error + infinity_error
    projection_error = (
        total_error + FLOAT32_DIRECTION_ERROR_RAD
    ) * focal_pixels_per_radian
    return RaySolution(
        outcome=OUTCOME_ESCAPED,
        escape_direction_icrs=direction_icrs,
        frequency_shift_g=frequency_shift,
        coordinate_lookback_time_m=coordinate_lookback,
        null_residual=max(null_residual, half_null, infinity_null),
        projection_error_px=projection_error,
        boundary_continuation_error_rad=boundary_error,
        accepted_steps=primary_steps + half_steps + infinity_steps,
    )


def solve_ray(
    screen_x: float,
    screen_y: float,
    focal_pixels_per_radian: float,
    *,
    spin: float = SPIN_A_M,
    absolute_tolerance: float = FINE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = FINE_RELATIVE_TOLERANCE,
    refinement_check: bool = True,
) -> RaySolution:
    """Solve a ray and gate it with an independent-tolerance global endpoint.

    The embedded DP local estimator controls individual steps.  It is not by
    itself a global observable error estimate, so production additionally
    traces the complete ray at the declared coarse tolerances.  Escaped
    endpoints are compared on the ICRS sphere; an outcome disagreement or an
    endpoint discrepancy above 0.25 pixel is fail-closed as unresolved.
    """
    if _is_equatorial_separatrix(screen_x, screen_y, spin):
        # These rays asymptote to an unstable equatorial spherical photon
        # orbit and reach neither declared terminal worldtube in finite affine
        # parameter.  Never turn a separatrix into a false capture/escape.
        return RaySolution(
            outcome=OUTCOME_UNRESOLVED,
            escape_direction_icrs=(0.0, 0.0, 0.0),
            frequency_shift_g=0.0,
            coordinate_lookback_time_m=0.0,
            null_residual=0.0,
            projection_error_px=0.0,
            boundary_continuation_error_rad=0.0,
            accepted_steps=0,
        )
    fine = _solve_ray_once(
        screen_x,
        screen_y,
        focal_pixels_per_radian,
        spin=spin,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not refinement_check or fine.outcome == OUTCOME_UNRESOLVED:
        return fine

    coarse = _solve_ray_once(
        screen_x,
        screen_y,
        focal_pixels_per_radian,
        spin=spin,
        absolute_tolerance=max(
            COARSE_ABSOLUTE_TOLERANCE,
            absolute_tolerance * 1_000.0,
        ),
        relative_tolerance=max(
            COARSE_RELATIVE_TOLERANCE,
            relative_tolerance * 100.0,
        ),
    )
    # Null residual is a property of the stored fine trajectory.  The coarse
    # trajectory is only a global endpoint oracle and is intentionally not
    # advertised as satisfying the fine null gate.
    maximum_null = fine.null_residual
    if fine.outcome != coarse.outcome:
        return RaySolution(
            outcome=OUTCOME_UNRESOLVED,
            escape_direction_icrs=(0.0, 0.0, 0.0),
            frequency_shift_g=0.0,
            coordinate_lookback_time_m=0.0,
            null_residual=maximum_null,
            projection_error_px=PROJECTION_CONVERGENCE_GATE_PX,
            boundary_continuation_error_rad=max(
                fine.boundary_continuation_error_rad,
                coarse.boundary_continuation_error_rad,
            ),
            accepted_steps=fine.accepted_steps + coarse.accepted_steps,
        )

    global_projection_error = 0.0
    if fine.outcome == OUTCOME_ESCAPED:
        global_projection_error = _angular_separation(
            fine.escape_direction_icrs,
            coarse.escape_direction_icrs,
        ) * focal_pixels_per_radian
    projection_error = max(
        fine.projection_error_px,
        global_projection_error + FLOAT32_DIRECTION_ERROR_RAD
        * focal_pixels_per_radian,
    )
    if projection_error > PROJECTION_CONVERGENCE_GATE_PX:
        return RaySolution(
            outcome=OUTCOME_UNRESOLVED,
            escape_direction_icrs=(0.0, 0.0, 0.0),
            frequency_shift_g=0.0,
            coordinate_lookback_time_m=0.0,
            null_residual=maximum_null,
            projection_error_px=PROJECTION_CONVERGENCE_GATE_PX,
            boundary_continuation_error_rad=max(
                fine.boundary_continuation_error_rad,
                coarse.boundary_continuation_error_rad,
            ),
            accepted_steps=fine.accepted_steps + coarse.accepted_steps,
        )
    return RaySolution(
        outcome=fine.outcome,
        escape_direction_icrs=fine.escape_direction_icrs,
        frequency_shift_g=fine.frequency_shift_g,
        coordinate_lookback_time_m=fine.coordinate_lookback_time_m,
        null_residual=maximum_null,
        projection_error_px=projection_error,
        boundary_continuation_error_rad=fine.boundary_continuation_error_rad,
        accepted_steps=fine.accepted_steps + coarse.accepted_steps,
    )


def _pack_record(solution: RaySolution) -> bytes:
    if solution.outcome == OUTCOME_ESCAPED:
        return RECORD.pack(
            *solution.escape_direction_icrs,
            solution.frequency_shift_g,
            solution.coordinate_lookback_time_m,
            solution.null_residual,
            solution.projection_error_px,
            OUTCOME_ESCAPED,
            CAPTURE_NONE,
            VALID_ALL,
        )
    if solution.outcome == OUTCOME_CAPTURED:
        return RECORD.pack(
            0.0,
            0.0,
            0.0,
            0.0,
            solution.coordinate_lookback_time_m,
            solution.null_residual,
            solution.projection_error_px,
            OUTCOME_CAPTURED,
            CAPTURE_BH,
            VALID_CAPTURED,
        )
    return RECORD.pack(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        solution.null_residual,
        solution.projection_error_px,
        OUTCOME_UNRESOLVED,
        CAPTURE_NONE,
        VALID_UNRESOLVED,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bl_vector_to_ks_cartesian(
    vector: Sequence[float],
    *,
    radius: float,
    theta: float,
    phi_ks: float,
    spin: float,
) -> list[float]:
    """Transform a BL contravariant vector to ingoing Cartesian KS."""
    _sigma, delta, _big_a, _lapse, _omega = _kerr_bl_quantities(
        radius, theta, spin
    )
    sine = math.sin(theta)
    cosine = math.cos(theta)
    cos_phi = math.cos(phi_ks)
    sin_phi = math.sin(phi_ks)
    x = (radius * cos_phi - spin * sin_phi) * sine
    y = (radius * sin_phi + spin * cos_phi) * sine
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
    dx_dphi = -y
    dy_dphi = x

    v_t, v_r, v_theta, v_phi = vector
    return [
        v_t + dt_dr * v_r,
        dx_dr * v_r + dx_dtheta * v_theta + dx_dphi * v_phi,
        dy_dr * v_r + dy_dtheta * v_theta + dy_dphi * v_phi,
        cosine * v_r + dz_dtheta * v_theta,
    ]


def _ks_observer_geometry(spin: float) -> dict[str, object]:
    radius = OBSERVER_RADIUS_M
    theta = 0.5 * math.pi
    phi_ks = -math.atan2(spin, radius)
    sigma, delta, big_a, lapse, omega = _kerr_bl_quantities(
        radius, theta, spin
    )
    x = math.sqrt(radius * radius + spin * spin)
    y = 0.0
    z = 0.0

    u_bl = [1.0 / lapse, 0.0, 0.0, omega / lapse]
    e_r_bl = [0.0, math.sqrt(delta / sigma), 0.0, 0.0]
    e_theta_bl = [0.0, 0.0, 1.0 / math.sqrt(sigma), 0.0]
    e_phi_bl = [0.0, 0.0, 0.0, math.sqrt(sigma / big_a)]
    u = _bl_vector_to_ks_cartesian(
        u_bl, radius=radius, theta=theta, phi_ks=phi_ks, spin=spin
    )
    right = [
        -value
        for value in _bl_vector_to_ks_cartesian(
            e_phi_bl,
            radius=radius,
            theta=theta,
            phi_ks=phi_ks,
            spin=spin,
        )
    ]
    up = [
        -value
        for value in _bl_vector_to_ks_cartesian(
            e_theta_bl,
            radius=radius,
            theta=theta,
            phi_ks=phi_ks,
            spin=spin,
        )
    ]
    forward = [
        -value
        for value in _bl_vector_to_ks_cartesian(
            e_r_bl,
            radius=radius,
            theta=theta,
            phi_ks=phi_ks,
            spin=spin,
        )
    ]

    h = radius / sigma
    l_x = (radius * x + spin * y) / (radius * radius + spin * spin)
    l_y = (radius * y - spin * x) / (radius * radius + spin * spin)
    l_z = z / radius
    l_spatial = [l_x, l_y, l_z]
    metric_covariant = [[0.0] * 4 for _ in range(4)]
    metric_covariant[0][0] = -1.0 + 2.0 * h
    for index in range(3):
        metric_covariant[0][index + 1] = 2.0 * h * l_spatial[index]
        metric_covariant[index + 1][0] = 2.0 * h * l_spatial[index]
        for second in range(3):
            metric_covariant[index + 1][second + 1] = (
                (1.0 if index == second else 0.0)
                + 2.0 * h * l_spatial[index] * l_spatial[second]
            )
    metric_contravariant = [[0.0] * 4 for _ in range(4)]
    metric_contravariant[0][0] = -1.0 - 2.0 * h
    for index in range(3):
        metric_contravariant[0][index + 1] = 2.0 * h * l_spatial[index]
        metric_contravariant[index + 1][0] = 2.0 * h * l_spatial[index]
        for second in range(3):
            metric_contravariant[index + 1][second + 1] = (
                (1.0 if index == second else 0.0)
                - 2.0 * h * l_spatial[index] * l_spatial[second]
            )

    return {
        "event": [0.0, x, y, z],
        "metricCovariant": [
            value for row in metric_covariant for value in row
        ],
        "metricContravariant": [
            value for row in metric_contravariant for value in row
        ],
        "fourVelocity": u,
        "tetrad": [u, right, up, forward],
        "position": [x, y, z],
    }


def _build_manifest(
    *,
    width: int,
    height: int,
    spin: float,
    chunks: list[dict[str, object]],
    solutions: Sequence[RaySolution],
) -> dict[str, object]:
    document = fixture_manifest(b"")
    total = width * height
    counts = {
        OUTCOME_ESCAPED: sum(
            solution.outcome == OUTCOME_ESCAPED for solution in solutions
        ),
        OUTCOME_CAPTURED: sum(
            solution.outcome == OUTCOME_CAPTURED for solution in solutions
        ),
        OUTCOME_UNRESOLVED: sum(
            solution.outcome == OUTCOME_UNRESOLVED for solution in solutions
        ),
    }
    null_residuals = [solution.null_residual for solution in solutions]
    projection_errors = [solution.projection_error_px for solution in solutions]
    boundary_errors = [
        solution.boundary_continuation_error_rad
        for solution in solutions
        if solution.outcome == OUTCOME_ESCAPED
    ]
    source_spin_vector = _remnant_spin_source_vector()
    source_spin_magnitude = math.sqrt(
        math.fsum(value * value for value in source_spin_vector)
    )
    spin_is_pinned_remnant = abs(abs(spin) - source_spin_magnitude) <= 5.0e-13
    if spin_is_pinned_remnant and spin >= 0.0:
        spin_description = (
            "The pinned SXS remnant vector "
            f"[{source_spin_vector[0]:.12g}, {source_spin_vector[1]:.12g}, "
            f"{source_spin_vector[2]:.12g}] is rigidly rotated (not "
            "component-truncated) onto world +Z; its invariant magnitude "
            f"a/M={source_spin_magnitude:.12f} is used."
        )
    else:
        spin_description = (
            f"Analytic Kerr parameter override a/M={spin:.12f}; the pinned "
            "SXS remnant-spin artifact is retained only as a comparison source."
        )
    geometry = _ks_observer_geometry(spin)
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    observer_x = float(geometry["position"][0])  # type: ignore[index]
    camera_to_world = [
        0.0, 0.0, -1.0, observer_x,
        -1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    world_to_camera = [
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0, observer_x,
        0.0, 0.0, 0.0, 1.0,
    ]

    document.update(
        {
            "id": "kerr-remnant-reference-v1",
            "datasetKind": "stationary-reference-transfer-map",
            "renderable": True,
            "scientificStatus": {
                "classification": (
                    "project-generated analytic stationary Kerr reference; not NR"
                ),
                "sourceIsNumericalRelativity": False,
                "derivedFromNearZoneSpacetime": False,
                "derivedWithSlowLightGeodesics": False,
                "description": (
                    "A fixed-camera vacuum transfer map from exact stationary "
                    "Kerr null geodesics for the SXS:BBH:0001 remnant spin. "
                    "It contains no accretion emission."
                ),
                "prohibitedClaim": (
                    "Do not describe this analytic Kerr reference as a binary "
                    "merger image, NR-backed pixels, or a GRMHD simulation."
                ),
            },
            "physicalSystem": {
                "kind": "stationary-black-hole",
                "vacuum": True,
                "componentIds": ["remnant"],
                "parameterEpochProtocolM": 0.0,
                "massRatioQ": None,
                "dimensionlessSpins": [
                    {"componentId": "remnant", "vector": [0.0, 0.0, spin]}
                ],
                "eccentricity": None,
                "referenceOrbitalPhaseRad": None,
                "remnant": None,
                "notApplicableReason": (
                    "Binary orbital parameters do not apply to the eternal "
                    "stationary Kerr reference."
                ),
                "description": (
                    f"Vacuum Kerr M=1 with spin along world +Z. {spin_description}"
                ),
            },
            "provenance": {
                "origin": "project-generated",
                "project": "ShuoleiWang/blackhole",
                "datasetVersion": "1.0.0",
                "license": "NOASSERTION",
                "artifactUriBase": "repository-root",
                "sourceSimulation": {
                    "kind": "stationary-reference",
                    "catalog": "SXS",
                    "identifier": "analytic-kerr-m1-sxs-bbh-0001-remnant-spin",
                    "version": "1",
                    "doi": None,
                    "evolutionCode": None,
                    "notApplicableReason": (
                        "The metric is the analytic Kerr solution; SXS supplies "
                        "only the pinned remnant spin parameter, not ray pixels."
                    ),
                },
                "generator": {
                    "name": Path(__file__).name,
                    "version": "1.0.0",
                    "uri": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "command": "python3 scripts/generate_kerr_transfer_map.py",
                    "codeRevision": (
                        f"sha256:{sha256_bytes(Path(__file__).resolve().read_bytes())}"
                    ),
                    "deterministic": True,
                },
                "sourceArtifacts": [
                    source_artifact("generator-source", Path(__file__).resolve()),
                    source_artifact("schema", SCHEMA_PATH),
                    source_artifact("manifest-template-helper", HELPER_PATH),
                    source_artifact(
                        "remnant-spin-source",
                        REMNANT_SPIN_SOURCE_PATH,
                    ),
                ],
            },
            "units": {
                "system": "geometric",
                "G": 1.0,
                "c": 1.0,
                "massNormalization": {
                    "quantity": "stationary black-hole mass",
                    "symbol": "M",
                    "value": 1.0,
                    "definition": (
                        "The Kerr mass parameter; all radii and coordinate "
                        "times are expressed in this M."
                    ),
                    "referenceEpochSourceM": None,
                },
                "coordinateTime": "M",
                "length": "M",
                "angle": "radian",
                "frequencyShift": "dimensionless",
            },
            "timeReference": {
                "sourceTimeAtProtocolZeroM": 0.0,
                "sourceTimeDirection": "future-increasing",
                "protocolTimeDefinition": (
                    "t_protocol=t_source-sourceTimeAtProtocolZeroM"
                ),
                "zeroEvent": {
                    "name": "stationary Kerr reference epoch",
                    "source": "project-generated",
                    "description": (
                        "An arbitrary t_KS=0 event in the stationary spacetime."
                    ),
                },
                "waveformTimeMapping": {
                    "status": "not-applicable",
                    "sourceQuantity": None,
                    "mapping": None,
                    "notApplicableReason": (
                        "An eternal stationary Kerr black hole has no merger waveform."
                    ),
                },
            },
            "coordinates": {
                "metricSignature": "-+++",
                "nrChart": {
                    "status": "declared",
                    "gauge": (
                        "ingoing Cartesian Kerr-Schild coordinates, with Kerr "
                        "radius defined by the standard oblate-spheroidal quartic"
                    ),
                    "coordinates": "Cartesian Kerr-Schild (t_KS,x,y,z)",
                    "timeSlicing": "constant ingoing Kerr-Schild coordinate time",
                },
                "worldFrame": {
                    "handedness": "right",
                    "axisOrder": ["x", "y", "z"],
                    "origin": "Kerr symmetry centre",
                    "matrixConvention": (
                        "row-major spatial affine 4x4 matrices multiplying "
                        "[x,y,z,1] column vectors; not spacetime coordinate transforms"
                    ),
                    "nrToWorld": identity,
                    "worldToNr": identity,
                },
                "sky": {
                    "referenceFrame": "ICRS",
                    "icrsAxes": {
                        "x": (
                            "ICRS right ascension 0 degrees, declination 0 degrees"
                        ),
                        "y": (
                            "ICRS right ascension 90 degrees, declination 0 degrees"
                        ),
                        "z": "ICRS north celestial pole",
                    },
                    "rotationConvention": (
                        "proper right-handed row-major 3x3 rotations multiplying "
                        "spatial column vectors"
                    ),
                    "worldToIcrs": [
                        value for row in WORLD_TO_ICRS for value in row
                    ],
                    "icrsToWorld": [
                        value for row in ICRS_TO_WORLD for value in row
                    ],
                    "projection": "equirectangular",
                    "longitudeMapping": "u=fract(longitude/(2*pi)+0.5)",
                    "latitudeMapping": "v=0.5-latitude/pi",
                    "escapeDirectionFrame": "ICRS",
                },
            },
            "observer": {
                "tetradBasisOrder": ["time", "right", "up", "forward"],
                "tetradIndexConvention": (
                    "e_(a)^mu; rows are local basis vectors in the NR coordinate basis"
                ),
                "samples": [
                    {
                        "sampleIndex": 0,
                        "protocolTimeM": 0.0,
                        "eventNr": geometry["event"],
                        "metricCovariantNr": geometry["metricCovariant"],
                        "metricContravariantNr": geometry[
                            "metricContravariant"
                        ],
                        "fourVelocityContravariantNr": geometry["fourVelocity"],
                        "properTimeM": 0.0,
                        "tetradContravariantNr": geometry["tetrad"],
                    }
                ],
            },
            "camera": {
                "frameType": "affine-visualization-frame",
                "motion": "fixed",
                "matrixConvention": (
                    "row-major spatial affine 4x4 matrices multiplying "
                    "[x,y,z,1] column vectors; not spacetime coordinate transforms"
                ),
                "basisOrder": ["right", "up", "forward"],
                "cameraToWorld": camera_to_world,
                "worldToCamera": world_to_camera,
                "physicalRelation": (
                    "The affine frame is for deterministic playback coordinates; "
                    "physical ray initialization uses the observer tetrad."
                ),
            },
            "projection": {
                "model": "rectilinear-pinhole",
                "widthPixels": width,
                "heightPixels": height,
                "verticalFieldOfViewRad": VERTICAL_FOV_RAD,
                "imageOrigin": "top-left",
                "pixelSampleLocation": "center",
                "aspectConvention": "aspect=widthPixels/heightPixels",
                "screenXFormula": (
                    "screenX=((x+0.5)/widthPixels*2-1)*aspect*"
                    "tan(verticalFieldOfViewRad/2)"
                ),
                "screenYFormula": (
                    "screenY=(1-(y+0.5)/heightPixels*2)*"
                    "tan(verticalFieldOfViewRad/2)"
                ),
                "rayTimeOrientation": "past-directed",
                "localRayConvention": (
                    "k^(a)=(-1,normalize(screenX,screenY,1)) in the "
                    "time/right/up/forward tetrad"
                ),
            },
            "sampling": {
                "observationTimesM": [0.0],
                "timeCoordinate": "protocol",
                "dimensionOrder": ["time", "y", "x"],
                "pixelOrder": "row-major",
                "tileOrder": "manifest-order",
                "interpolation": {
                    "time": "none",
                    "continuous": "none-nearest-texel-center",
                    "escapeDirection": "nearest-no-blend",
                    "categorical": "nearest-no-blend",
                    "invalidRecords": "never-sample-sky",
                },
            },
            "rayIntegration": {
                "spacetimeMode": "stationary",
                "spatialInterpolation": (
                    "none; exact analytic Kerr metric and separated potentials"
                ),
                "temporalInterpolation": (
                    "none; Kerr is stationary and the map has one sample"
                ),
                "integrator": {
                    "name": "dormand-prince-kerr-carter-mino",
                    "method": (
                        "adaptive Dormand-Prince 5(4) integration of the exact "
                        "separated Kerr Hamilton-Jacobi equations in Mino time "
                        "for u=1/r and theta, conserving E, Lz and Carter Q; "
                        "ingoing Kerr-Schild t and phi are integrated by the "
                        "exact BL-to-KS differential transform"
                    ),
                },
                "tolerances": {
                    "absolute": FINE_ABSOLUTE_TOLERANCE,
                    "relative": FINE_RELATIVE_TOLERANCE,
                    "nullConstraint": 1.0e-8,
                },
                "initialNormalization": "u_observer·k_observer=1",
                "timeOrientation": "past-directed",
                "termination": {
                    "escaped": "intersect escapeBoundary",
                    "captured": "intersect a captureTargets surface",
                    "unresolved": "step or affine-parameter budget exhausted",
                    "outside-domain": (
                        "left the declared spacetime domain away from escapeBoundary"
                    ),
                    "integrator-failure": "non-finite state or tolerance failure",
                    "missing": "record was not generated",
                },
                "integrationPrecision": "float64",
                "outputPrecision": "float32",
            },
            "escapeBoundary": {
                "surface": {
                    "kind": "constant-Kerr-r-oblate-worldtube",
                    "centreWorldM": [0.0, 0.0, 0.0],
                    "radiusM": ESCAPE_RADIUS_M,
                },
                "referenceObserver": {
                    "kind": "Boyer-Lindquist-ZAMO",
                    "definition": (
                        "Future-directed normal to Boyer-Lindquist t slices, "
                        "equivalently the ZAMO "
                        "u=alpha^-1(partial_t+omega partial_phi), evaluated "
                        "where the ray crosses the oblate Kerr coordinate "
                        "surface r=1000M; this is not the normal to a "
                        "Cartesian Kerr-Schild t slice."
                    ),
                    "sourceArtifactRole": None,
                },
                "frequencyShiftConvention": (
                    "g=(u_observer·k_observer)/(u_boundary·k_boundary)"
                ),
                "storedEscapeDirection": {
                    "frame": "ICRS",
                    "normalization": "unit Euclidean spatial vector",
                    "continuationBeyondBoundary": (
                        "After time/frequency termination at r=1000M, the exact "
                        "stationary Kerr equations are integrated numerically "
                        "for theta and ingoing-KS phi through r=2000M to u=0. "
                        "The asymptotic Cartesian "
                        "radial direction is stored; a boundary-doubling "
                        "Richardson audit is included in generator metrics."
                    ),
                },
            },
            "captureTargets": [
                {
                    "code": CAPTURE_BH,
                    "id": "remnant",
                    "description": (
                        "Stretched future-horizon worldtube at constant Kerr "
                        f"r=r_++0.02M={capture_radius_m(spin):.15g}M. "
                        "It is an oblate Kerr-r surface, not a Euclidean sphere."
                    ),
                    "surfaceKind": (
                        "constant-Kerr-r oblate stretched-horizon worldtube"
                    ),
                    "validityIntervalProtocolM": [0.0, 0.0],
                    "classificationPriority": 0,
                    "sourceArtifactRole": None,
                }
            ],
            "accuracy": {
                "status": "measured",
                "notMeasuredReason": None,
                "nrConvergence": {
                    "quantity": "NR spacetime grid-convergence order",
                    "status": "not-applicable",
                    "method": None,
                    "value": None,
                },
                "constraintNorms": {
                    "quantity": "NR constraint norms",
                    "status": "not-applicable",
                    "method": None,
                    "value": None,
                },
                "geodesicNullResidual": {
                    "quantity": (
                        "maximum reconstructed |g^{mu nu}p_mu p_nu|"
                    ),
                    "status": "measured",
                    "method": (
                        "Reconstructed in float64 from E,Lz, radial and polar "
                        "canonical momenta at every accepted trajectory state."
                    ),
                    "value": max(null_residuals),
                },
                "interpolationError": {
                    "quantity": (
                        "p95 stored-texel numerical and float32 direction "
                        "projection estimate in pixels"
                    ),
                    "status": "measured",
                    "method": (
                        "Maximum of the embedded DP5(4) fine local estimate "
                        "and the full-ray fine/coarse ICRS endpoint separation, "
                        "plus a 2e-7 rad float32 allowance, converted with the "
                        "pinhole focal length. Outcome disagreement or error "
                        "above 0.25 pixel is fail-closed as unresolved. Runtime "
                        "interpolation is disabled. "
                        f"Boundary-doubling p95={_percentile(boundary_errors, 0.95):.6e} rad."
                    ),
                    "value": _percentile(projection_errors, 0.95),
                },
                "unresolvedFraction": counts[OUTCOME_UNRESOLVED] / total,
                "outcomeFractions": {
                    "escaped": counts[OUTCOME_ESCAPED] / total,
                    "captured": counts[OUTCOME_CAPTURED] / total,
                    "unresolved": counts[OUTCOME_UNRESOLVED] / total,
                    "outside-domain": 0.0,
                    "integrator-failure": 0.0,
                    "missing": 0.0,
                    "unusable": counts[OUTCOME_UNRESOLVED] / total,
                },
                "fixtureAssertions": None,
            },
            "chunks": chunks,
        }
    )
    return document


def _solve_row(arguments: tuple[int, int, int, float, float]) -> tuple[int, bytes, list[RaySolution]]:
    y, width, height, screen_scale, spin = arguments
    grid_y = height - 2 * y - 1
    screen_y = grid_y * screen_scale
    focal_pixels_per_radian = height / (
        2.0 * math.tan(0.5 * VERTICAL_FOV_RAD)
    )
    payload = bytearray()
    solutions: list[RaySolution] = []
    for x in range(width):
        grid_x = 2 * x + 1 - width
        screen_x = grid_x * screen_scale
        solution = solve_ray(
            screen_x,
            screen_y,
            focal_pixels_per_radian,
            spin=spin,
        )
        solutions.append(solution)
        payload.extend(_pack_record(solution))
    return y, bytes(payload), solutions


def generate_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    tile_height: int = TILE_HEIGHT,
    spin: float = SPIN_A_M,
    jobs: int = 1,
) -> GenerationReport:
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("width and height must be even integers >= 2")
    if tile_height < 1:
        raise ValueError("tile height must be positive")
    if abs(spin) > 1.0:
        raise ValueError("dimensionless Kerr spin must satisfy |a/M| <= 1")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if not SCHEMA_PATH.is_file() or not HELPER_PATH.is_file():
        raise FileNotFoundError("schema and manifest template helper are required")

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    screen_scale = math.tan(0.5 * VERTICAL_FOV_RAD) / height
    row_arguments = [
        (y, width, height, screen_scale, spin) for y in range(height)
    ]
    if jobs == 1:
        solved_rows = [_solve_row(arguments) for arguments in row_arguments]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=jobs) as pool:
            solved_rows = list(pool.imap(_solve_row, row_arguments, chunksize=1))
    solved_rows.sort(key=lambda item: item[0])

    chunks: list[dict[str, object]] = []
    all_solutions: list[RaySolution] = []
    expected_names: set[str] = set()
    for tile_y in range(0, height, tile_height):
        current_height = min(tile_height, height - tile_y)
        name = f"t0000-y{tile_y:04d}-x0000.bin"
        expected_names.add(name)
        row_slice = solved_rows[tile_y : tile_y + current_height]
        payload = b"".join(row[1] for row in row_slice)
        for _y, _bytes, solutions in row_slice:
            all_solutions.extend(solutions)
        record_count = width * current_height
        if len(payload) != record_count * RECORD_BYTES:
            raise AssertionError("generated chunk does not match 32-byte ABI")
        (chunk_dir / name).write_bytes(payload)
        chunks.append(
            {
                "sampleIndex": 0,
                "tile": {
                    "x": 0,
                    "y": tile_y,
                    "width": width,
                    "height": current_height,
                },
                "uri": f"chunks/{name}",
                "recordCount": record_count,
                "recordBytes": RECORD_BYTES,
                "byteLength": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        print(
            f"  wrote rows {tile_y:04d}-{tile_y + current_height - 1:04d}"
        )
    for stale in chunk_dir.glob("*.bin"):
        if stale.name not in expected_names:
            stale.unlink()

    manifest = _build_manifest(
        width=width,
        height=height,
        spin=spin,
        chunks=chunks,
        solutions=all_solutions,
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_hash = sha256_bytes(manifest_bytes)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    (output_dir / "manifest.sha256").write_bytes(
        f"{manifest_hash}  manifest.json\n".encode("ascii")
    )

    escaped = sum(
        solution.outcome == OUTCOME_ESCAPED for solution in all_solutions
    )
    captured = sum(
        solution.outcome == OUTCOME_CAPTURED for solution in all_solutions
    )
    unresolved = sum(
        solution.outcome == OUTCOME_UNRESOLVED for solution in all_solutions
    )
    boundary_errors = [
        solution.boundary_continuation_error_rad
        for solution in all_solutions
        if solution.outcome == OUTCOME_ESCAPED
    ]
    projections = [solution.projection_error_px for solution in all_solutions]
    return GenerationReport(
        width=width,
        height=height,
        chunks=len(chunks),
        records=width * height,
        escaped=escaped,
        captured=captured,
        unresolved=unresolved,
        max_null_residual=max(
            solution.null_residual for solution in all_solutions
        ),
        p95_projection_error_px=_percentile(projections, 0.95),
        max_projection_error_px=max(projections),
        p95_boundary_continuation_error_rad=_percentile(
            boundary_errors, 0.95
        ),
        max_boundary_continuation_error_rad=max(boundary_errors),
        max_accepted_steps=max(
            solution.accepted_steps for solution in all_solutions
        ),
        elapsed_seconds=time.perf_counter() - started,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--tile-height", type=int, default=TILE_HEIGHT)
    parser.add_argument("--spin", type=float, default=SPIN_A_M)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="spawned CPU worker processes",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = generate_dataset(
        arguments.output.resolve(),
        width=arguments.width,
        height=arguments.height,
        tile_height=arguments.tile_height,
        spin=arguments.spin,
        jobs=arguments.jobs,
    )
    print("Kerr remnant reference transfer map generated")
    print(
        f"  resolution={report.width}x{report.height}, "
        f"records={report.records}, chunks={report.chunks}"
    )
    print(
        f"  outcomes=escaped:{report.escaped}, captured:{report.captured}, "
        f"unresolved:{report.unresolved}"
    )
    print(
        f"  max null residual={report.max_null_residual:.3e}; "
        f"p95/max projection={report.p95_projection_error_px:.3e}/"
        f"{report.max_projection_error_px:.3e} px"
    )
    print(
        "  p95/max infinity-continuation boundary-doubling error="
        f"{report.p95_boundary_continuation_error_rad:.3e}/"
        f"{report.max_boundary_continuation_error_rad:.3e} rad"
    )
    print(
        f"  max accepted steps={report.max_accepted_steps}; "
        f"elapsed={report.elapsed_seconds:.2f} s"
    )


if __name__ == "__main__":
    main()
