"""Independent selected-ray Kerr/Novikov--Thorne calibration oracle.

This module intentionally does *not* call the production adaptive DOPRI
geodesic tracer, accepted-step surface locator, or opaque-disk early-stop
implementation.  It evolves canonical Boyer--Lindquist coordinates and
covectors with a fixed-step classical RK4 method, locates terminal events by
partial-RK4 bisection, and requires an explicit ``h`` versus ``h/2`` replay.

The scope is deliberately narrow: a few selected, high-accuracy calibration
rays in stationary analytic Kerr spacetime.  It is not a full-frame proof, a
surface-complete ray bundle, numerical relativity, GRMHD, an atmosphere
solution, returning-radiation transport, or polarization transfer.

The optional spectral calculation independently reconstructs the local
blackbody and transfer factors, but it reuses the repository's analytic
Page--Thorne radial flux scalar.  Reports must therefore describe that part as
``shared-page-thorne-radial-scalar`` rather than as a wholly independent disk
emission oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Final, Literal, Mapping, NoReturn, Sequence

from offline.novikov_thorne import page_thorne_flux_shape


SUPPORTED_SAMPLER_IMPLEMENTATION_ID: Final = (
    "exact-kerr-nt-spectral-ray-sampler/v2"
)
SELECTED_RAY_ORACLE_IMPLEMENTATION_ID: Final = (
    "independent-bl-hamiltonian-fixed-rk4-selected-rays/v1"
)

GRAVITATIONAL_CONSTANT_M3_KG_S2: Final = 6.67430e-11
LIGHT_SPEED_M_S: Final = 299_792_458.0
PLANCK_CONSTANT_J_S: Final = 6.62607015e-34
BOLTZMANN_CONSTANT_J_K: Final = 1.380649e-23
STEFAN_BOLTZMANN_W_M2_K4: Final = (
    2.0
    * math.pi**5
    * BOLTZMANN_CONSTANT_J_K**4
    / (15.0 * PLANCK_CONSTANT_J_S**3 * LIGHT_SPEED_M_S**2)
)

RayOutcome = Literal["disk", "captured", "escaped", "unresolved"]


class KerrSelectedOracleError(RuntimeError):
    """Fail-closed selected-ray oracle configuration or integration error."""


def _fail(path: str, message: str) -> NoReturn:
    raise KerrSelectedOracleError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(path, "expected a finite number")
    return float(value)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "expected a boolean")
    return value


def _sequence(value: Any, length: int, path: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        _fail(path, f"expected {length} finite numbers")
    try:
        entries = tuple(value)
    except TypeError:
        _fail(path, f"expected {length} finite numbers")
    if len(entries) != length:
        _fail(path, f"expected {length} finite numbers")
    return tuple(_number(entry, f"{path}[{index}]") for index, entry in enumerate(entries))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _close(first: float, second: float, tolerance: float = 2.0e-11) -> bool:
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def _independent_isco_radius(
    spin_magnitude: float,
    orientation: str,
) -> float:
    """Bardeen--Press--Teukolsky ISCO root, implemented locally."""

    if not 0.0 <= spin_magnitude < 1.0:
        raise ValueError("spin magnitude must satisfy 0 <= a/M < 1")
    if orientation not in ("prograde", "retrograde"):
        raise ValueError("unsupported disk orientation")
    sign = 1.0 if orientation == "prograde" else -1.0
    z1 = 1.0 + (1.0 - spin_magnitude * spin_magnitude) ** (1.0 / 3.0) * (
        (1.0 + spin_magnitude) ** (1.0 / 3.0)
        + (1.0 - spin_magnitude) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * spin_magnitude * spin_magnitude + z1 * z1)
    radical = max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 3.0 + z2 - sign * math.sqrt(radical)


@dataclass(frozen=True, slots=True)
class KerrSelectedRayConfiguration:
    """Content extracted from one supported exact-Kerr/NT sampler descriptor."""

    mass_m: float
    spin_a_m: float
    observer_radius_m: float
    observer_theta_rad: float
    observer_phi_ks_rad: float
    capture_radius_m: float
    escape_radius_m: float
    isco_radius_m: float
    disk_outer_radius_m: float
    disk_orientation: str
    black_hole_mass_kg: float
    mass_accretion_rate_kg_s: float
    colour_correction: float
    angular_implementation_id: str
    angular_coefficient: float | None

    @property
    def dimensionless_spin(self) -> float:
        return self.spin_a_m / self.mass_m


def configuration_from_sampler_descriptor(
    descriptor: Mapping[str, Any],
) -> KerrSelectedRayConfiguration:
    """Strictly extract the closed v2 sampler configuration.

    The parser validates fields used by this oracle and the coordinate
    conventions connecting the production ZAMO screen to the independent BL
    initial state.  Unknown sampler, metric, screen, observer, angular-law, or
    boundary identities fail closed.
    """

    raw = _mapping(descriptor, "$.sampler.descriptor")
    if raw.get("implementationId") != SUPPORTED_SAMPLER_IMPLEMENTATION_ID:
        _fail(
            "$.sampler.descriptor.implementationId",
            f"only {SUPPORTED_SAMPLER_IMPLEMENTATION_ID!r} is supported",
        )
    if raw.get("version") != 2:
        _fail("$.sampler.descriptor.version", "only sampler version 2 is supported")
    try:
        metric = _mapping(raw["metric"], "$.sampler.descriptor.metric")
        observer = _mapping(raw["observer"], "$.sampler.descriptor.observer")
        termination = _mapping(
            raw["termination"], "$.sampler.descriptor.termination"
        )
        disk = _mapping(raw["disk"], "$.sampler.descriptor.disk")
        screen = _mapping(
            raw["screenConvention"], "$.sampler.descriptor.screenConvention"
        )
        angular_wrapper = _mapping(
            raw["angularEmission"], "$.sampler.descriptor.angularEmission"
        )
        angular = _mapping(
            angular_wrapper["descriptor"],
            "$.sampler.descriptor.angularEmission.descriptor",
        )
    except KeyError as error:
        _fail("$.sampler.descriptor", f"missing required field {error.args[0]!r}")

    if _string(metric.get("sourceId"), "$.sampler.descriptor.metric.sourceId") != (
        "analytic-kerr-kerr-schild"
    ):
        _fail("$.sampler.descriptor.metric.sourceId", "unsupported metric provider")
    if _boolean(
        metric.get("timeDependent"), "$.sampler.descriptor.metric.timeDependent"
    ):
        _fail("$.sampler.descriptor.metric.timeDependent", "metric must be stationary")
    mass = _number(metric.get("massM"), "$.sampler.descriptor.metric.massM")
    spin = _number(metric.get("spinAM"), "$.sampler.descriptor.metric.spinAM")
    if mass <= 0.0 or abs(spin) >= mass:
        _fail("$.sampler.descriptor.metric", "Novikov--Thorne requires M>0 and |a|<M")

    expected_screen = {
        "projection": "pinhole",
        "screenX": "ZAMO-right-negative-azimuthal",
        "screenY": "ZAMO-up-negative-polar",
        "viewForward": "ZAMO-negative-radial",
    }
    if dict(screen) != expected_screen:
        _fail("$.sampler.descriptor.screenConvention", "unsupported screen convention")
    if _string(observer.get("type"), "$.sampler.descriptor.observer.type") != (
        "Boyer-Lindquist-ZAMO"
    ):
        _fail("$.sampler.descriptor.observer.type", "unsupported observer type")
    radius = _number(observer.get("radiusM"), "$.sampler.descriptor.observer.radiusM")
    theta = _number(observer.get("thetaRad"), "$.sampler.descriptor.observer.thetaRad")
    phi_ks = _number(observer.get("phiKsRad"), "$.sampler.descriptor.observer.phiKsRad")
    coordinate_time = _number(
        observer.get("coordinateTimeM"),
        "$.sampler.descriptor.observer.coordinateTimeM",
    )
    event = _sequence(observer.get("event"), 4, "$.sampler.descriptor.observer.event")
    _sequence(
        observer.get("fourVelocity"),
        4,
        "$.sampler.descriptor.observer.fourVelocity",
    )
    if not 0.0 < theta < math.pi or abs(math.cos(theta)) <= 1.0e-10:
        _fail("$.sampler.descriptor.observer.thetaRad", "observer is degenerate")
    sine = math.sin(theta)
    expected_event = (
        coordinate_time,
        (radius * math.cos(phi_ks) - spin * math.sin(phi_ks)) * sine,
        (radius * math.sin(phi_ks) + spin * math.cos(phi_ks)) * sine,
        radius * math.cos(theta),
    )
    if any(not _close(actual, expected) for actual, expected in zip(event, expected_event)):
        _fail(
            "$.sampler.descriptor.observer.event",
            "event disagrees with the declared oblate ZAMO coordinates",
        )

    termination_spin = _number(
        termination.get("spinAM"), "$.sampler.descriptor.termination.spinAM"
    )
    if not _close(termination_spin, spin):
        _fail("$.sampler.descriptor.termination.spinAM", "termination spin disagrees")
    capture_target = _string(
        termination.get("captureTargetId"),
        "$.sampler.descriptor.termination.captureTargetId",
    )
    if capture_target not in (
        "analytic-kerr-event-horizon",
        "analytic-kerr-stretched-horizon",
    ):
        _fail("$.sampler.descriptor.termination.captureTargetId", "unsupported target")
    if _string(
        termination.get("escapeTargetId"),
        "$.sampler.descriptor.termination.escapeTargetId",
    ) != "analytic-kerr-escape-worldtube":
        _fail("$.sampler.descriptor.termination.escapeTargetId", "unsupported target")
    capture = _number(
        termination.get("captureRadiusM"),
        "$.sampler.descriptor.termination.captureRadiusM",
    )
    escape = _number(
        termination.get("escapeRadiusM"),
        "$.sampler.descriptor.termination.escapeRadiusM",
    )
    horizon = mass + math.sqrt((mass - abs(spin)) * (mass + abs(spin)))

    orientation = _string(
        disk.get("orientation"), "$.sampler.descriptor.disk.orientation"
    )
    if orientation not in ("prograde", "retrograde"):
        _fail("$.sampler.descriptor.disk.orientation", "unsupported orientation")
    declared_isco = _number(
        disk.get("iscoRadiusM"), "$.sampler.descriptor.disk.iscoRadiusM"
    )
    expected_isco = mass * _independent_isco_radius(abs(spin / mass), orientation)
    if not _close(declared_isco, expected_isco, 5.0e-11):
        _fail("$.sampler.descriptor.disk.iscoRadiusM", "ISCO disagrees with Kerr")
    outer = _number(
        disk.get("outerRadiusM"), "$.sampler.descriptor.disk.outerRadiusM"
    )
    if not horizon <= capture < declared_isco <= outer < radius < escape:
        _fail(
            "$.sampler.descriptor",
            "horizon/capture/disk/observer/escape radii are not strictly ordered",
        )

    angular_id = _string(
        angular.get("implementationId"),
        "$.sampler.descriptor.angularEmission.descriptor.implementationId",
    )
    coefficient: float | None
    if angular_id == "isotropic-angular-emission/v1":
        coefficient = None
    elif angular_id == "flux-conserving-linear-limb-darkening/v1":
        coefficient = _number(
            angular.get("coefficient"),
            "$.sampler.descriptor.angularEmission.descriptor.coefficient",
        )
        if coefficient < 0.0:
            _fail(
                "$.sampler.descriptor.angularEmission.descriptor.coefficient",
                "coefficient must be non-negative",
            )
    else:
        _fail(
            "$.sampler.descriptor.angularEmission.descriptor.implementationId",
            "unsupported angular law",
        )
    declared_angular_hash = _string(
        angular_wrapper.get("descriptorSha256"),
        "$.sampler.descriptor.angularEmission.descriptorSha256",
    )
    if declared_angular_hash != _canonical_sha256(angular):
        _fail(
            "$.sampler.descriptor.angularEmission.descriptorSha256",
            "hash does not bind the angular descriptor",
        )

    black_hole_mass = _number(
        disk.get("blackHoleMassKg"),
        "$.sampler.descriptor.disk.blackHoleMassKg",
    )
    accretion_rate = _number(
        disk.get("massAccretionRateKgS"),
        "$.sampler.descriptor.disk.massAccretionRateKgS",
    )
    colour_correction = _number(
        disk.get("colourCorrection"),
        "$.sampler.descriptor.disk.colourCorrection",
    )
    if black_hole_mass <= 0.0 or accretion_rate < 0.0 or colour_correction <= 0.0:
        _fail("$.sampler.descriptor.disk", "disk SI parameters are invalid")

    return KerrSelectedRayConfiguration(
        mass_m=mass,
        spin_a_m=spin,
        observer_radius_m=radius,
        observer_theta_rad=theta,
        observer_phi_ks_rad=phi_ks,
        capture_radius_m=capture,
        escape_radius_m=escape,
        isco_radius_m=declared_isco,
        disk_outer_radius_m=outer,
        disk_orientation=orientation,
        black_hole_mass_kg=black_hole_mass,
        mass_accretion_rate_kg_s=accretion_rate,
        colour_correction=colour_correction,
        angular_implementation_id=angular_id,
        angular_coefficient=coefficient,
    )


@dataclass(frozen=True, slots=True)
class FixedRk4Options:
    """Finite work and convergence controls for a selected ray."""

    step_m: float = 0.005
    maximum_affine_length_m: float = 250.0
    maximum_steps: int = 200_000
    event_bisection_iterations: int = 52
    pole_guard_sine: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("step_m", "maximum_affine_length_m", "pole_guard_sine"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if type(self.maximum_steps) is not int or self.maximum_steps < 1:
            raise ValueError("maximum_steps must be a positive integer")
        if (
            type(self.event_bisection_iterations) is not int
            or self.event_bisection_iterations < 16
            or self.event_bisection_iterations > 80
        ):
            raise ValueError("event_bisection_iterations must lie in [16, 80]")


@dataclass(frozen=True, slots=True)
class KerrPhotonConstants:
    energy: float
    angular_momentum_z: float
    carter_q: float
    carter_k: float


@dataclass(frozen=True, slots=True)
class SelectedRayResult:
    screen_x: float
    screen_y: float
    outcome: RayOutcome
    affine_length_m: float
    terminal_radius_m: float
    disk_radius_m: float | None
    frequency_shift_g: float | None
    emission_angle_cosine: float | None
    constants: KerrPhotonConstants
    maximum_hamiltonian_residual: float
    maximum_relative_carter_drift: float
    steps: int


@dataclass(frozen=True, slots=True)
class SelectedRayRefinement:
    coarse: SelectedRayResult
    fine: SelectedRayResult
    outcome_agrees: bool
    disk_radius_difference_m: float | None
    relative_g_difference: float | None


# State order: (t/M, r/M, theta, phi_BL, p_t, p_r, p_theta, p_phi).
_State = tuple[float, float, float, float, float, float, float, float]


def _inverse_metric_and_derivatives(
    radius: float,
    theta: float,
    spin: float,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    """Return BL inverse metric plus analytic r/theta derivatives for M=1."""

    sine = math.sin(theta)
    cosine = math.cos(theta)
    if abs(sine) <= 1.0e-15:
        raise KerrSelectedOracleError("fixed RK4 entered the BL polar singularity")
    sine2 = sine * sine
    sigma = radius * radius + spin * spin * cosine * cosine
    delta = radius * radius - 2.0 * radius + spin * spin
    radial_sum = radius * radius + spin * spin
    big_a = radial_sum * radial_sum - spin * spin * delta * sine2
    denominator = sigma * delta
    phi_denominator = denominator * sine2
    phi_numerator = delta - spin * spin * sine2
    if sigma <= 0.0 or delta <= 0.0 or denominator <= 0.0:
        raise KerrSelectedOracleError("fixed RK4 sampled at or inside the BL horizon")

    dsigma_r = 2.0 * radius
    dsigma_theta = -2.0 * spin * spin * cosine * sine
    ddelta_r = 2.0 * radius - 2.0
    dbig_a_r = 4.0 * radius * radial_sum - spin * spin * ddelta_r * sine2
    dbig_a_theta = -2.0 * spin * spin * delta * sine * cosine
    ddenominator_r = dsigma_r * delta + sigma * ddelta_r
    ddenominator_theta = dsigma_theta * delta
    dphi_denominator_r = ddenominator_r * sine2
    dphi_denominator_theta = (
        ddenominator_theta * sine2 + denominator * 2.0 * sine * cosine
    )
    dphi_numerator_r = ddelta_r
    dphi_numerator_theta = -2.0 * spin * spin * sine * cosine

    def quotient_derivative(
        numerator: float,
        numerator_derivative: float,
        denominator_value: float,
        denominator_derivative: float,
    ) -> float:
        return (
            numerator_derivative * denominator_value
            - numerator * denominator_derivative
        ) / (denominator_value * denominator_value)

    inverse = [[0.0] * 4 for _ in range(4)]
    radial = [[0.0] * 4 for _ in range(4)]
    polar = [[0.0] * 4 for _ in range(4)]
    inverse[0][0] = -big_a / denominator
    inverse[0][3] = inverse[3][0] = -2.0 * spin * radius / denominator
    inverse[1][1] = delta / sigma
    inverse[2][2] = 1.0 / sigma
    inverse[3][3] = phi_numerator / phi_denominator

    radial[0][0] = quotient_derivative(
        -big_a, -dbig_a_r, denominator, ddenominator_r
    )
    polar[0][0] = quotient_derivative(
        -big_a, -dbig_a_theta, denominator, ddenominator_theta
    )
    radial[0][3] = radial[3][0] = quotient_derivative(
        -2.0 * spin * radius,
        -2.0 * spin,
        denominator,
        ddenominator_r,
    )
    polar[0][3] = polar[3][0] = quotient_derivative(
        -2.0 * spin * radius,
        0.0,
        denominator,
        ddenominator_theta,
    )
    radial[1][1] = quotient_derivative(delta, ddelta_r, sigma, dsigma_r)
    polar[1][1] = quotient_derivative(delta, 0.0, sigma, dsigma_theta)
    radial[2][2] = -dsigma_r / (sigma * sigma)
    polar[2][2] = -dsigma_theta / (sigma * sigma)
    radial[3][3] = quotient_derivative(
        phi_numerator,
        dphi_numerator_r,
        phi_denominator,
        dphi_denominator_r,
    )
    polar[3][3] = quotient_derivative(
        phi_numerator,
        dphi_numerator_theta,
        phi_denominator,
        dphi_denominator_theta,
    )
    return (
        tuple(tuple(row) for row in inverse),
        tuple(tuple(row) for row in radial),
        tuple(tuple(row) for row in polar),
    )


def _quadratic(matrix: Sequence[Sequence[float]], covector: Sequence[float]) -> float:
    return math.fsum(
        covector[row] * matrix[row][column] * covector[column]
        for row in range(4)
        for column in range(4)
    )


def _rhs(state: _State, spin: float) -> _State:
    _time, radius, theta, _phi, p_t, p_r, p_theta, p_phi = state
    inverse, derivative_r, derivative_theta = _inverse_metric_and_derivatives(
        radius, theta, spin
    )
    covector = (p_t, p_r, p_theta, p_phi)
    coordinate_rate = tuple(
        math.fsum(inverse[row][column] * covector[column] for column in range(4))
        for row in range(4)
    )
    return (
        coordinate_rate[0],
        coordinate_rate[1],
        coordinate_rate[2],
        coordinate_rate[3],
        0.0,
        -0.5 * _quadratic(derivative_r, covector),
        -0.5 * _quadratic(derivative_theta, covector),
        0.0,
    )


def _rk4_step(state: _State, step: float, spin: float) -> _State:
    k1 = _rhs(state, spin)
    second = tuple(state[index] + 0.5 * step * k1[index] for index in range(8))
    k2 = _rhs(second, spin)  # type: ignore[arg-type]
    third = tuple(state[index] + 0.5 * step * k2[index] for index in range(8))
    k3 = _rhs(third, spin)  # type: ignore[arg-type]
    fourth = tuple(state[index] + step * k3[index] for index in range(8))
    k4 = _rhs(fourth, spin)  # type: ignore[arg-type]
    return tuple(  # type: ignore[return-value]
        state[index]
        + step
        * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        / 6.0
        for index in range(8)
    )


def _initial_state(
    configuration: KerrSelectedRayConfiguration,
    screen_x: float,
    screen_y: float,
) -> _State:
    radius = configuration.observer_radius_m / configuration.mass_m
    theta = configuration.observer_theta_rad
    spin = configuration.dimensionless_spin
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
    inverse_norm = 1.0 / math.sqrt(1.0 + screen_x * screen_x + screen_y * screen_y)
    p_phi = -inverse_norm * screen_x * sine * math.sqrt(big_a / sigma)
    p_theta = -inverse_norm * screen_y * math.sqrt(sigma)
    p_r = -inverse_norm * math.sqrt(sigma / delta)
    p_t = lapse - omega * p_phi
    return (0.0, radius, theta, 0.0, p_t, p_r, p_theta, p_phi)


def photon_constants(state: _State, spin: float) -> KerrPhotonConstants:
    """Independently recover E, Lz, Carter Q and K from a BL state."""

    theta = state[2]
    p_t, _p_r, p_theta, p_phi = state[4:]
    energy = -p_t
    sine = math.sin(theta)
    cosine = math.cos(theta)
    if abs(sine) <= 1.0e-15:
        raise KerrSelectedOracleError("Carter constant is singular on the BL axis")
    carter_q = p_theta * p_theta + cosine * cosine * (
        p_phi * p_phi / (sine * sine) - spin * spin * energy * energy
    )
    carter_k = carter_q + (p_phi - spin * energy) ** 2
    if not all(math.isfinite(value) for value in (energy, p_phi, carter_q, carter_k)):
        raise KerrSelectedOracleError("Kerr photon constants became non-finite")
    return KerrPhotonConstants(energy, p_phi, carter_q, carter_k)


def _hamiltonian_residual(state: _State, spin: float) -> float:
    inverse, _radial, _polar = _inverse_metric_and_derivatives(
        state[1], state[2], spin
    )
    return abs(_quadratic(inverse, state[4:]))


def _locate_crossing(
    start: _State,
    step: float,
    spin: float,
    value,
    iterations: int,
) -> tuple[float, _State]:
    start_value = float(value(start))
    end = _rk4_step(start, step, spin)
    end_value = float(value(end))
    if start_value == 0.0:
        return 0.0, start
    if end_value == 0.0:
        return 1.0, end
    if (start_value > 0.0) == (end_value > 0.0):
        raise KerrSelectedOracleError("event bisection lacks a sign-changing bracket")
    lower = 0.0
    upper = 1.0
    candidate = end
    for _iteration in range(iterations):
        middle = 0.5 * (lower + upper)
        candidate = _rk4_step(start, middle * step, spin)
        middle_value = float(value(candidate))
        if middle_value == 0.0:
            lower = upper = middle
            break
        if (middle_value > 0.0) == (start_value > 0.0):
            lower = middle
        else:
            upper = middle
    fraction = 0.5 * (lower + upper)
    return fraction, _rk4_step(start, fraction * step, spin)


def _disk_orbit(
    configuration: KerrSelectedRayConfiguration,
    radius_m: float,
) -> tuple[float, float, float, float]:
    """Return dimensionless Omega, orbital E/Lz and u^t, locally implemented."""

    radius = radius_m / configuration.mass_m
    spin_magnitude = abs(configuration.dimensionless_spin)
    orientation_sign = 1.0 if configuration.disk_orientation == "prograde" else -1.0
    spin_axis_sign = -1.0 if configuration.spin_a_m < 0.0 else 1.0
    sign = orientation_sign
    root = math.sqrt(radius)
    radius_three_halves = radius * root
    radicand = radius_three_halves - 3.0 * root + 2.0 * sign * spin_magnitude
    if radicand <= 0.0:
        raise KerrSelectedOracleError("disk circular orbit is not timelike")
    denominator = radius**0.75 * math.sqrt(radicand)
    relative_omega = sign / (radius_three_halves + sign * spin_magnitude)
    orbital_energy = (
        radius_three_halves - 2.0 * root + sign * spin_magnitude
    ) / denominator
    relative_lz = sign * (
        radius * radius - 2.0 * sign * spin_magnitude * root + spin_magnitude**2
    ) / denominator
    omega = spin_axis_sign * relative_omega
    angular_momentum = spin_axis_sign * relative_lz
    redshift_denominator = orbital_energy - omega * angular_momentum
    if redshift_denominator <= 0.0:
        raise KerrSelectedOracleError("disk circular redshift is not positive")
    return omega, orbital_energy, angular_momentum, 1.0 / redshift_denominator


def _disk_transfer(
    configuration: KerrSelectedRayConfiguration,
    state: _State,
) -> tuple[float, float]:
    radius_m = state[1] * configuration.mass_m
    omega, _orbital_energy, _orbital_lz, u_t = _disk_orbit(
        configuration, radius_m
    )
    constants = photon_constants(state, configuration.dimensionless_spin)
    emitted_frequency_ratio = u_t * (
        -constants.energy + omega * constants.angular_momentum_z
    )
    if emitted_frequency_ratio <= 0.0 or not math.isfinite(emitted_frequency_ratio):
        raise KerrSelectedOracleError("disk-frame photon frequency is not positive")
    shift = 1.0 / emitted_frequency_ratio
    cosine = math.sqrt(max(0.0, constants.carter_q)) / (
        state[1] * emitted_frequency_ratio
    )
    if cosine < -2.0e-10 or cosine > 1.0 + 2.0e-10:
        raise KerrSelectedOracleError("disk emission angle is outside [0, 1]")
    return shift, min(1.0, max(0.0, cosine))


def trace_selected_ray(
    configuration: KerrSelectedRayConfiguration,
    screen_x: float,
    screen_y: float,
    options: FixedRk4Options = FixedRk4Options(),
) -> SelectedRayResult:
    """Trace one independent fixed-step BL Hamiltonian calibration ray."""

    if not isinstance(configuration, KerrSelectedRayConfiguration):
        raise TypeError("configuration must be KerrSelectedRayConfiguration")
    if not isinstance(options, FixedRk4Options):
        raise TypeError("options must be FixedRk4Options")
    for value, name in ((screen_x, "screen_x"), (screen_y, "screen_y")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
    screen_x = float(screen_x)
    screen_y = float(screen_y)
    mass = configuration.mass_m
    spin = configuration.dimensionless_spin
    step = options.step_m / mass
    maximum_affine = options.maximum_affine_length_m / mass
    capture = configuration.capture_radius_m / mass
    escape = configuration.escape_radius_m / mass
    isco = configuration.isco_radius_m / mass
    outer = configuration.disk_outer_radius_m / mass
    state = _initial_state(configuration, screen_x, screen_y)
    initial_constants = photon_constants(state, spin)
    maximum_hamiltonian = _hamiltonian_residual(state, spin)
    maximum_carter_drift = 0.0
    affine = 0.0

    def finish(outcome: RayOutcome, terminal: _State, steps: int) -> SelectedRayResult:
        constants = photon_constants(terminal, spin)
        disk_radius = None
        shift = None
        cosine = None
        if outcome == "disk":
            disk_radius = terminal[1] * mass
            shift, cosine = _disk_transfer(configuration, terminal)
        return SelectedRayResult(
            screen_x=screen_x,
            screen_y=screen_y,
            outcome=outcome,
            affine_length_m=affine * mass,
            terminal_radius_m=terminal[1] * mass,
            disk_radius_m=disk_radius,
            frequency_shift_g=shift,
            emission_angle_cosine=cosine,
            constants=constants,
            maximum_hamiltonian_residual=maximum_hamiltonian,
            maximum_relative_carter_drift=maximum_carter_drift,
            steps=steps,
        )

    maximum_steps = min(options.maximum_steps, math.ceil(maximum_affine / step) + 1)
    for step_index in range(1, maximum_steps + 1):
        remaining = maximum_affine - affine
        if remaining <= 0.0:
            return finish("unresolved", state, step_index - 1)
        actual_step = min(step, remaining)
        candidate = _rk4_step(state, actual_step, spin)
        if not all(math.isfinite(value) for value in candidate):
            raise KerrSelectedOracleError("fixed RK4 state became non-finite")
        if abs(math.sin(candidate[2])) <= options.pole_guard_sine:
            raise KerrSelectedOracleError(
                "selected ray reached the guarded BL polar-coordinate axis"
            )

        events: list[tuple[float, RayOutcome, _State]] = []
        capture_start = state[1] - capture
        capture_end = candidate[1] - capture
        if capture_start > 0.0 and capture_end <= 0.0:
            fraction, located = _locate_crossing(
                state,
                actual_step,
                spin,
                lambda entry: entry[1] - capture,
                options.event_bisection_iterations,
            )
            events.append((fraction, "captured", located))

        escape_start = state[1] - escape
        escape_end = candidate[1] - escape
        if escape_start < 0.0 and escape_end >= 0.0 and candidate[5] > 0.0:
            fraction, located = _locate_crossing(
                state,
                actual_step,
                spin,
                lambda entry: entry[1] - escape,
                options.event_bisection_iterations,
            )
            events.append((fraction, "escaped", located))

        plane_start = math.cos(state[2])
        plane_end = math.cos(candidate[2])
        if plane_start != 0.0 and (
            plane_end == 0.0 or (plane_start > 0.0) != (plane_end > 0.0)
        ):
            fraction, located = _locate_crossing(
                state,
                actual_step,
                spin,
                lambda entry: math.cos(entry[2]),
                options.event_bisection_iterations,
            )
            if isco <= located[1] <= outer:
                events.append((fraction, "disk", located))

        if events:
            fraction, outcome, terminal = min(events, key=lambda entry: entry[0])
            affine += fraction * actual_step
            maximum_hamiltonian = max(
                maximum_hamiltonian,
                _hamiltonian_residual(terminal, spin),
            )
            terminal_constants = photon_constants(terminal, spin)
            maximum_carter_drift = max(
                maximum_carter_drift,
                abs(terminal_constants.carter_q - initial_constants.carter_q)
                / max(
                    1.0,
                    abs(initial_constants.carter_q),
                    abs(initial_constants.carter_k),
                ),
            )
            return finish(outcome, terminal, step_index)

        state = candidate
        affine += actual_step
        maximum_hamiltonian = max(
            maximum_hamiltonian, _hamiltonian_residual(state, spin)
        )
        current_constants = photon_constants(state, spin)
        maximum_carter_drift = max(
            maximum_carter_drift,
            abs(current_constants.carter_q - initial_constants.carter_q)
            / max(
                1.0,
                abs(initial_constants.carter_q),
                abs(initial_constants.carter_k),
            ),
        )
    return finish("unresolved", state, maximum_steps)


def trace_selected_ray_refined(
    configuration: KerrSelectedRayConfiguration,
    screen_x: float,
    screen_y: float,
    options: FixedRk4Options = FixedRk4Options(),
) -> SelectedRayRefinement:
    """Require distinct fixed-step ``h`` and ``h/2`` selected-ray traces."""

    coarse = trace_selected_ray(configuration, screen_x, screen_y, options)
    fine = trace_selected_ray(
        configuration,
        screen_x,
        screen_y,
        FixedRk4Options(
            step_m=0.5 * options.step_m,
            maximum_affine_length_m=options.maximum_affine_length_m,
            maximum_steps=2 * options.maximum_steps,
            event_bisection_iterations=options.event_bisection_iterations,
            pole_guard_sine=options.pole_guard_sine,
        ),
    )
    radius_difference = None
    relative_g_difference = None
    if coarse.disk_radius_m is not None and fine.disk_radius_m is not None:
        radius_difference = abs(coarse.disk_radius_m - fine.disk_radius_m)
    if coarse.frequency_shift_g is not None and fine.frequency_shift_g is not None:
        relative_g_difference = abs(coarse.frequency_shift_g - fine.frequency_shift_g) / max(
            abs(coarse.frequency_shift_g), abs(fine.frequency_shift_g)
        )
    return SelectedRayRefinement(
        coarse=coarse,
        fine=fine,
        outcome_agrees=coarse.outcome == fine.outcome,
        disk_radius_difference_m=radius_difference,
        relative_g_difference=relative_g_difference,
    )


def _angular_multiplier(
    configuration: KerrSelectedRayConfiguration,
    emission_angle_cosine: float,
) -> float:
    if configuration.angular_implementation_id == "isotropic-angular-emission/v1":
        return 1.0
    coefficient = configuration.angular_coefficient
    if coefficient is None:
        raise KerrSelectedOracleError("linear angular law lacks its coefficient")
    return (1.0 + coefficient * emission_angle_cosine) / (
        1.0 + 2.0 * coefficient / 3.0
    )


def selected_ray_observed_intensities_nu(
    configuration: KerrSelectedRayConfiguration,
    result: SelectedRayResult,
    observer_frequencies_hz: Sequence[float],
) -> tuple[float, ...]:
    """Recompute selected-ray I_nu with a declared shared NT radial scalar."""

    frequencies = tuple(float(value) for value in observer_frequencies_hz)
    if not frequencies or any(not math.isfinite(value) or value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be finite, positive, and non-empty")
    if result.outcome != "disk":
        return tuple(0.0 for _value in frequencies)
    if (
        result.disk_radius_m is None
        or result.frequency_shift_g is None
        or result.emission_angle_cosine is None
    ):
        raise KerrSelectedOracleError("disk ray lacks transfer diagnostics")
    radius_over_mass = result.disk_radius_m / configuration.mass_m
    flux_shape = page_thorne_flux_shape(
        radius_over_mass,
        abs(configuration.dimensionless_spin),
        configuration.disk_orientation,
    )
    if flux_shape == 0.0 or configuration.mass_accretion_rate_kg_s == 0.0:
        return tuple(0.0 for _value in frequencies)
    maximum_log = math.log(sys.float_info.max)
    minimum_log = math.log(math.ulp(0.0))
    log_surface_flux = (
        6.0 * math.log(LIGHT_SPEED_M_S)
        + math.log(configuration.mass_accretion_rate_kg_s)
        + math.log(flux_shape)
        - math.log(4.0 * math.pi)
        - 2.0 * math.log(GRAVITATIONAL_CONSTANT_M3_KG_S2)
        - 2.0 * math.log(configuration.black_hole_mass_kg)
    )
    if log_surface_flux > maximum_log or log_surface_flux < minimum_log:
        raise KerrSelectedOracleError(
            "shared Page--Thorne surface flux lies outside binary64"
        )
    log_temperature = (
        math.log(configuration.colour_correction)
        + 0.25
        * (log_surface_flux - math.log(STEFAN_BOLTZMANN_W_M2_K4))
    )
    angular = _angular_multiplier(configuration, result.emission_angle_cosine)
    shift = result.frequency_shift_g
    observed: list[float] = []
    for observer_frequency in frequencies:
        emitted_frequency = observer_frequency / shift
        log_exponent = (
            math.log(PLANCK_CONSTANT_J_S)
            + math.log(emitted_frequency)
            - math.log(BOLTZMANN_CONSTANT_J_K)
            - log_temperature
        )
        if log_exponent > maximum_log:
            observed.append(0.0)
            continue
        if log_exponent < minimum_log:
            log_denominator = log_exponent
        else:
            exponent = math.exp(log_exponent)
            log_denominator = (
                exponent if exponent > 50.0 else math.log(math.expm1(exponent))
            )
        log_value = (
            3.0 * math.log(shift)
            + math.log(angular)
            + math.log(2.0 * PLANCK_CONSTANT_J_S)
            - 2.0 * math.log(LIGHT_SPEED_M_S)
            + 3.0 * math.log(emitted_frequency)
            - log_denominator
            - 4.0 * math.log(configuration.colour_correction)
        )
        if log_value < minimum_log:
            value = 0.0
        elif log_value > maximum_log:
            raise KerrSelectedOracleError(
                "selected-ray spectral intensity overflowed binary64"
            )
        else:
            value = math.exp(log_value)
        if not math.isfinite(value) or value < 0.0:
            raise KerrSelectedOracleError("selected-ray spectral intensity is invalid")
        observed.append(value)
    return tuple(observed)


__all__ = (
    "FixedRk4Options",
    "KerrPhotonConstants",
    "KerrSelectedOracleError",
    "KerrSelectedRayConfiguration",
    "SELECTED_RAY_ORACLE_IMPLEMENTATION_ID",
    "SUPPORTED_SAMPLER_IMPLEMENTATION_ID",
    "SelectedRayRefinement",
    "SelectedRayResult",
    "configuration_from_sampler_descriptor",
    "photon_constants",
    "selected_ray_observed_intensities_nu",
    "trace_selected_ray",
    "trace_selected_ray_refined",
)
