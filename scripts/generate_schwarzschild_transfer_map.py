#!/usr/bin/env python3
"""Generate a renderable stationary Schwarzschild transfer map.

The map is camera-specific.  It uses the exact Schwarzschild null-orbit
equation

    d²u/dψ² = -u + 3u²,  u = M/r,

through its first integral

    (du/dψ)² + u² - 2u³ = 1/b².

Spherical symmetry makes the radial solution depend only on the pixel's
angular distance from the optical axis.  We solve each distinct radius once,
then rotate the asymptotic direction into the pixel's ray plane.  This is an
analytic stationary reference product, not numerical-relativity data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

try:
    from scripts.generate_nr_contract_fixture import manifest as fixture_manifest
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from generate_nr_contract_fixture import manifest as fixture_manifest


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA_PATH: Final = ROOT / "schemas" / "nr-transfer-map-v1.schema.json"
HELPER_PATH: Final = ROOT / "scripts" / "generate_nr_contract_fixture.py"
DEFAULT_OUTPUT_DIR: Final = (
    ROOT / "assets" / "transfer-maps" / "schwarzschild-reference-v1"
)

RECORD: Final = struct.Struct("<7fBBH")
RECORD_BYTES: Final = RECORD.size
WIDTH: Final = 1024
HEIGHT: Final = 576
TILE_HEIGHT: Final = 64
VERTICAL_FOV_RAD: Final = math.radians(40.0)
OBSERVER_RADIUS_M: Final = 40.0
ESCAPE_RADIUS_M: Final = 1_000.0
CAPTURE_RADIUS_M: Final = 2.02
CRITICAL_IMPACT_M: Final = 3.0 * math.sqrt(3.0)

FINE_ABSOLUTE_TOLERANCE: Final = 2.0e-11
FINE_RELATIVE_TOLERANCE: Final = 2.0e-11
COARSE_ABSOLUTE_TOLERANCE: Final = 2.0e-8
COARSE_RELATIVE_TOLERANCE: Final = 2.0e-8
TIME_ABSOLUTE_TOLERANCE_M: Final = 2.0e-8
TIME_RELATIVE_TOLERANCE: Final = 2.0e-10
FLOAT32_DIRECTION_ERROR_RAD: Final = 2.0e-7

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

# A proper right-handed rotation selected so that the camera centre looks
# toward ICRS RA=270°, Dec=0°, camera-right points toward ICRS +X, and
# camera-up points toward the ICRS north pole. Rows multiply world columns.
WORLD_TO_ICRS: Final = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
ICRS_TO_WORLD: Final = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


@dataclass(frozen=True)
class RadialSolution:
    """One spherical-symmetry radial solution shared by equal-radius pixels."""

    outcome: int
    terminal_angle_rad: float
    coordinate_lookback_time_m: float
    frequency_shift_g: float
    null_residual: float
    projection_error_px: float
    refinement_difference_rad: float


@dataclass(frozen=True)
class GenerationReport:
    width: int
    height: int
    chunks: int
    records: int
    escaped: int
    captured: int
    unresolved: int
    unique_radial_rays: int
    shadow_diameter_deg: float
    frequency_shift_g: float
    max_refinement_difference_rad: float
    p95_projection_error_px: float
    p99_null_residual: float
    max_null_residual: float
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


def lapse_squared(radius_m: float) -> float:
    return 1.0 - 2.0 / radius_m


def critical_shadow_radius_rad(observer_radius_m: float = OBSERVER_RADIUS_M) -> float:
    """Angular shadow radius for a static Schwarzschild observer."""
    return math.asin(
        CRITICAL_IMPACT_M
        * math.sqrt(lapse_squared(observer_radius_m))
        / observer_radius_m
    )


def impact_parameter_from_screen_radius(
    screen_radius: float,
    observer_radius_m: float = OBSERVER_RADIUS_M,
) -> float:
    """Map pinhole radius tan(alpha) to conserved b=L/E."""
    sin_alpha = screen_radius / math.sqrt(1.0 + screen_radius * screen_radius)
    return observer_radius_m * sin_alpha / math.sqrt(
        lapse_squared(observer_radius_m)
    )


def orbit_polynomial(u: float, impact_m: float) -> float:
    return 1.0 / (impact_m * impact_m) - u * u + 2.0 * u * u * u


def orbit_polynomial_derivative(u: float) -> float:
    return -2.0 * u + 6.0 * u * u


def classify_impact_parameter(impact_m: float) -> int:
    """Classify b without pretending the exact photon separatrix terminates."""
    if impact_m > CRITICAL_IMPACT_M:
        return OUTCOME_ESCAPED
    if impact_m < CRITICAL_IMPACT_M:
        return OUTCOME_CAPTURED
    return OUTCOME_UNRESOLVED


def outer_turning_point(impact_m: float) -> float:
    """Return the outer positive root of the null-orbit first integral."""
    if impact_m <= CRITICAL_IMPACT_M:
        raise ValueError("captured rays have no outer turning point")
    low = 1.0 / OBSERVER_RADIUS_M
    high = 1.0 / 3.0
    if orbit_polynomial(low, impact_m) <= 0.0:
        raise ValueError("ray is not inward-going from the declared observer")
    if orbit_polynomial(high, impact_m) >= 0.0:
        raise ValueError("impact parameter does not have an outer turning root")
    for _ in range(96):
        middle = 0.5 * (low + high)
        if orbit_polynomial(middle, impact_m) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _adaptive_simpson(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_depth: int = 28,
) -> float:
    """Deterministic adaptive Simpson integration with a Richardson correction."""
    if not upper > lower:
        return 0.0

    midpoint = 0.5 * (lower + upper)
    f_lower = function(lower)
    f_middle = function(midpoint)
    f_upper = function(upper)
    whole = (upper - lower) * (f_lower + 4.0 * f_middle + f_upper) / 6.0

    def recurse(
        a: float,
        b: float,
        fa: float,
        fm: float,
        fb: float,
        estimate: float,
        absolute_budget: float,
        depth: int,
    ) -> float:
        middle = 0.5 * (a + b)
        left_middle = 0.5 * (a + middle)
        right_middle = 0.5 * (middle + b)
        f_left_middle = function(left_middle)
        f_right_middle = function(right_middle)
        left = (
            (middle - a)
            * (fa + 4.0 * f_left_middle + fm)
            / 6.0
        )
        right = (
            (b - middle)
            * (fm + 4.0 * f_right_middle + fb)
            / 6.0
        )
        refined = left + right
        correction = (refined - estimate) / 15.0
        error = abs(correction)
        tolerance = max(absolute_budget, relative_tolerance * abs(refined))
        if error <= tolerance:
            return refined + correction
        if depth <= 0:
            raise ArithmeticError(
                "adaptive Simpson recursion budget exhausted "
                f"on [{lower:.9g}, {upper:.9g}] with error {error:.3e}"
            )
        return recurse(
            a,
            middle,
            fa,
            f_left_middle,
            fm,
            left,
            0.5 * absolute_budget,
            depth - 1,
        ) + recurse(
            middle,
            b,
            fm,
            f_right_middle,
            fb,
            right,
            0.5 * absolute_budget,
            depth - 1,
        )

    return recurse(
        lower,
        upper,
        f_lower,
        f_middle,
        f_upper,
        whole,
        absolute_tolerance,
        max_depth,
    )


def _turning_integrand(
    impact_m: float,
    turning_u: float,
    coordinate_time: bool,
) -> Callable[[float], float]:
    derivative_limit = -orbit_polynomial_derivative(turning_u)
    if derivative_limit <= 0.0:
        raise ArithmeticError("outer turning point has an invalid derivative")
    root_limit = math.sqrt(derivative_limit)

    def integrand(s: float) -> float:
        if s == 0.0:
            base = 2.0 / root_limit
            if not coordinate_time:
                return base
            f = 1.0 - 2.0 * turning_u
            return base / (impact_m * turning_u * turning_u * f)
        u = turning_u - s * s
        if u == turning_u:
            base = 2.0 / root_limit
            if not coordinate_time:
                return base
            f = 1.0 - 2.0 * turning_u
            return base / (impact_m * turning_u * turning_u * f)
        # Evaluate F(u)-F(u_turn) in factored form.  Directly subtracting
        # u² and 2u³ next to the turning root creates an artificial noisy
        # endpoint that can defeat a tight adaptive tolerance.
        radicand = (u - turning_u) * (
            2.0 * (u * u + u * turning_u + turning_u * turning_u)
            - (u + turning_u)
        )
        if radicand <= 0.0:
            raise ArithmeticError(
                f"negative orbit radicand {radicand:.3e} at u={u:.9g}"
            )
        base = 2.0 * s / math.sqrt(radicand)
        if not coordinate_time:
            return base
        f = 1.0 - 2.0 * u
        return base / (impact_m * u * u * f)

    return integrand


def _integral_to_turn(
    lower_u: float,
    turning_u: float,
    impact_m: float,
    *,
    coordinate_time: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    upper_s = math.sqrt(turning_u - lower_u)
    return _adaptive_simpson(
        _turning_integrand(impact_m, turning_u, coordinate_time),
        0.0,
        upper_s,
        absolute_tolerance,
        relative_tolerance,
    )


def _direct_integrand(
    impact_m: float,
    coordinate_time: bool,
) -> Callable[[float], float]:
    def integrand(u: float) -> float:
        radicand = orbit_polynomial(u, impact_m)
        if radicand <= 0.0:
            raise ArithmeticError(
                f"non-positive captured-orbit radicand {radicand:.3e}"
            )
        base = 1.0 / math.sqrt(radicand)
        if not coordinate_time:
            return base
        return base / (impact_m * u * u * (1.0 - 2.0 * u))

    return integrand


def _null_residual(impact_m: float, samples_u: tuple[float, ...]) -> float:
    """Reconstruct max |g(k,k)| with u_observer·k_observer normalized to one."""
    observer_lapse_squared = lapse_squared(OBSERVER_RADIUS_M)
    maximum = 0.0
    for u in samples_u:
        f = 1.0 - 2.0 * u
        polynomial = orbit_polynomial(u, impact_m)
        p_squared = max(0.0, polynomial)
        normalized = (
            (-1.0 + impact_m * impact_m * p_squared) / f
            + impact_m * impact_m * u * u
        )
        maximum = max(maximum, abs(observer_lapse_squared * normalized))
    return maximum


def _regularized_path_samples(
    lower_u: float,
    upper_u: float,
    count: int = 129,
) -> tuple[float, ...]:
    """Sample a radial leg densely, with extra resolution near its upper end."""
    if count < 2:
        raise ValueError("path audit requires at least two samples")
    span = upper_u - lower_u
    return tuple(
        upper_u - span * (1.0 - index / (count - 1)) ** 2
        for index in range(count)
    )


def solve_radial_ray(
    impact_m: float,
    focal_pixels_per_radian: float,
) -> RadialSolution:
    """Solve one exact Schwarzschild orbital plane and estimate refinement error."""
    observer_u = 1.0 / OBSERVER_RADIUS_M
    capture_u = 1.0 / CAPTURE_RADIUS_M
    boundary_u = 1.0 / ESCAPE_RADIUS_M

    outcome = classify_impact_parameter(impact_m)
    if outcome == OUTCOME_UNRESOLVED:
        # The exact b=3*sqrt(3)M ray approaches r=3M asymptotically. It
        # reaches neither declared termination surface in finite coordinate
        # time and therefore must not be mislabelled captured or escaped.
        return RadialSolution(
            outcome=OUTCOME_UNRESOLVED,
            terminal_angle_rad=0.0,
            coordinate_lookback_time_m=0.0,
            frequency_shift_g=0.0,
            null_residual=0.0,
            projection_error_px=0.0,
            refinement_difference_rad=0.0,
        )

    if outcome == OUTCOME_ESCAPED:
        turning_u = outer_turning_point(impact_m)
        fine_angle = _integral_to_turn(
            observer_u,
            turning_u,
            impact_m,
            coordinate_time=False,
            absolute_tolerance=FINE_ABSOLUTE_TOLERANCE,
            relative_tolerance=FINE_RELATIVE_TOLERANCE,
        ) + _integral_to_turn(
            0.0,
            turning_u,
            impact_m,
            coordinate_time=False,
            absolute_tolerance=FINE_ABSOLUTE_TOLERANCE,
            relative_tolerance=FINE_RELATIVE_TOLERANCE,
        )
        coarse_angle = _integral_to_turn(
            observer_u,
            turning_u,
            impact_m,
            coordinate_time=False,
            absolute_tolerance=COARSE_ABSOLUTE_TOLERANCE,
            relative_tolerance=COARSE_RELATIVE_TOLERANCE,
        ) + _integral_to_turn(
            0.0,
            turning_u,
            impact_m,
            coordinate_time=False,
            absolute_tolerance=COARSE_ABSOLUTE_TOLERANCE,
            relative_tolerance=COARSE_RELATIVE_TOLERANCE,
        )
        lookback = _integral_to_turn(
            observer_u,
            turning_u,
            impact_m,
            coordinate_time=True,
            absolute_tolerance=TIME_ABSOLUTE_TOLERANCE_M,
            relative_tolerance=TIME_RELATIVE_TOLERANCE,
        ) + _integral_to_turn(
            boundary_u,
            turning_u,
            impact_m,
            coordinate_time=True,
            absolute_tolerance=TIME_ABSOLUTE_TOLERANCE_M,
            relative_tolerance=TIME_RELATIVE_TOLERANCE,
        )
        refinement = abs(fine_angle - coarse_angle)
        return RadialSolution(
            outcome=OUTCOME_ESCAPED,
            terminal_angle_rad=fine_angle,
            coordinate_lookback_time_m=lookback,
            frequency_shift_g=math.sqrt(
                lapse_squared(ESCAPE_RADIUS_M)
                / lapse_squared(OBSERVER_RADIUS_M)
            ),
            null_residual=_null_residual(
                impact_m,
                _regularized_path_samples(observer_u, turning_u)
                + _regularized_path_samples(boundary_u, turning_u),
            ),
            projection_error_px=(
                refinement + FLOAT32_DIRECTION_ERROR_RAD
            )
            * focal_pixels_per_radian,
            refinement_difference_rad=refinement,
        )

    fine_angle = _adaptive_simpson(
        _direct_integrand(impact_m, coordinate_time=False),
        observer_u,
        capture_u,
        FINE_ABSOLUTE_TOLERANCE,
        FINE_RELATIVE_TOLERANCE,
    )
    coarse_angle = _adaptive_simpson(
        _direct_integrand(impact_m, coordinate_time=False),
        observer_u,
        capture_u,
        COARSE_ABSOLUTE_TOLERANCE,
        COARSE_RELATIVE_TOLERANCE,
    )
    lookback = _adaptive_simpson(
        _direct_integrand(impact_m, coordinate_time=True),
        observer_u,
        capture_u,
        TIME_ABSOLUTE_TOLERANCE_M,
        TIME_RELATIVE_TOLERANCE,
    )
    refinement = abs(fine_angle - coarse_angle)
    return RadialSolution(
        outcome=OUTCOME_CAPTURED,
        terminal_angle_rad=fine_angle,
        coordinate_lookback_time_m=lookback,
        frequency_shift_g=0.0,
        null_residual=_null_residual(
            impact_m,
            _regularized_path_samples(observer_u, capture_u, count=257),
        ),
        projection_error_px=(
            refinement + FLOAT32_DIRECTION_ERROR_RAD
        )
        * focal_pixels_per_radian,
        refinement_difference_rad=refinement,
    )


def _percentile(values: list[float], fraction: float) -> float:
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


def _pack_record(
    screen_x: float,
    screen_y: float,
    solution: RadialSolution,
) -> bytes:
    if solution.outcome == OUTCOME_ESCAPED:
        radius = math.hypot(screen_x, screen_y)
        tangent_x = screen_x / radius
        tangent_y = -screen_y / radius
        sine = math.sin(solution.terminal_angle_rad)
        world_direction = [
            sine * tangent_x,
            sine * tangent_y,
            math.cos(solution.terminal_angle_rad),
        ]
        direction = [
            sum(row[index] * world_direction[index] for index in range(3))
            for row in WORLD_TO_ICRS
        ]
        inverse_norm = 1.0 / math.sqrt(sum(value * value for value in direction))
        direction = [value * inverse_norm for value in direction]
        return RECORD.pack(
            *direction,
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


def _build_manifest(
    width: int,
    height: int,
    chunks: list[dict[str, object]],
    escaped: int,
    captured: int,
    unresolved: int,
    radial_solutions: list[RadialSolution],
) -> dict[str, object]:
    document = fixture_manifest(b"")
    total = width * height
    escaped_fraction = escaped / total
    captured_fraction = captured / total
    unresolved_fraction = unresolved / total
    refinement_values = [
        solution.refinement_difference_rad for solution in radial_solutions
    ]
    projection_errors = [
        solution.projection_error_px for solution in radial_solutions
    ]
    null_residuals = [solution.null_residual for solution in radial_solutions]
    observer_f = lapse_squared(OBSERVER_RADIUS_M)
    observer_lapse = math.sqrt(observer_f)
    inverse_observer_f = 1.0 / observer_f
    radial_norm = math.sqrt(observer_f)

    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    world_to_icrs = [value for row in WORLD_TO_ICRS for value in row]
    icrs_to_world = [value for row in ICRS_TO_WORLD for value in row]
    camera_to_world = [
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, OBSERVER_RADIUS_M,
        0.0, 0.0, 0.0, 1.0,
    ]
    world_to_camera = [
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, OBSERVER_RADIUS_M,
        0.0, 0.0, 0.0, 1.0,
    ]
    metric_covariant = [
        -observer_f, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, inverse_observer_f,
    ]
    metric_contravariant = [
        -inverse_observer_f, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, observer_f,
    ]

    document.update(
        {
            "id": "schwarzschild-reference-v1",
            "datasetKind": "stationary-reference-transfer-map",
            "renderable": True,
            "scientificStatus": {
                "classification": (
                    "project-generated analytic Schwarzschild reference; not NR"
                ),
                "sourceIsNumericalRelativity": False,
                "derivedFromNearZoneSpacetime": False,
                "derivedWithSlowLightGeodesics": False,
                "description": (
                    "A fixed-camera vacuum transfer map obtained from the exact "
                    "stationary Schwarzschild null-orbit equation. It is a "
                    "single-black-hole reference and contains no accretion emission."
                ),
                "prohibitedClaim": (
                    "Do not describe this stationary analytic reference as "
                    "numerical relativity, a binary merger, slow-light NR output, "
                    "or a GRMHD accretion simulation."
                ),
            },
            "physicalSystem": {
                "kind": "stationary-black-hole",
                "vacuum": True,
                "componentIds": ["BH"],
                "parameterEpochProtocolM": 0.0,
                "massRatioQ": None,
                "dimensionlessSpins": [
                    {"componentId": "BH", "vector": [0.0, 0.0, 0.0]}
                ],
                "eccentricity": None,
                "referenceOrbitalPhaseRad": None,
                "remnant": None,
                "notApplicableReason": (
                    "Binary orbital and remnant parameters do not apply to the "
                    "eternal, non-spinning Schwarzschild solution."
                ),
                "description": (
                    "A vacuum Schwarzschild black hole with M=1 and zero spin."
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
                    "catalog": None,
                    "identifier": "analytic-schwarzschild-m1",
                    "version": "1",
                    "doi": None,
                    "evolutionCode": None,
                    "notApplicableReason": (
                        "The source is the analytic Schwarzschild metric, not a "
                        "catalog numerical-relativity evolution."
                    ),
                },
                "generator": {
                    "name": Path(__file__).name,
                    "version": "1.0.0",
                    "uri": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                    "command": (
                        "python3 scripts/generate_schwarzschild_transfer_map.py"
                    ),
                    "codeRevision": (
                        f"sha256:{sha256_bytes(Path(__file__).resolve().read_bytes())}"
                    ),
                    "deterministic": True,
                },
                "sourceArtifacts": [
                    source_artifact(
                        "generator-source",
                        Path(__file__).resolve(),
                    ),
                    source_artifact("schema", SCHEMA_PATH),
                    source_artifact("manifest-template-helper", HELPER_PATH),
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
                        "The Schwarzschild mass parameter; all radii and times "
                        "are expressed in this M."
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
                    "name": "stationary reference epoch",
                    "source": "project-generated",
                    "description": (
                        "An arbitrary t=0 slice of the time-translation-invariant "
                        "Schwarzschild spacetime."
                    ),
                },
                "waveformTimeMapping": {
                    "status": "not-applicable",
                    "sourceQuantity": None,
                    "mapping": None,
                    "notApplicableReason": (
                        "A stationary Schwarzschild black hole has no merger waveform."
                    ),
                },
            },
            "coordinates": {
                "metricSignature": "-+++",
                "nrChart": {
                    "status": "declared",
                    "gauge": (
                        "Schwarzschild areal-radius coordinates with a Cartesian "
                        "spatial embedding x=r sin(theta) cos(phi), "
                        "y=r sin(theta) sin(phi), z=r cos(theta)"
                    ),
                    "coordinates": (
                        "Schwarzschild t and Cartesian-embedded areal spatial "
                        "coordinates (x,y,z)"
                    ),
                    "timeSlicing": "constant Schwarzschild coordinate-time slices",
                },
                "worldFrame": {
                    "handedness": "right",
                    "axisOrder": ["x", "y", "z"],
                    "origin": "Schwarzschild symmetry centre",
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
                    "worldToIcrs": world_to_icrs,
                    "icrsToWorld": icrs_to_world,
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
                        "eventNr": [0.0, 0.0, 0.0, OBSERVER_RADIUS_M],
                        "metricCovariantNr": metric_covariant,
                        "metricContravariantNr": metric_contravariant,
                        "fourVelocityContravariantNr": [
                            1.0 / observer_lapse,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        "properTimeM": 0.0,
                        "tetradContravariantNr": [
                            [1.0 / observer_lapse, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, -radial_norm],
                        ],
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
                    "none; metric and null-orbit first integral are analytic"
                ),
                "temporalInterpolation": (
                    "none; Schwarzschild is stationary and the map has one sample"
                ),
                "integrator": {
                    "name": "adaptive-simpson-schwarzschild-orbit",
                    "method": (
                        "adaptive Simpson quadrature of the exact first integral "
                        "(du/dpsi)^2+u^2-2u^3=1/b^2, equivalent to "
                        "d2u/dpsi2=-u+3u^2; the square-root turning singularity "
                        "is removed by u=u_turn-s^2"
                    ),
                },
                "tolerances": {
                    "absolute": FINE_ABSOLUTE_TOLERANCE,
                    "relative": FINE_RELATIVE_TOLERANCE,
                    "nullConstraint": 5.0e-12,
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
                    "integrator-failure": (
                        "non-finite state or tolerance failure"
                    ),
                    "missing": "record was not generated",
                },
                "integrationPrecision": "float64",
                "outputPrecision": "float32",
            },
            "escapeBoundary": {
                "surface": {
                    "kind": "areal-radius-worldtube",
                    "centreWorldM": [0.0, 0.0, 0.0],
                    "radiusM": ESCAPE_RADIUS_M,
                },
                "referenceObserver": {
                    "kind": "eulerian-normal",
                    "definition": (
                        "Future-directed static Schwarzschild observer normal to "
                        f"t=constant at areal radius {ESCAPE_RADIUS_M:g}M."
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
                        "The exact stationary Schwarzschild null orbit is "
                        "continued analytically from the escape worldtube to "
                        "u=0; the asymptotic radial direction is stored."
                    ),
                },
            },
            "captureTargets": [
                {
                    "code": CAPTURE_BH,
                    "id": "BH",
                    "description": (
                        "A numerical stretched-horizon termination sphere at "
                        f"Schwarzschild areal radius {CAPTURE_RADIUS_M:g}M; "
                        "analytic b<3*sqrt(3)M decides capture."
                    ),
                    "surfaceKind": (
                        "Schwarzschild stretched-horizon areal-radius sphere"
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
                    "quantity": "NR Hamiltonian and momentum constraint norms",
                    "status": "not-applicable",
                    "method": None,
                    "value": None,
                },
                "geodesicNullResidual": {
                    "quantity": (
                        "maximum sampled analytic first-integral "
                        "null-consistency residual along each radial path"
                    ),
                    "status": "measured",
                    "method": (
                        "Reconstructed |g_mu_nu k^mu k^nu| at 257 radial "
                        "audit states per captured path and 129 regularized "
                        "states per leg of each escaped path. This audits the "
                        "analytic first-integral state, not an NR trajectory "
                        "constraint norm."
                    ),
                    "value": max(null_residuals),
                },
                "interpolationError": {
                    "quantity": (
                        "p95 stored-texel terminal direction solver and float32 "
                        "projection error in image pixels; runtime direction "
                        "interpolation is disabled"
                    ),
                    "status": "measured",
                    "method": (
                        "Fine/coarse terminal-angle discrepancy plus a "
                        "conservative 2e-7 rad float32 direction allowance, "
                        "converted with the declared pinhole focal length at "
                        "detector texel centres. The renderer selects the "
                        "nearest texel and performs no ray-direction blending."
                    ),
                    "value": _percentile(projection_errors, 0.95),
                },
                "unresolvedFraction": unresolved_fraction,
                "outcomeFractions": {
                    "escaped": escaped_fraction,
                    "captured": captured_fraction,
                    "unresolved": unresolved_fraction,
                    "outside-domain": 0.0,
                    "integrator-failure": 0.0,
                    "missing": 0.0,
                    "unusable": unresolved_fraction,
                },
                "fixtureAssertions": None,
            },
            "chunks": chunks,
        }
    )
    return document


def generate_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    tile_height: int = TILE_HEIGHT,
) -> GenerationReport:
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("width and height must be even integers >= 2")
    if tile_height < 1:
        raise ValueError("tile height must be positive")
    if not SCHEMA_PATH.is_file() or not HELPER_PATH.is_file():
        raise FileNotFoundError("the schema and fixture manifest helper are required")

    started = time.perf_counter()
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tangent_half_fov = math.tan(0.5 * VERTICAL_FOV_RAD)
    screen_scale = tangent_half_fov / height
    focal_pixels_per_radian = height / (2.0 * tangent_half_fov)
    radial_cache: dict[int, RadialSolution] = {}
    chunks: list[dict[str, object]] = []
    expected_chunk_names: set[str] = set()
    escaped = 0
    captured = 0
    unresolved = 0

    for tile_y in range(0, height, tile_height):
        current_height = min(tile_height, height - tile_y)
        name = f"t0000-y{tile_y:04d}-x0000.bin"
        expected_chunk_names.add(name)
        payload = bytearray()
        for y in range(tile_y, tile_y + current_height):
            grid_y = height - 2 * y - 1
            screen_y = grid_y * screen_scale
            for x in range(width):
                grid_x = 2 * x + 1 - width
                screen_x = grid_x * screen_scale
                radial_key = grid_x * grid_x + grid_y * grid_y
                solution = radial_cache.get(radial_key)
                if solution is None:
                    impact_m = impact_parameter_from_screen_radius(
                        math.hypot(screen_x, screen_y)
                    )
                    solution = solve_radial_ray(
                        impact_m,
                        focal_pixels_per_radian,
                    )
                    radial_cache[radial_key] = solution
                if solution.outcome == OUTCOME_ESCAPED:
                    escaped += 1
                elif solution.outcome == OUTCOME_CAPTURED:
                    captured += 1
                else:
                    unresolved += 1
                payload.extend(_pack_record(screen_x, screen_y, solution))

        chunk_path = chunk_dir / name
        chunk_bytes = bytes(payload)
        chunk_path.write_bytes(chunk_bytes)
        record_count = width * current_height
        if len(chunk_bytes) != record_count * RECORD_BYTES:
            raise AssertionError("generated chunk does not match the v1 record ABI")
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
                "byteLength": len(chunk_bytes),
                "sha256": sha256_bytes(chunk_bytes),
            }
        )
        print(
            f"  solved rows {tile_y:04d}-{tile_y + current_height - 1:04d}; "
            f"{len(radial_cache)} unique radial rays"
        )

    for stale_path in chunk_dir.glob("*.bin"):
        if stale_path.name not in expected_chunk_names:
            stale_path.unlink()

    solutions = list(radial_cache.values())
    document = _build_manifest(
        width,
        height,
        chunks,
        escaped,
        captured,
        unresolved,
        solutions,
    )
    manifest_bytes = (
        json.dumps(
            document,
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

    frequency_shift = math.sqrt(
        lapse_squared(ESCAPE_RADIUS_M) / lapse_squared(OBSERVER_RADIUS_M)
    )
    report = GenerationReport(
        width=width,
        height=height,
        chunks=len(chunks),
        records=width * height,
        escaped=escaped,
        captured=captured,
        unresolved=unresolved,
        unique_radial_rays=len(solutions),
        shadow_diameter_deg=math.degrees(
            2.0 * critical_shadow_radius_rad()
        ),
        frequency_shift_g=frequency_shift,
        max_refinement_difference_rad=max(
            solution.refinement_difference_rad for solution in solutions
        ),
        p95_projection_error_px=_percentile(
            [solution.projection_error_px for solution in solutions],
            0.95,
        ),
        p99_null_residual=_percentile(
            [solution.null_residual for solution in solutions],
            0.99,
        ),
        max_null_residual=max(
            solution.null_residual for solution in solutions
        ),
        elapsed_seconds=time.perf_counter() - started,
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="dataset output directory",
    )
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--tile-height", type=int, default=TILE_HEIGHT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = generate_dataset(
        arguments.output.resolve(),
        width=arguments.width,
        height=arguments.height,
        tile_height=arguments.tile_height,
    )
    print("Schwarzschild transfer map generated")
    print(
        f"  resolution = {report.width}x{report.height}, "
        f"records = {report.records}, chunks = {report.chunks}"
    )
    print(
        f"  outcomes = escaped:{report.escaped}, captured:{report.captured}, "
        f"unresolved:{report.unresolved}"
    )
    print(f"  unique radial rays = {report.unique_radial_rays}")
    print(f"  analytic shadow diameter = {report.shadow_diameter_deg:.6f} deg")
    print(f"  static-observer frequency shift g = {report.frequency_shift_g:.9f}")
    print(
        "  max fine/coarse direction difference = "
        f"{report.max_refinement_difference_rad:.3e} rad"
    )
    print(
        f"  p95 projection estimate = {report.p95_projection_error_px:.3e} px"
    )
    print(
        f"  p99/max null residual = {report.p99_null_residual:.3e} / "
        f"{report.max_null_residual:.3e}"
    )
    print(f"  elapsed = {report.elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
