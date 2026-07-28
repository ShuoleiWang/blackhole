#!/usr/bin/env python3
"""Independently verify the bundled stationary Schwarzschild transfer map.

This supplements ``verify_nr_contract.py``.  The contract validator proves
schema, provenance, hashes, frames, tiling, and record invariants; this script
checks the stationary physics that the generic protocol intentionally does not
claim to validate.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

try:
    from scripts.verify_nr_contract import validate_contract
except ModuleNotFoundError:  # Direct ``python3 scripts/...`` execution.
    from verify_nr_contract import validate_contract


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST: Final = (
    ROOT
    / "assets"
    / "transfer-maps"
    / "schwarzschild-reference-v1"
    / "manifest.json"
)
RECORD: Final = struct.Struct("<7fBBH")
CRITICAL_IMPACT_M: Final = 3.0 * math.sqrt(3.0)
INDEPENDENT_SIMPSON_INTERVALS: Final = 65_536
INDEPENDENT_DIRECTION_TOLERANCE_RAD: Final = 1.5e-6
FREQUENCY_FLOAT32_TOLERANCE: Final = 2.0e-7
LOOKBACK_FLOAT32_ULPS: Final = 2.0
LOOKBACK_ABSOLUTE_TOLERANCE_M: Final = 2.0e-4
EXPECTED_CAPTURE_RADIUS_M: Final = 2.02
AXIS_MAPPING_TOLERANCE: Final = 1.0e-12

OUTCOME_ESCAPED: Final = 0
OUTCOME_CAPTURED: Final = 1
OUTCOME_UNRESOLVED: Final = 2


@dataclass(frozen=True)
class PhysicsReport:
    width: int
    height: int
    records: int
    escaped: int
    captured: int
    unresolved: int
    shadow_diameter_deg: float
    expected_frequency_shift_g: float
    max_direction_norm_error: float
    max_independent_direction_error_rad: float
    direction_probe_count: int
    max_independent_lookback_error_m: float
    lookback_probe_count: int
    max_axis_mapping_error: float
    max_null_residual: float
    max_projection_error_px: float


@dataclass(frozen=True)
class _RayProbe:
    x: int
    y: int
    screen_x: float
    screen_y: float
    impact_m: float
    record: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
    ]


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
    screen_x = ((x + 0.5) / width * 2.0 - 1.0) * (width / height) * tangent
    screen_y = (1.0 - (y + 0.5) / height * 2.0) * tangent
    return screen_x, screen_y


def _impact_parameter(
    screen_x: float,
    screen_y: float,
    observer_radius_m: float,
) -> float:
    screen_radius = math.hypot(screen_x, screen_y)
    sin_alpha = screen_radius / math.sqrt(1.0 + screen_radius * screen_radius)
    return (
        observer_radius_m
        * sin_alpha
        / math.sqrt(1.0 - 2.0 / observer_radius_m)
    )


def _expected_outcome_for_impact(impact_m: float) -> int:
    """Classify the exact Schwarzschild separatrix without hiding equality."""
    if impact_m < CRITICAL_IMPACT_M:
        return OUTCOME_CAPTURED
    if impact_m > CRITICAL_IMPACT_M:
        return OUTCOME_ESCAPED
    return OUTCOME_UNRESOLVED


def _turning_root(impact_m: float, observer_radius_m: float) -> float:
    def polynomial(u: float) -> float:
        return 1.0 / (impact_m * impact_m) - u * u + 2.0 * u * u * u

    lower = 1.0 / observer_radius_m
    upper = 1.0 / 3.0
    if polynomial(lower) <= 0.0 or polynomial(upper) >= 0.0:
        _fail(f"independent root bracket failed for b={impact_m:.9g}M")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if polynomial(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _composite_simpson(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int = INDEPENDENT_SIMPSON_INTERVALS,
) -> float:
    """Fixed-grid Simpson rule used only by the independent verifier."""
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson interval count must be a positive even integer")
    if not upper > lower:
        return 0.0
    step = (upper - lower) / intervals
    odd = math.fsum(
        function(lower + index * step)
        for index in range(1, intervals, 2)
    )
    even = math.fsum(
        function(lower + index * step)
        for index in range(2, intervals, 2)
    )
    return (
        step
        * (
            function(lower)
            + 4.0 * odd
            + 2.0 * even
            + function(upper)
        )
        / 3.0
    )


def _composite_simpson_to_turn(
    lower_u: float,
    turning_u: float,
    intervals: int = INDEPENDENT_SIMPSON_INTERVALS,
) -> float:
    """Fixed-grid Simpson rule, independent of the generator's adaptive rule."""
    upper_s = math.sqrt(turning_u - lower_u)
    derivative_limit = 2.0 * turning_u * (1.0 - 3.0 * turning_u)
    endpoint = 2.0 / math.sqrt(derivative_limit)

    def transformed(s: float) -> float:
        if s == 0.0:
            return endpoint
        u = turning_u - s * s
        polynomial_from_root = (u - turning_u) * (
            2.0 * (u * u + u * turning_u + turning_u * turning_u)
            - (u + turning_u)
        )
        return 2.0 * s / math.sqrt(polynomial_from_root)

    return _composite_simpson(
        transformed,
        0.0,
        upper_s,
        intervals,
    )


def _independent_escape_angle(
    impact_m: float,
    observer_radius_m: float,
) -> float:
    turning_u = _turning_root(impact_m, observer_radius_m)
    return _composite_simpson_to_turn(
        1.0 / observer_radius_m,
        turning_u,
    ) + _composite_simpson_to_turn(0.0, turning_u)


def _composite_simpson_time_to_turn(
    lower_u: float,
    turning_u: float,
    impact_m: float,
    intervals: int = INDEPENDENT_SIMPSON_INTERVALS,
) -> float:
    """Independently integrate positive Schwarzschild coordinate lookback."""
    upper_s = math.sqrt(turning_u - lower_u)
    derivative_limit = 2.0 * turning_u * (1.0 - 3.0 * turning_u)
    endpoint = (
        2.0
        / math.sqrt(derivative_limit)
        / (
            impact_m
            * turning_u
            * turning_u
            * (1.0 - 2.0 * turning_u)
        )
    )

    def transformed(s: float) -> float:
        if s == 0.0:
            return endpoint
        u = turning_u - s * s
        polynomial_from_root = (u - turning_u) * (
            2.0 * (u * u + u * turning_u + turning_u * turning_u)
            - (u + turning_u)
        )
        return (
            2.0
            * s
            / math.sqrt(polynomial_from_root)
            / (impact_m * u * u * (1.0 - 2.0 * u))
        )

    return _composite_simpson(
        transformed,
        0.0,
        upper_s,
        intervals,
    )


def _independent_coordinate_lookback(
    impact_m: float,
    outcome: int,
    observer_radius_m: float,
    escape_radius_m: float,
    capture_radius_m: float = EXPECTED_CAPTURE_RADIUS_M,
) -> float:
    """Compute the declared terminal lookback without generator code reuse."""
    observer_u = 1.0 / observer_radius_m
    if outcome == OUTCOME_ESCAPED:
        turning_u = _turning_root(impact_m, observer_radius_m)
        return _composite_simpson_time_to_turn(
            observer_u,
            turning_u,
            impact_m,
        ) + _composite_simpson_time_to_turn(
            1.0 / escape_radius_m,
            turning_u,
            impact_m,
        )
    if outcome == OUTCOME_CAPTURED:
        capture_u = 1.0 / capture_radius_m

        def integrand(u: float) -> float:
            polynomial = (
                1.0 / (impact_m * impact_m)
                - u * u
                + 2.0 * u * u * u
            )
            if polynomial <= 0.0:
                _fail(
                    "independent captured-lookback integrand reached a "
                    f"non-positive radicand {polynomial:.3e}"
                )
            return (
                1.0
                / math.sqrt(polynomial)
                / (impact_m * u * u * (1.0 - 2.0 * u))
            )

        return _composite_simpson(integrand, observer_u, capture_u)
    _fail(f"outcome {outcome} has no terminal coordinate lookback oracle")


def _float32_ulp(value: float) -> float:
    """Return a conservative adjacent-value spacing for a stored float32."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits >= 0x7F80_0000:
        _fail(f"cannot compute a float32 ULP for non-finite value {value!r}")
    upper = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    lower = (
        struct.unpack("<f", struct.pack("<I", bits - 1))[0]
        if bits > 0
        else value
    )
    return max(abs(upper - value), abs(value - lower))


def _flat_matrix3(values: object, path: str) -> list[list[float]]:
    if not isinstance(values, list) or len(values) != 9:
        _fail(f"{path} must contain nine row-major values")
    matrix: list[list[float]] = []
    for row in range(3):
        current = values[row * 3 : row * 3 + 3]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in current
        ):
            _fail(f"{path} contains a non-finite or non-numeric value")
        matrix.append([float(value) for value in current])
    return matrix


def _mat3_vector(
    matrix: list[list[float]],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        math.fsum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _determinant3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _max_vector_error(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return max(abs(left - right) for left, right in zip(first, second))


def _angular_separation(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    cross_norm = math.sqrt(math.fsum(component * component for component in cross))
    dot = math.fsum(left * right for left, right in zip(first, second))
    return math.atan2(cross_norm, max(-1.0, min(1.0, dot)))


def _validate_icrs_axis_mapping(
    manifest: dict[str, object],
) -> tuple[list[list[float]], float]:
    """Lock the stationary reference to a proper, explicitly oriented ICRS frame."""
    coordinates = manifest["coordinates"]  # type: ignore[index]
    sky = coordinates["sky"]  # type: ignore[index]
    world_to_icrs = _flat_matrix3(
        sky["worldToIcrs"],  # type: ignore[index]
        "$.coordinates.sky.worldToIcrs",
    )
    icrs_to_world = _flat_matrix3(
        sky["icrsToWorld"],  # type: ignore[index]
        "$.coordinates.sky.icrsToWorld",
    )
    determinant_error = max(
        abs(_determinant3(world_to_icrs) - 1.0),
        abs(_determinant3(icrs_to_world) - 1.0),
    )
    if determinant_error > AXIS_MAPPING_TOLERANCE:
        _fail(
            "world/ICRS transforms are not proper rotations; determinant error "
            f"{determinant_error:.3e}"
        )

    inverse_error = max(
        abs(
            math.fsum(
                world_to_icrs[row][inner] * icrs_to_world[inner][column]
                for inner in range(3)
            )
            - (1.0 if row == column else 0.0)
        )
        for row in range(3)
        for column in range(3)
    )
    if inverse_error > AXIS_MAPPING_TOLERANCE:
        _fail(
            "world/ICRS transforms are not mutual inverses; error "
            f"{inverse_error:.3e}"
        )

    camera_to_world = manifest["camera"]["cameraToWorld"]  # type: ignore[index]
    if not isinstance(camera_to_world, list) or len(camera_to_world) != 16:
        _fail("$.camera.cameraToWorld must contain sixteen row-major values")
    camera_axes = {
        "right": (
            float(camera_to_world[0]),
            float(camera_to_world[4]),
            float(camera_to_world[8]),
        ),
        "up": (
            float(camera_to_world[1]),
            float(camera_to_world[5]),
            float(camera_to_world[9]),
        ),
        "forward": (
            float(camera_to_world[2]),
            float(camera_to_world[6]),
            float(camera_to_world[10]),
        ),
    }
    expected_icrs = {
        "right": (1.0, 0.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "forward": (0.0, -1.0, 0.0),
    }
    axis_error = 0.0
    for name, world_axis in camera_axes.items():
        mapped = _mat3_vector(world_to_icrs, world_axis)
        expected = expected_icrs[name]
        for sign in (-1.0, 1.0):
            axis_error = max(
                axis_error,
                _max_vector_error(
                    tuple(sign * component for component in mapped),
                    tuple(sign * component for component in expected),
                ),
            )
    if axis_error > AXIS_MAPPING_TOLERANCE:
        _fail(
            "stationary camera/ICRS axis mapping must be "
            "right->+X, up->+Z, forward->-Y; max error "
            f"{axis_error:.3e}"
        )
    return icrs_to_world, max(determinant_error, inverse_error, axis_error)


def _load_records(
    manifest: dict[str, object],
    manifest_path: Path,
) -> list[tuple[float, float, float, float, float, float, float, int, int, int]]:
    width = int(manifest["projection"]["widthPixels"])  # type: ignore[index]
    height = int(manifest["projection"]["heightPixels"])  # type: ignore[index]
    records: list[
        tuple[float, float, float, float, float, float, float, int, int, int]
        | None
    ] = [None] * (width * height)
    for chunk in manifest["chunks"]:  # type: ignore[index]
        tile = chunk["tile"]
        tile_x = int(tile["x"])
        tile_y = int(tile["y"])
        tile_width = int(tile["width"])
        tile_height = int(tile["height"])
        payload = (manifest_path.parent / chunk["uri"]).read_bytes()
        offset = 0
        for local_y in range(tile_height):
            for local_x in range(tile_width):
                record = RECORD.unpack_from(payload, offset)
                offset += RECORD.size
                index = (tile_y + local_y) * width + tile_x + local_x
                if records[index] is not None:
                    _fail(f"duplicate decoded pixel at flat index {index}")
                records[index] = record
    if any(record is None for record in records):
        _fail("chunk tiles did not populate every image pixel")
    return [record for record in records if record is not None]


def validate_stationary_physics(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> PhysicsReport:
    manifest_path = manifest_path.resolve()
    contract = validate_contract(manifest_path)
    if contract["id"] != "schwarzschild-reference-v1":
        _fail(f"unexpected dataset id {contract['id']!r}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["datasetKind"] != "stationary-reference-transfer-map":
        _fail("dataset is not a stationary reference transfer map")
    if manifest["scientificStatus"]["sourceIsNumericalRelativity"] is not False:
        _fail("stationary reference incorrectly claims an NR source")
    if manifest["scientificStatus"]["derivedWithSlowLightGeodesics"] is not False:
        _fail("stationary reference incorrectly claims NR slow-light derivation")

    width = manifest["projection"]["widthPixels"]
    height = manifest["projection"]["heightPixels"]
    vertical_fov_rad = manifest["projection"]["verticalFieldOfViewRad"]
    observer_sample = manifest["observer"]["samples"][0]
    observer_position = observer_sample["eventNr"][1:]
    observer_radius_m = math.sqrt(
        math.fsum(component * component for component in observer_position)
    )
    escape_radius_m = manifest["escapeBoundary"]["surface"]["radiusM"]
    capture_description = manifest["captureTargets"][0]["description"]
    if "analytic b<3*sqrt(3)M decides capture" not in capture_description:
        _fail("capture surface does not declare the analytic shadow gate")
    if f"radius {EXPECTED_CAPTURE_RADIUS_M:g}M" not in capture_description:
        _fail(
            "capture surface does not declare the verifier's "
            f"{EXPECTED_CAPTURE_RADIUS_M:g}M terminal radius"
        )
    icrs_to_world, max_axis_mapping_error = _validate_icrs_axis_mapping(manifest)

    expected_frequency_shift = math.sqrt(
        (1.0 - 2.0 / escape_radius_m)
        / (1.0 - 2.0 / observer_radius_m)
    )
    shadow_radius = math.asin(
        CRITICAL_IMPACT_M
        * math.sqrt(1.0 - 2.0 / observer_radius_m)
        / observer_radius_m
    )
    records = _load_records(manifest, manifest_path)

    escaped = 0
    captured = 0
    unresolved = 0
    max_norm_error = 0.0
    max_null_residual = 0.0
    max_projection_error = 0.0
    radial_invariants: dict[int, tuple[int, float, float, float]] = {}
    lookback_candidates: dict[str, tuple[float, _RayProbe]] = {}
    octant_candidates: dict[int, tuple[float, _RayProbe]] = {}
    corner_probes: dict[tuple[int, int], _RayProbe] = {}

    def keep_best_lookback(
        name: str,
        score: float,
        probe: _RayProbe,
    ) -> None:
        previous = lookback_candidates.get(name)
        if previous is None or score < previous[0]:
            lookback_candidates[name] = (score, probe)

    def keep_best_octant(
        octant: int,
        score: float,
        probe: _RayProbe,
    ) -> None:
        previous = octant_candidates.get(octant)
        if previous is None or score < previous[0]:
            octant_candidates[octant] = (score, probe)

    for y in range(height):
        grid_y = height - 2 * y - 1
        for x in range(width):
            grid_x = 2 * x + 1 - width
            radial_key = grid_x * grid_x + grid_y * grid_y
            record = records[y * width + x]
            (
                direction_x,
                direction_y,
                direction_z,
                frequency_shift,
                lookback,
                null_residual,
                projection_error,
                outcome,
                capture_target,
                validity_mask,
            ) = record
            screen_x, screen_y = _screen_coordinates(
                x,
                y,
                width,
                height,
                vertical_fov_rad,
            )
            impact_m = _impact_parameter(
                screen_x,
                screen_y,
                observer_radius_m,
            )
            expected_outcome = _expected_outcome_for_impact(impact_m)
            if outcome != expected_outcome:
                _fail(
                    f"pixel ({x},{y}) has outcome {outcome}, expected "
                    f"{expected_outcome} from analytic b=3*sqrt(3)M"
                )
            probe = _RayProbe(
                x=x,
                y=y,
                screen_x=screen_x,
                screen_y=screen_y,
                impact_m=impact_m,
                record=record,
            )
            if outcome == OUTCOME_ESCAPED:
                escaped += 1
                if capture_target != 255 or validity_mask != 0x1F:
                    _fail(f"escaped pixel ({x},{y}) has invalid sentinels")
                norm = math.sqrt(
                    direction_x * direction_x
                    + direction_y * direction_y
                    + direction_z * direction_z
                )
                max_norm_error = max(max_norm_error, abs(norm - 1.0))
                if abs(frequency_shift - expected_frequency_shift) > (
                    FREQUENCY_FLOAT32_TOLERANCE
                ):
                    _fail(
                        f"pixel ({x},{y}) frequency shift {frequency_shift:.9g} "
                        f"does not match {expected_frequency_shift:.9g}"
                    )
                distance_above_critical = impact_m - CRITICAL_IMPACT_M
                keep_best_lookback(
                    "nearest-escaped",
                    distance_above_critical,
                    probe,
                )
                keep_best_lookback(
                    "ordinary-escaped",
                    abs(impact_m - 1.5 * CRITICAL_IMPACT_M),
                    probe,
                )
                if x in (0, width - 1) or y in (0, height - 1):
                    keep_best_lookback("edge-escaped", -impact_m, probe)
                polar_angle = math.atan2(screen_y, screen_x)
                octant = int(
                    math.floor((polar_angle + math.pi / 8.0) / (math.pi / 4.0))
                ) % 8
                keep_best_octant(octant, distance_above_critical, probe)
                if (x, y) in {
                    (0, 0),
                    (width - 1, 0),
                    (0, height - 1),
                    (width - 1, height - 1),
                }:
                    corner_probes[(x, y)] = probe
            elif outcome == OUTCOME_CAPTURED:
                captured += 1
                if capture_target != 0 or validity_mask != 0x1C:
                    _fail(f"captured pixel ({x},{y}) has invalid sentinels")
                if any(
                    value != 0.0
                    for value in (
                        direction_x,
                        direction_y,
                        direction_z,
                        frequency_shift,
                    )
                ):
                    _fail(f"captured pixel ({x},{y}) has nonzero invalid fields")
                keep_best_lookback(
                    "nearest-captured",
                    CRITICAL_IMPACT_M - impact_m,
                    probe,
                )
            elif outcome == OUTCOME_UNRESOLVED:
                unresolved += 1
                if capture_target != 255 or validity_mask != 0x18:
                    _fail(f"critical pixel ({x},{y}) has invalid unresolved sentinels")
                if any(
                    value != 0.0
                    for value in (
                        direction_x,
                        direction_y,
                        direction_z,
                        frequency_shift,
                        lookback,
                    )
                ):
                    _fail(f"critical unresolved pixel ({x},{y}) has nonzero invalid fields")
            else:
                _fail(f"pixel ({x},{y}) has unsupported analytic outcome {outcome}")
            if outcome in (OUTCOME_ESCAPED, OUTCOME_CAPTURED) and not lookback > 0.0:
                _fail(f"pixel ({x},{y}) has non-positive coordinate lookback")
            max_null_residual = max(max_null_residual, null_residual)
            max_projection_error = max(max_projection_error, projection_error)
            radial_signature = (
                outcome,
                lookback,
                null_residual,
                projection_error,
            )
            previous = radial_invariants.setdefault(radial_key, radial_signature)
            if previous != radial_signature:
                _fail(
                    f"spherical-symmetry scalar mismatch at radial key {radial_key}"
                )

    declared_counts = manifest["accuracy"]["outcomeFractions"]
    if escaped / len(records) != declared_counts["escaped"]:
        _fail("declared escaped fraction does not match analytic classification")
    if captured / len(records) != declared_counts["captured"]:
        _fail("declared captured fraction does not match analytic classification")
    if unresolved / len(records) != declared_counts["unresolved"]:
        _fail("declared unresolved fraction does not match analytic separatrix")
    declared_null = manifest["accuracy"]["geodesicNullResidual"]["value"]
    if not math.isclose(
        max_null_residual,
        declared_null,
        rel_tol=2.0e-6,
        abs_tol=1.0e-18,
    ):
        _fail(
            "decoded max null residual does not match the declared measured value"
        )
    if max_null_residual >= manifest["rayIntegration"]["tolerances"]["nullConstraint"]:
        _fail("stored null residual exceeds the declared integration tolerance")

    required_lookback_probes = {
        "ordinary-escaped",
        "nearest-escaped",
        "nearest-captured",
        "edge-escaped",
    }
    missing_lookback_probes = required_lookback_probes - lookback_candidates.keys()
    if missing_lookback_probes:
        _fail(
            "unable to construct independent coordinate-lookback probes: "
            + ", ".join(sorted(missing_lookback_probes))
        )
    max_lookback_error = 0.0
    for name in sorted(required_lookback_probes):
        probe = lookback_candidates[name][1]
        expected_lookback = _independent_coordinate_lookback(
            probe.impact_m,
            probe.record[7],
            observer_radius_m,
            escape_radius_m,
        )
        stored_lookback = probe.record[4]
        tolerance = max(
            LOOKBACK_ABSOLUTE_TOLERANCE_M,
            LOOKBACK_FLOAT32_ULPS * _float32_ulp(stored_lookback),
        )
        error = abs(stored_lookback - expected_lookback)
        max_lookback_error = max(max_lookback_error, error)
        if error > tolerance:
            _fail(
                f"{name} pixel ({probe.x},{probe.y}) coordinate lookback "
                f"differs from independent fixed-grid quadrature by {error:.3e}M "
                f"(allowed {tolerance:.3e}M)"
            )

    if set(octant_candidates) != set(range(8)):
        _fail("independent direction probes do not cover all eight image-plane octants")
    direction_probes: dict[tuple[int, int], _RayProbe] = {
        (probe.x, probe.y): probe
        for _, probe in octant_candidates.values()
    }
    direction_probes.update(corner_probes)
    if len(direction_probes) < 8:
        _fail("independent direction verification requires at least eight unique probes")

    max_direction_error = 0.0
    for probe in direction_probes.values():
        expected_angle = _independent_escape_angle(
            probe.impact_m,
            observer_radius_m,
        )
        screen_radius = math.hypot(probe.screen_x, probe.screen_y)
        tangent_x = probe.screen_x / screen_radius
        tangent_y = -probe.screen_y / screen_radius
        expected_world = (
            math.sin(expected_angle) * tangent_x,
            math.sin(expected_angle) * tangent_y,
            math.cos(expected_angle),
        )
        stored_world = _mat3_vector(
            icrs_to_world,
            (probe.record[0], probe.record[1], probe.record[2]),
        )
        stored_norm = math.sqrt(
            math.fsum(component * component for component in stored_world)
        )
        stored_world = tuple(
            component / stored_norm for component in stored_world
        )
        error = _angular_separation(expected_world, stored_world)
        max_direction_error = max(max_direction_error, error)
        if error > INDEPENDENT_DIRECTION_TOLERANCE_RAD:
            _fail(
                f"pixel ({probe.x},{probe.y}) direction differs from independent "
                f"fixed-grid quadrature by {error:.3e} rad"
            )

    return PhysicsReport(
        width=width,
        height=height,
        records=len(records),
        escaped=escaped,
        captured=captured,
        unresolved=unresolved,
        shadow_diameter_deg=math.degrees(2.0 * shadow_radius),
        expected_frequency_shift_g=expected_frequency_shift,
        max_direction_norm_error=max_norm_error,
        max_independent_direction_error_rad=max_direction_error,
        direction_probe_count=len(direction_probes),
        max_independent_lookback_error_m=max_lookback_error,
        lookback_probe_count=len(required_lookback_probes),
        max_axis_mapping_error=max_axis_mapping_error,
        max_null_residual=max_null_residual,
        max_projection_error_px=max_projection_error,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    return parser.parse_args()


def main() -> None:
    report = validate_stationary_physics(_parse_args().manifest)
    print("Stationary Schwarzschild transfer-map physics checks passed")
    print(
        f"  resolution = {report.width}x{report.height}, "
        f"records = {report.records}"
    )
    print(
        f"  analytic outcomes = escaped:{report.escaped}, "
        f"captured:{report.captured}, unresolved:{report.unresolved}"
    )
    print(
        f"  finite-distance shadow diameter = "
        f"{report.shadow_diameter_deg:.6f} deg"
    )
    print(
        f"  boundary frequency shift g = "
        f"{report.expected_frequency_shift_g:.9f}"
    )
    print(
        f"  max float32 direction norm error = "
        f"{report.max_direction_norm_error:.3e}"
    )
    print(
        f"  max independent direction error = "
        f"{report.max_independent_direction_error_rad:.3e} rad "
        f"({report.direction_probe_count} probes)"
    )
    print(
        f"  max independent coordinate-lookback error = "
        f"{report.max_independent_lookback_error_m:.3e} M "
        f"({report.lookback_probe_count} probes)"
    )
    print(
        f"  max declared ICRS axis-mapping error = "
        f"{report.max_axis_mapping_error:.3e}"
    )
    print(f"  max null residual = {report.max_null_residual:.3e}")
    print(
        f"  max per-ray projection estimate = "
        f"{report.max_projection_error_px:.3e} px"
    )


if __name__ == "__main__":
    main()
