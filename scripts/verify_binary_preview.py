#!/usr/bin/env python3
"""Validate the compact, explicitly approximate binary-black-hole preview."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "scenes" / "binary-pn-equal-mass-v1.json"
ABS_TOL = 1.0e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def numeric_values(value: Any, path: str = "$") -> Iterable[tuple[str, float]]:
    """Yield all JSON numbers with paths, excluding booleans."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from numeric_values(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_values(item, f"{path}.{key}")


def angle_distance(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    weight = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def vector_length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    inverse_length = 1.0 / math.sqrt(
        max(
            vector[0] * vector[0]
            + vector[1] * vector[1]
            + vector[2] * vector[2],
            1.0e-18,
        )
    )
    return (
        vector[0] * inverse_length,
        vector[1] * inverse_length,
        vector[2] * inverse_length,
    )


def dot(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def binary_acceleration(
    position: tuple[float, float, float],
    direction: tuple[float, float, float],
    centre_a: tuple[float, float, float],
    centre_b: tuple[float, float, float],
    remnant_blend: float,
    mass_a: float,
    mass_b: float,
    remnant_mass: float,
) -> tuple[float, float, float]:
    offset_a = (
        position[0] - centre_a[0],
        position[1] - centre_a[1],
        position[2] - centre_a[2],
    )
    offset_b = (
        position[0] - centre_b[0],
        position[1] - centre_b[1],
        position[2] - centre_b[2],
    )
    radius_a = max(vector_length(offset_a), 0.05)
    radius_b = max(vector_length(offset_b), 0.05)
    radius_r = max(vector_length(position), 0.05)
    inverse_a3 = 1.0 / (radius_a * radius_a * radius_a)
    inverse_b3 = 1.0 / (radius_b * radius_b * radius_b)
    inverse_r3 = 1.0 / (radius_r * radius_r * radius_r)
    side_weight = 1.0 - remnant_blend
    gradient = (
        side_weight
        * (mass_a * offset_a[0] * inverse_a3 + mass_b * offset_b[0] * inverse_b3)
        + remnant_blend * remnant_mass * position[0] * inverse_r3,
        side_weight
        * (mass_a * offset_a[1] * inverse_a3 + mass_b * offset_b[1] * inverse_b3)
        + remnant_blend * remnant_mass * position[1] * inverse_r3,
        side_weight
        * (mass_a * offset_a[2] * inverse_a3 + mass_b * offset_b[2] * inverse_b3)
        + remnant_blend * remnant_mass * position[2] * inverse_r3,
    )
    projected = dot(direction, gradient)
    acceleration = (
        -2.0 * (gradient[0] - direction[0] * projected),
        -2.0 * (gradient[1] - direction[1] * projected),
        -2.0 * (gradient[2] - direction[2] * projected),
    )
    magnitude = vector_length(acceleration)
    limiter = min(1.0, 6.0 / max(magnitude, 1.0e-6))
    return (
        acceleration[0] * limiter,
        acceleration[1] * limiter,
        acceleration[2] * limiter,
    )


def trace_binary_preview_ray(
    initial_direction: tuple[float, float, float],
    camera_position: tuple[float, float, float],
    separation: float,
    orbital_phase: float,
    remnant_blend: float,
    mass_a: float,
    mass_b: float,
    remnant_mass: float,
    maximum_steps: int,
) -> tuple[str, int]:
    """Mirror the v1 shader ODE and return its first terminal classification."""
    total_binary_mass = max(mass_a + mass_b, 1.0e-6)
    axis = (math.cos(orbital_phase), 0.0, math.sin(orbital_phase))
    centre_a = (
        -separation * (mass_b / total_binary_mass) * axis[0],
        0.0,
        -separation * (mass_b / total_binary_mass) * axis[2],
    )
    centre_b = (
        separation * (mass_a / total_binary_mass) * axis[0],
        0.0,
        separation * (mass_a / total_binary_mass) * axis[2],
    )
    position = camera_position
    direction = normalize(initial_direction)
    camera_radius = vector_length(camera_position)
    previous_radius = camera_radius
    approached = False

    for step in range(maximum_steps):
        radius = vector_length(position)
        distance_a = vector_length(
            (
                position[0] - centre_a[0],
                position[1] - centre_a[1],
                position[2] - centre_a[2],
            )
        )
        distance_b = vector_length(
            (
                position[0] - centre_b[0],
                position[1] - centre_b[1],
                position[2] - centre_b[2],
            )
        )
        binary_distance = min(
            distance_a - 2.0 * mass_a,
            distance_b - 2.0 * mass_b,
        )
        remnant_distance = radius - 2.0 * remnant_mass
        topology_blend = smoothstep(0.12, 0.88, remnant_blend)
        capture_distance = (
            (1.0 - topology_blend) * binary_distance
            + topology_blend * remnant_distance
        )
        if capture_distance <= 0.0:
            return "captured", step

        approached = approached or radius < previous_radius
        if (
            approached
            and radius > 1.035 * camera_radius
            and dot(position, direction) > 0.0
        ):
            return "escaped", step
        previous_radius = radius

        nearest_mass_radius = 1.0e3
        if remnant_blend < 0.999:
            nearest_mass_radius = min(
                distance_a / mass_a,
                distance_b / mass_b,
            )
        if remnant_blend > 0.001:
            nearest_mass_radius = min(
                nearest_mass_radius,
                radius / remnant_mass,
            )
        step_length = clamp(
            0.045 + 0.020 * nearest_mass_radius,
            0.045,
            0.58,
        )
        step_length = min(
            step_length,
            clamp(0.34 * capture_distance, 0.018, 0.58),
        )

        acceleration_0 = binary_acceleration(
            position,
            direction,
            centre_a,
            centre_b,
            remnant_blend,
            mass_a,
            mass_b,
            remnant_mass,
        )
        midpoint_direction = normalize(
            (
                direction[0] + 0.5 * step_length * acceleration_0[0],
                direction[1] + 0.5 * step_length * acceleration_0[1],
                direction[2] + 0.5 * step_length * acceleration_0[2],
            )
        )
        midpoint = (
            position[0] + 0.5 * step_length * midpoint_direction[0],
            position[1] + 0.5 * step_length * midpoint_direction[1],
            position[2] + 0.5 * step_length * midpoint_direction[2],
        )
        acceleration_mid = binary_acceleration(
            midpoint,
            midpoint_direction,
            centre_a,
            centre_b,
            remnant_blend,
            mass_a,
            mass_b,
            remnant_mass,
        )
        direction = normalize(
            (
                direction[0] + step_length * acceleration_mid[0],
                direction[1] + step_length * acceleration_mid[1],
                direction[2] + step_length * acceleration_mid[2],
            )
        )
        position = (
            position[0] + step_length * midpoint_direction[0],
            position[1] + step_length * midpoint_direction[1],
            position[2] + step_length * midpoint_direction[2],
        )

    return "unresolved", maximum_steps


def verify_ray_budget(
    data: dict[str, Any],
) -> tuple[
    tuple[int, int],
    list[tuple[str, dict[int, dict[str, int]]]],
]:
    """Check representative timeline states at production and comparison budgets."""
    defaults = data["rendererDefaults"]
    samples = data["timeline"]["samples"]
    bodies = data["system"]["bodies"]
    mass_a = float(bodies[0]["massFraction"])
    mass_b = float(bodies[1]["massFraction"])
    remnant_mass = float(data["system"]["previewRemnant"]["massFraction"])
    production_budget = defaults.get("raySteps")
    require(
        isinstance(production_budget, int)
        and not isinstance(production_budget, bool)
        and production_budget > 64,
        "rendererDefaults.raySteps must be an integer greater than 64",
    )
    comparison_budget = production_budget - 64
    budgets = (comparison_budget, production_budget)

    def sample_at(time_m: float) -> dict[str, Any]:
        matching = [
            sample
            for sample in samples
            if math.isclose(float(sample["tM"]), time_m, abs_tol=ABS_TOL)
        ]
        require(len(matching) == 1, f"expected one timeline sample at t={time_m:g}M")
        return matching[0]

    representative_states = [
        ("initial", samples[0]),
        ("t=0 / a=6", sample_at(0.0)),
        ("transition / t=24", sample_at(24.0)),
        ("remnant / t=48", sample_at(48.0)),
    ]
    require(
        math.isclose(
            float(representative_states[1][1]["separationM"]),
            6.0,
            abs_tol=ABS_TOL,
        ),
        "t=0 representative state must have a=6M",
    )
    require(
        0.0 < float(representative_states[2][1]["mergerBlend"]) < 1.0,
        "t=24 representative state must be inside the merger transition",
    )
    require(
        math.isclose(
            float(representative_states[3][1]["mergerBlend"]),
            1.0,
            abs_tol=ABS_TOL,
        )
        and math.isclose(
            float(representative_states[3][1]["separationM"]),
            0.0,
            abs_tol=ABS_TOL,
        ),
        "t=48 representative state must be a completed remnant",
    )

    camera_radius = float(defaults["observerRadiusM"])
    inclination = math.radians(float(defaults["initialViewingInclinationDeg"]))
    latitude = math.pi / 2.0 - inclination
    azimuth = 0.58
    position_unit = (
        math.cos(latitude) * math.cos(azimuth),
        math.sin(latitude),
        math.cos(latitude) * math.sin(azimuth),
    )
    camera_position = (
        camera_radius * position_unit[0],
        camera_radius * position_unit[1],
        camera_radius * position_unit[2],
    )
    forward = (-position_unit[0], -position_unit[1], -position_unit[2])
    right = normalize((-math.sin(azimuth), 0.0, math.cos(azimuth)))
    up = normalize(cross(forward, right))
    tan_half_fov = math.tan(
        0.5 * math.radians(float(defaults["fieldOfViewDeg"]))
    )

    width = 90
    height = 45
    aspect = width / height
    rays = []
    for pixel_y in range(height):
        screen_y = 1.0 - 2.0 * (pixel_y + 0.5) / height
        for pixel_x in range(width):
            screen_x = (
                (2.0 * (pixel_x + 0.5) / width - 1.0)
                * aspect
            )
            rays.append(
                normalize(
                    (
                        forward[0]
                        + tan_half_fov * screen_x * right[0]
                        + tan_half_fov * screen_y * up[0],
                        forward[1]
                        + tan_half_fov * screen_x * right[1]
                        + tan_half_fov * screen_y * up[1],
                        forward[2]
                        + tan_half_fov * screen_x * right[2]
                        + tan_half_fov * screen_y * up[2],
                    )
                )
            )

    total_rays = width * height
    state_counts = []
    for label, sample in representative_states:
        counts = {
            budget: {"captured": 0, "escaped": 0, "unresolved": 0}
            for budget in budgets
        }
        for ray in rays:
            classification, terminal_step = trace_binary_preview_ray(
                ray,
                camera_position,
                float(sample["separationM"]),
                float(sample["orbitalPhaseRad"]),
                float(sample["mergerBlend"]),
                mass_a,
                mass_b,
                remnant_mass,
                production_budget,
            )
            for budget in budgets:
                budget_classification = (
                    classification
                    if terminal_step < budget
                    else "unresolved"
                )
                counts[budget][budget_classification] += 1

        for budget in budgets:
            require(
                sum(counts[budget].values()) == total_rays,
                f"{label}: {budget}-step classifications do not cover the grid",
            )
        require(
            counts[production_budget]["captured"] > 0
            and counts[production_budget]["escaped"] > 0,
            f"{label}: production grid does not exercise capture and escape",
        )
        require(
            counts[production_budget]["unresolved"] == 0,
            f"{label}: {production_budget}-step production budget leaves rays unresolved",
        )
        require(
            counts[comparison_budget]["unresolved"] * 1000 <= total_rays,
            f"{label}: {comparison_budget}-step comparison exceeds 0.1% unresolved",
        )
        require(
            counts[comparison_budget]["captured"]
            == counts[production_budget]["captured"],
            f"{label}: comparison budget changes the converged capture classification",
        )
        require(
            counts[comparison_budget]["escaped"]
            + counts[comparison_budget]["unresolved"]
            == counts[production_budget]["escaped"],
            f"{label}: comparison-budget unresolved rays do not converge to escape",
        )
        state_counts.append((label, counts))

    return budgets, state_counts


def verify_schema(data: dict[str, Any]) -> None:
    require(data.get("schema") == "blackhole.binary-scene/v1", "unexpected schema")
    require(data.get("id") == "binary-pn-equal-mass-v1", "unexpected scene id")

    for key in (
        "accuracy",
        "units",
        "referenceConfiguration",
        "system",
        "model",
        "rendererDefaults",
        "timeline",
    ):
        require(key in data, f"missing top-level key: {key}")

    accuracy = data["accuracy"]
    classification = accuracy.get("classification", "")
    require(
        classification == "PN / phenomenological weak-field preview",
        "accuracy classification must identify the PN/phenomenological weak-field boundary",
    )
    require(
        accuracy.get("fullNumericalRelativity") is False,
        "fullNumericalRelativity must be explicitly false",
    )
    require(
        accuracy.get("nearZoneSpacetimeIncluded") is False,
        "nearZoneSpacetimeIncluded must be explicitly false",
    )
    require(
        "not" in accuracy.get("description", "").lower(),
        "accuracy description must state what is not included",
    )
    require(
        "exact" in accuracy.get("prohibitedClaim", "").lower(),
        "accuracy metadata must explicitly prohibit an exact-simulation claim",
    )

    provenance = data["referenceConfiguration"].get("sampleProvenance", "").lower()
    require("not extracted" in provenance, "sample provenance must reject an NR-data claim")
    reference = data["referenceConfiguration"]
    require(
        reference.get("url") == "https://doi.org/10.5281/zenodo.3273935",
        "SXS reference must use the pinned dataset DOI",
    )
    require(
        reference.get("metadataMd5") == "099d4c93d9466fe4b7ecad6c94499cf3",
        "SXS Lev5 metadata digest drifted",
    )

    timeline = data["timeline"]
    interpolation = timeline.get("interpolation", {})
    require(
        interpolation.get("pnInspiral") == "analytic from declared equations",
        "PN playback must use analytic interpolation",
    )
    require(
        interpolation.get("phenomenologicalMerger") == "linear between samples",
        "v1 merger playback requires declared linear interpolation",
    )
    require(timeline.get("timeDirection") == "increasing", "time direction must be increasing")
    require(
        timeline.get("separationPolicy") == "non-increasing",
        "separation policy must be non-increasing",
    )
    require(len(timeline.get("samples", [])) >= 8, "timeline is too sparse")


def verify_equal_mass_symmetry(data: dict[str, Any]) -> None:
    system = data["system"]
    bodies = system["bodies"]
    require(len(bodies) == 2, "binary scene must contain exactly two bodies")

    first, second = bodies
    m1 = float(first["massFraction"])
    m2 = float(second["massFraction"])
    require(math.isclose(m1, m2, abs_tol=ABS_TOL), "masses are not equal")
    require(math.isclose(m1 + m2, 1.0, abs_tol=ABS_TOL), "mass fractions do not sum to one")
    require(math.isclose(system["massRatioQ"], 1.0, abs_tol=ABS_TOL), "q must be one")
    require(
        math.isclose(system["symmetricMassRatioEta"], 0.25, abs_tol=ABS_TOL),
        "equal-mass eta must be 1/4",
    )

    x1 = float(first["orbitPositionScale"])
    x2 = float(second["orbitPositionScale"])
    require(math.isclose(x1, -x2, abs_tol=ABS_TOL), "orbit offsets are not antisymmetric")
    require(math.isclose(m1 * x1 + m2 * x2, 0.0, abs_tol=ABS_TOL), "center of mass drifts")

    phase1 = float(first["orbitalPhaseOffsetRad"])
    phase2 = float(second["orbitalPhaseOffsetRad"])
    require(
        angle_distance(phase2, phase1) < ABS_TOL,
        "signed orbit scales already encode opposition; phase offsets must match",
    )
    require(
        first["dimensionlessSpin"] == second["dimensionlessSpin"] == [0.0, 0.0, 0.0],
        "reference binary must be non-spinning",
    )


def verify_timeline(data: dict[str, Any]) -> None:
    samples = data["timeline"]["samples"]
    times = [float(sample["tM"]) for sample in samples]
    separations = [float(sample["separationM"]) for sample in samples]
    phases = [float(sample["orbitalPhaseRad"]) for sample in samples]
    blends = [float(sample["mergerBlend"]) for sample in samples]

    require(
        all(current > previous for previous, current in zip(times, times[1:])),
        "sample times must increase strictly",
    )
    require(
        all(current <= previous for previous, current in zip(separations, separations[1:])),
        "orbital separation must never increase",
    )
    require(
        all(current >= previous for previous, current in zip(phases, phases[1:])),
        "orbital phase must never run backward",
    )
    require(
        all(0.0 <= blend <= 1.0 for blend in blends),
        "merger blend must stay in [0, 1]",
    )
    require(
        all(current >= previous for previous, current in zip(blends, blends[1:])),
        "merger blend must never decrease",
    )
    require(blends[0] == 0.0 and blends[-1] == 1.0, "transition endpoints are incomplete")
    require(any(0.0 < blend < 1.0 for blend in blends), "transition has no intermediate state")

    first_merged = next(
        (sample for sample in samples if math.isclose(sample["mergerBlend"], 1.0)),
        None,
    )
    require(first_merged is not None, "timeline never reaches the remnant")
    require(
        math.isclose(first_merged["separationM"], 0.0, abs_tol=ABS_TOL),
        "completed merger must have zero binary separation",
    )

    pn_samples = [sample for sample in samples if sample["regime"] == "pn-inspiral"]
    require(len(pn_samples) >= 4, "not enough PN inspiral samples")
    require(
        all(
            current["separationM"] < previous["separationM"]
            for previous, current in zip(pn_samples, pn_samples[1:])
        ),
        "PN inspiral separation must shrink strictly",
    )
    require(
        all(sample["mergerBlend"] == 0.0 for sample in pn_samples),
        "PN inspiral samples must not contain merger blending",
    )

    transition = data["model"]["mergerTransition"]
    first_transition = next(sample for sample in samples if sample["mergerBlend"] > 0.0)
    require(
        first_transition["tM"] >= transition["startTimeM"],
        "merger blend begins before its declared start",
    )
    require(
        math.isclose(first_merged["tM"], transition["completeTimeM"], abs_tol=ABS_TOL),
        "first remnant sample disagrees with completeTimeM",
    )


def verify_pn_samples(data: dict[str, Any]) -> None:
    """Check the bundled inspiral samples against their declared leading-order model."""
    eta = float(data["system"]["symmetricMassRatioEta"])
    inspiral = data["model"]["inspiral"]
    start = float(inspiral["startSeparationM"])
    match = float(inspiral["matchingSeparationM"])
    coefficient = (256.0 / 5.0) * eta

    for sample in data["timeline"]["samples"]:
        if sample["regime"] != "pn-inspiral":
            continue
        separation = float(sample["separationM"])
        expected_time = (match**4 - separation**4) / coefficient
        expected_phase = (start**2.5 - separation**2.5) / (32.0 * eta)
        require(
            math.isclose(sample["tM"], expected_time, abs_tol=ABS_TOL),
            f"PN time mismatch at a={separation:g}M",
        )
        require(
            math.isclose(sample["orbitalPhaseRad"], expected_phase, abs_tol=1.0e-6),
            f"PN phase mismatch at a={separation:g}M",
        )


def verify_finite_parameters(data: dict[str, Any]) -> None:
    values = list(numeric_values(data))
    require(values, "manifest contains no numeric parameters")
    for path, value in values:
        require(math.isfinite(value), f"non-finite parameter at {path}")

    defaults = data["rendererDefaults"]
    require(defaults["observerRadiusM"] > 0.0, "observer radius must be positive")
    require(
        0.0 < defaults["initialViewingInclinationDeg"] < 180.0,
        "initial viewing inclination is invalid",
    )
    require(0.0 < defaults["fieldOfViewDeg"] < 180.0, "field of view is invalid")
    require(
        isinstance(defaults["raySteps"], int)
        and not isinstance(defaults["raySteps"], bool)
        and defaults["raySteps"] > 64,
        "production ray budget must be an integer greater than 64",
    )
    require(defaults["cycleDurationSeconds"] > 0.0, "cycle duration must be positive")
    require(defaults["exposure"] > 0.0, "exposure must be positive")

    remnant = data["system"]["previewRemnant"]
    require(0.0 < remnant["massFraction"] < 1.0, "remnant mass fraction is invalid")
    spin_z = remnant["dimensionlessSpin"][2]
    require(abs(spin_z) < 1.0, "dimensionless remnant spin violates the Kerr bound")
    require(
        math.isclose(remnant["massFraction"], 0.951609417715, abs_tol=1.0e-12),
        "remnant mass disagrees with pinned SXS metadata",
    )
    require(
        math.isclose(spin_z, 0.686461676493, abs_tol=1.0e-12),
        "remnant spin disagrees with pinned SXS metadata",
    )
    rendering = remnant["rendering"]
    require(rendering.get("spinRendered") is False, "preview must not claim to render spin")
    require(
        rendering.get("frameDraggingRendered") is False,
        "preview must not claim to render frame dragging",
    )
    require(
        "not a computed horizon" in rendering.get("captureBoundaryModel", ""),
        "capture boundary must reject a computed-horizon claim",
    )

    propagation = data["model"]["lightPropagation"]
    require(
        propagation.get("positionsFrozenPerRay") is True,
        "fast-light contract must freeze positions per ray",
    )
    require(
        propagation.get("retardedTimeIncluded") is False,
        "preview must reject retarded-time propagation",
    )
    require(
        propagation.get("gravitomagneticTermsIncluded") is False,
        "preview must reject gravitomagnetic terms",
    )


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as stream:
        data = json.load(stream)

    verify_schema(data)
    verify_equal_mass_symmetry(data)
    verify_timeline(data)
    verify_pn_samples(data)
    verify_finite_parameters(data)
    ray_budgets, ray_counts = verify_ray_budget(data)

    samples = data["timeline"]["samples"]
    print("Binary preview manifest checks passed")
    print(f"  manifest = {MANIFEST.relative_to(ROOT)}")
    print(f"  samples = {len(samples)}")
    print(f"  time range = {samples[0]['tM']:.2f}M .. {samples[-1]['tM']:.2f}M")
    print(
        "  separation = "
        f"{samples[0]['separationM']:.1f}M -> {samples[-1]['separationM']:.1f}M"
    )
    print("  classification = PN / phenomenological weak-field preview (not full NR)")
    for label, counts in ray_counts:
        for budget in ray_budgets:
            print(
                f"  ray grid ({label}, 90x45, {budget} steps) = "
                f"{counts[budget]['captured']} captured / "
                f"{counts[budget]['escaped']} escaped / "
                f"{counts[budget]['unresolved']} unresolved"
            )


if __name__ == "__main__":
    main()
