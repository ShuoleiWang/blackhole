#!/usr/bin/env python3
"""Small numerical regressions for the Schwarzschild shader equations."""

from __future__ import annotations

import math


def trace_impact_parameter(
    impact: float,
    observer_radius: float = 10_000.0,
    step: float = 0.004,
    max_steps: int = 20_000,
) -> tuple[str, float, float]:
    lapse = math.sqrt(1.0 - 2.0 / observer_radius)
    tangent = impact * lapse / observer_radius
    if not 0.0 < tangent < 1.0:
        raise ValueError("impact parameter is outside the observer's local sky")

    radial = -math.sqrt(1.0 - tangent * tangent)
    u = 1.0 / observer_radius
    velocity = -lapse * radial / (observer_radius * tangent)
    psi = 0.0
    invariant_target = 1.0 / (impact * impact)
    max_error = 0.0

    def acceleration(value: float) -> float:
        return -value + 3.0 * value * value

    for _ in range(max_steps):
        velocity += 0.5 * step * acceleration(u)
        u += step * velocity
        velocity += 0.5 * step * acceleration(u)
        psi += step

        invariant = velocity * velocity + u * u - 2.0 * u * u * u
        max_error = max(max_error, abs(invariant - invariant_target))

        if u >= 0.5:
            return "captured", psi, max_error
        if u <= 0.0:
            return "escaped", psi, max_error

    return "unconverged", psi, max_error


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: got {actual:.8g}, expected {expected:.8g} ± {tolerance:.3g}")


def main() -> None:
    critical = 3.0 * math.sqrt(3.0)
    inside, _, inside_error = trace_impact_parameter(critical - 0.025)
    outside, _, outside_error = trace_impact_parameter(critical + 0.025)
    if inside != "captured" or outside != "escaped":
        raise AssertionError(f"critical shadow regression failed: {inside=}, {outside=}")

    shader_inside, _, shader_inside_error = trace_impact_parameter(
        critical - 0.025,
        observer_radius=50.0,
        step=0.03,
        max_steps=288,
    )
    shader_outside, _, shader_outside_error = trace_impact_parameter(
        critical + 0.025,
        observer_radius=50.0,
        step=0.03,
        max_steps=288,
    )
    if shader_inside != "captured" or shader_outside != "escaped":
        raise AssertionError(
            "shader-budget critical regression failed: "
            f"{shader_inside=}, {shader_outside=}"
        )
    if max(shader_inside_error, shader_outside_error) > 3.0e-5:
        raise AssertionError("shader-budget invariant drift exceeded 3e-5")

    weak_b = 50.0
    observer_radius = 100_000.0
    outcome, psi, weak_error = trace_impact_parameter(
        weak_b,
        observer_radius=observer_radius,
        step=0.002,
        max_steps=40_000,
    )
    if outcome != "escaped":
        raise AssertionError(f"weak-field ray did not escape: {outcome}")
    alpha = math.asin(weak_b * math.sqrt(1.0 - 2.0 / observer_radius) / observer_radius)
    deflection = psi - (math.pi - alpha)
    assert_close(deflection, 4.0 / weak_b, 0.008, "weak-field deflection")

    observer = 40.0
    analytic_shadow_diameter = 2.0 * math.asin(
        critical * math.sqrt(1.0 - 2.0 / observer) / observer
    )
    assert_close(math.degrees(analytic_shadow_diameter), 14.55, 0.05, "finite-distance shadow")

    if max(inside_error, outside_error, weak_error) > 1.0e-5:
        raise AssertionError("Störmer-Verlet invariant drift exceeded the regression budget")

    print("Schwarzschild numerical checks passed")
    print(f"  b_critical = {critical:.8f} M")
    print(f"  weak deflection(b=50M) = {deflection:.6f} rad (4M/b = {4 / weak_b:.6f})")
    print(f"  shadow diameter(R=40M) = {math.degrees(analytic_shadow_diameter):.4f} deg")
    print(f"  max invariant error = {max(inside_error, outside_error, weak_error):.3e}")
    print(
        "  shader budget (h=0.03, 288 steps) = "
        f"{shader_inside}/{shader_outside}, max error "
        f"{max(shader_inside_error, shader_outside_error):.3e}"
    )


if __name__ == "__main__":
    main()
