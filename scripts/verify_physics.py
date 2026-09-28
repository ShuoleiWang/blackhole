#!/usr/bin/env python3
"""Numerical regressions for the Schwarzschild shader equations.

The shader integrates u'' = -u + 3u^2 (u = M/r) with Störmer-Verlet at a fixed
angular step and interpolates the horizon and escape crossings. This script
reads the step angle, step budgets and camera-distance range from the shader
and runtime sources, runs the same integrator in float64, and compares it with
exact quadratures of the orbit equation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITICAL = 3.0 * math.sqrt(3.0)


def _source_constant(path: str, pattern: str, label: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"cannot find {label} in {path}")
    return match.group(1)


def shader_constants() -> dict[str, float]:
    """Integrator constants as declared by the production sources."""
    budget = re.search(
        r"return clamp\(steps, (\d+), (\d+)\);",
        (ROOT / "src/main.js").read_text(encoding="utf-8"),
    )
    distance = re.search(
        r"cameraDistanceLimits \?\? \{ min: (\d+), max: (\d+) \}",
        (ROOT / "src/main.js").read_text(encoding="utf-8"),
    )
    if budget is None or distance is None:
        raise AssertionError("cannot find the step budget or camera range in src/main.js")
    return {
        "step": float(_source_constant("src/shaders.js", r"let stepAngle = ([0-9.]+);", "stepAngle")),
        "max_steps": int(_source_constant("src/shaders.js", r"const MAX_STEPS: i32 = (\d+);", "MAX_STEPS")),
        "critical_band": float(_source_constant(
            "src/shaders.js",
            r"if \(abs\(impact - PHOTON_IMPACT\) < ([0-9.]+)\)",
            "critical band",
        )),
        "budget_min": int(budget.group(1)),
        "budget_max": int(budget.group(2)),
        "camera_min": float(distance.group(1)),
        "camera_max": float(distance.group(2)),
    }


def trace_impact_parameter(
    impact: float,
    observer_radius: float = 10_000.0,
    step: float = 0.004,
    max_steps: int = 20_000,
) -> tuple[str, float, float]:
    """Return (outcome, swept azimuth at the crossing, max invariant error)."""
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
        previous_u = u
        velocity += 0.5 * step * acceleration(u)
        u += step * velocity
        velocity += 0.5 * step * acceleration(u)
        psi += step

        invariant = velocity * velocity + u * u - 2.0 * u * u * u
        max_error = max(max_error, abs(invariant - invariant_target))

        # Interpolate the crossing inside the step, as the shader does.
        if u >= 0.5:
            return "captured", psi - step * (u - 0.5) / (u - previous_u), max_error
        if u <= 0.0:
            return "escaped", psi - step * u / (u - previous_u), max_error

    return "unconverged", psi, max_error


def shader_outcome(impact: float, observer_radius: float, step: float, max_steps: int) -> str:
    """Classification the shader shows, including its budget-exhaustion rule.

    A ray that exhausts the step budget is drawn as captured when b < b_c and
    it is still moving inward (u' >= 0); otherwise the sky is sampled along its
    current direction.
    """
    outcome, psi, _ = trace_impact_parameter(impact, observer_radius, step, max_steps)
    if outcome != "unconverged":
        return outcome
    lapse = math.sqrt(1.0 - 2.0 / observer_radius)
    tangent = impact * lapse / observer_radius
    u = 1.0 / observer_radius
    velocity = lapse * math.sqrt(1.0 - tangent * tangent) / (observer_radius * tangent)
    for _ in range(max_steps):
        velocity += 0.5 * step * (-u + 3.0 * u * u)
        u += step * velocity
        velocity += 0.5 * step * (-u + 3.0 * u * u)
    return "captured" if impact < CRITICAL and velocity >= 0.0 else "sky-unresolved"


def _simpson(function, lower: float, upper: float, intervals: int = 4000) -> float:
    width = (upper - lower) / intervals
    total = function(lower) + function(upper)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * function(lower + index * width)
    return total * width / 3.0


def exact_escape_azimuth(impact: float, observer_radius: float) -> float:
    """Azimuth swept from the observer through periapsis to infinity.

    (du/dphi)^2 = F(u) = 1/b^2 - u^2 + 2u^3. With u = u_p - s^2 near the
    periapsis root u_p the integrand 2s/sqrt(F) is regular, so Simpson's rule
    converges quickly on both legs.
    """
    inverse_b2 = 1.0 / (impact * impact)

    def potential(u: float) -> float:
        return inverse_b2 - u * u + 2.0 * u * u * u

    # Smallest positive root of F: bisection between 0 and the photon sphere.
    low, high = 0.0, 1.0 / 3.0
    if potential(high) > 0.0:
        raise ValueError("impact parameter is captured; no periapsis")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if potential(middle) > 0.0:
            low = middle
        else:
            high = middle
    periapsis = 0.5 * (low + high)

    def leg(u_start: float) -> float:
        extent = math.sqrt(max(periapsis - u_start, 0.0))

        def integrand(s: float) -> float:
            value = potential(periapsis - s * s)
            if s == 0.0:
                derivative = -2.0 * periapsis + 6.0 * periapsis * periapsis
                return 2.0 / math.sqrt(-derivative)
            return 2.0 * s / math.sqrt(max(value, 1.0e-300))

        return _simpson(integrand, 0.0, extent)

    return leg(1.0 / observer_radius) + leg(0.0)


def local_angle(impact: float, observer_radius: float) -> float:
    return math.asin(impact * math.sqrt(1.0 - 2.0 / observer_radius) / observer_radius)


def shadow_edge(observer_radius: float, step: float, max_steps: int) -> float:
    """Bisect the capture/escape boundary in b with the shader integrator."""
    low, high = CRITICAL - 0.2, CRITICAL + 0.2
    for _ in range(40):
        middle = 0.5 * (low + high)
        outcome, _, _ = trace_impact_parameter(middle, observer_radius, step, max_steps)
        if outcome == "captured":
            low = middle
        elif outcome == "escaped":
            high = middle
        else:
            raise AssertionError(f"ray at b={middle:.9f} did not terminate")
    return 0.5 * (low + high)


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: got {actual:.10g}, expected {expected:.10g} ± {tolerance:.3g}")


def main() -> None:
    constants = shader_constants()
    step = constants["step"]

    # 1. Weak-field deflection against the exact integral (not just 4M/b).
    weak_b, far_radius = 50.0, 100_000.0
    outcome, psi, weak_error = trace_impact_parameter(weak_b, far_radius, 0.002, 40_000)
    if outcome != "escaped":
        raise AssertionError(f"weak-field ray did not escape: {outcome}")
    exact_psi = exact_escape_azimuth(weak_b, far_radius)
    deflection = psi - (math.pi - local_angle(weak_b, far_radius))
    exact_deflection = exact_psi - (math.pi - local_angle(weak_b, far_radius))
    assert_close(deflection, exact_deflection, 1.0e-5, "weak-field deflection")
    # The exact value itself follows the known strong-deflection series
    # 4x + (15 pi/4)x^2 + (128/3)x^3 + (3465 pi/64)x^4 + ..., x = M/b.
    x = 1.0 / weak_b
    series = 4.0 * x + 15.0 * math.pi / 4.0 * x**2 + 128.0 / 3.0 * x**3 + 3465.0 * math.pi / 64.0 * x**4
    assert_close(exact_deflection, series, 1.0e-5, "deflection series")

    # 2. The shader step at ordinary camera distances reproduces the exact
    # escape azimuth; its error bounds the sky-direction error.
    shader_direction_errors = []
    for observer_radius in (constants["camera_min"], 50.0, constants["camera_max"]):
        for offset in (0.45, 1.0, 3.0, 10.0):
            impact = CRITICAL + offset
            outcome, psi, _ = trace_impact_parameter(impact, observer_radius, step, constants["budget_min"])
            if outcome != "escaped":
                raise AssertionError(
                    f"R={observer_radius:g}, b=b_c+{offset:g} did not escape within "
                    f"{constants['budget_min']} steps: {outcome}"
                )
            shader_direction_errors.append(abs(psi - exact_escape_azimuth(impact, observer_radius)))
    worst_direction = max(shader_direction_errors)
    if worst_direction > 2.0e-3:
        raise AssertionError(f"shader-step escape azimuth error {worst_direction:.3e} rad exceeds 2e-3")

    # 3. Rays inside the near-critical band get MAX_STEPS and must terminate.
    band_errors = []
    for offset in (-0.025, 0.025, 0.1, constants["critical_band"] - 1.0e-3):
        outcome, _, error = trace_impact_parameter(CRITICAL + offset, 50.0, step, constants["max_steps"])
        expected = "captured" if offset < 0.0 else "escaped"
        if outcome != expected:
            raise AssertionError(f"near-critical b=b_c{offset:+g}: {outcome}, expected {expected}")
        band_errors.append(error)
    if max(band_errors) > 3.0e-5:
        raise AssertionError("shader-step invariant drift exceeded 3e-5")

    # 4. Shadow edge measured with the integrator, and its finite-distance
    # angular diameter from a static observer at R = 40 M.
    # Störmer-Verlet shifts the separatrix at O(h^2): halving h must cut the
    # edge error about fourfold.
    edge = shadow_edge(40.0, 0.004, 20_000)
    finer_edge = shadow_edge(40.0, 0.002, 40_000)
    assert_close(edge, CRITICAL, 1.0e-5, "shadow-edge impact parameter")
    if abs(finer_edge - CRITICAL) > abs(edge - CRITICAL) / 3.0:
        raise AssertionError(
            f"shadow edge is not converging at second order: {edge - CRITICAL:.3e} -> "
            f"{finer_edge - CRITICAL:.3e}"
        )
    edge = finer_edge
    shadow_diameter = math.degrees(2.0 * local_angle(edge, 40.0))
    # The shader's coarser step moves its separatrix outward by O(h^2); it must
    # stay far below a pixel (1e-3 M is 0.09' at R = 40 M).
    shader_separatrix = shadow_edge(40.0, step, 20_000)
    if not 0.0 <= shader_separatrix - CRITICAL < 1.0e-3:
        raise AssertionError(f"shader-step separatrix offset {shader_separatrix - CRITICAL:.3e} M")
    # With the production budget, rays inside the critical curve are drawn
    # black and rays outside it show sky beyond that offset.
    for offset in (1.0e-3, 1.0e-2, 0.1, 0.3):
        inside = shader_outcome(CRITICAL - offset, 40.0, step, constants["max_steps"])
        outside = shader_outcome(CRITICAL + offset, 40.0, step, constants["max_steps"])
        if inside != "captured" or outside == "captured":
            raise AssertionError(f"shader shadow edge wrong at |b-b_c|={offset:g}: {inside}/{outside}")

    if weak_error > 1.0e-5:
        raise AssertionError("Störmer-Verlet invariant drift exceeded the regression budget")

    print("Schwarzschild numerical checks passed")
    print(
        f"  shader constants: step={step} rad, budget {constants['budget_min']}-"
        f"{constants['budget_max']}, near-critical {constants['max_steps']} within "
        f"|b-b_c|<{constants['critical_band']}, camera {constants['camera_min']:g}-"
        f"{constants['camera_max']:g} M"
    )
    print(f"  weak deflection(b=50M, R=1e5M) = {deflection:.7f} rad (exact {exact_deflection:.7f})")
    print(f"  shader-step escape azimuth error <= {worst_direction:.2e} rad")
    print(f"  shadow edge b = {edge:.8f} M (3 sqrt 3 = {CRITICAL:.8f}); shader-step separatrix +{shader_separatrix - CRITICAL:.1e} M")
    print(f"  shadow diameter(R=40M) = {shadow_diameter:.4f} deg")


if __name__ == "__main__":
    main()
