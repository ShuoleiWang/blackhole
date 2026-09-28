#!/usr/bin/env python3
"""Independent acceptance checks for the strong-field CPU spacetime oracle.

The JavaScript oracle is compared with an independent Python implementation of
the Cartesian Kerr-Schild metric at off-equatorial points and arbitrary spin
directions. The null Hamiltonian is checked against the closed-form inverse
metric eta - 2H l l rather than against quantities derived from the oracle's
own 3+1 split, and the fail-closed paths are exercised directly.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "strong-field-spacetime.js"

# (mass, dimensionless spin vector, position) — off-equatorial, arbitrary spin
# axes, prograde and retrograde, all outside the regularized ring region.
CASES = [
    (1.0, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
    (1.3, [0.0, 0.0, 0.0], [3.1, -2.4, 5.7]),
    (1.0, [0.0, 0.0, 0.7], [8.0, 0.0, 0.0]),
    (1.0, [0.0, 0.0, -0.7], [8.0, 0.0, 0.0]),
    (1.0, [0.0, 0.0, 0.95], [1.2, 2.3, 1.7]),
    (0.8, [0.3, -0.5, 0.4], [-2.2, 1.9, 3.3]),
    (1.0, [0.0, 0.686, 0.0], [0.9, 0.4, -2.6]),
]
MOMENTA = [[0.7, -0.2, 1.1], [-1.0, 0.3, 0.05], [0.2, 0.9, -0.6]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    require(
        math.isfinite(actual) and abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
        f"{label}: expected {expected:.17g}, received {actual:.17g}",
    )


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def kerr_schild(mass: float, chi: list[float], x: list[float]) -> dict:
    """Closed-form Kerr-Schild metric, inverse and 3+1 fields, (-,+,+,+)."""
    a = [mass * c for c in chi]
    rho2, a2, ax = dot(x, x), dot(a, a), dot(a, x)
    d = rho2 - a2
    r = math.sqrt(0.5 * (d + math.sqrt(d * d + 4.0 * ax * ax)))
    h = mass * r**3 / (r**4 + ax * ax)
    xa = cross(x, a)
    spatial = [(r * x[i] + xa[i] + ax * a[i] / r) / (r * r + a2) for i in range(3)]
    l_down = [1.0, *spatial]
    l_up = [-1.0, *spatial]
    eta = [[-1.0 if i == j == 0 else (1.0 if i == j else 0.0) for j in range(4)] for i in range(4)]
    metric = [[eta[i][j] + 2.0 * h * l_down[i] * l_down[j] for j in range(4)] for i in range(4)]
    inverse = [[eta[i][j] - 2.0 * h * l_up[i] * l_up[j] for j in range(4)] for i in range(4)]
    lapse = 1.0 / math.sqrt(-inverse[0][0])
    shift = [-inverse[0][i + 1] / inverse[0][0] for i in range(3)]
    gamma_inverse = [
        [inverse[i + 1][j + 1] + shift[i] * shift[j] / lapse**2 for j in range(3)]
        for i in range(3)
    ]
    return {"metric": metric, "inverse": inverse, "lapse": lapse, "shift": shift, "gammaInverse": gamma_inverse}


def node_binary() -> str:
    candidates = [os.environ.get("NODE_BINARY"), shutil.which("node")]
    node = next(
        (candidate for candidate in candidates if candidate and pathlib.Path(candidate).is_file()),
        None,
    )
    require(node is not None, "Node.js is required; install it or set NODE_BINARY to its executable")
    return node


def run_node_probe() -> dict:
    module_uri = MODULE.as_uri()
    source = f"""
      import {{
        createPnEobOrbitAdapter,
        createStrongFieldSpacetimeProvider,
        evaluateKerrSchild3p1,
        nullHamiltonian,
      }} from {json.dumps(module_uri)};
      const cases = {json.dumps(CASES)};
      const momenta = {json.dumps(MOMENTA)};
      const samples = cases.map(([massM, dimensionlessSpin, positionM]) => {{
        const fields = evaluateKerrSchild3p1({{ massM, dimensionlessSpin, positionM }});
        return {{
          metric: fields.covariantMetric,
          lapse: fields.lapse,
          shift: fields.shift,
          gammaInverse: fields.inverseSpatialMetric,
          regularized: fields.regularized,
          energies: momenta.map((p) => nullHamiltonian(fields, p)),
        }};
      }});

      const hole = (id, x) => ({{
        id, massM: 0.5, positionM: [x, 0, 0], velocityC: [0, 0, 0], dimensionlessSpin: [0, 0, 0],
      }});
      const sample = (separationM) => () => ({{
        bodies: [hole("A", -separationM / 2), hole("B", separationM / 2)],
        remnant: {{ id: "R", massM: 0.95, positionM: [0, 0, 0], velocityC: [0, 0, 0], dimensionlessSpin: [0, 0, 0.68] }},
        mergerBlend: 0,
      }});
      const adapter = (separationM) => createPnEobOrbitAdapter({{
        dynamicsModel: "PN verification orbit",
        coordinateFrame: "asymptotically-inertial-kerr-schild-com",
        source: "verify_strong_field.py",
        usesSxsGaugeCentroids: false,
        sample: sample(separationM),
      }});
      // Two unboosted m = 0.5 terms lose the Lorentzian signature at their
      // midpoint once the separation is below 8 sqrt(m_A m_B) = 4 M.
      const midpoint = (separationM) => createStrongFieldSpacetimeProvider({{
        orbitAdapter: adapter(separationM),
      }}).frameAt(0).evaluateOrUnresolved([0, 0, 0]);
      let sxsRejected = false;
      try {{
        createPnEobOrbitAdapter({{
          dynamicsModel: "EOB",
          coordinateFrame: "asymptotically-inertial-kerr-schild-com",
          source: "SXS horizon coordinate centers",
          usesSxsGaugeCentroids: true,
          sample: sample(10),
        }});
      }} catch (error) {{
        sxsRejected = /SXS horizon-centroid/.test(String(error?.message));
      }}
      console.log(JSON.stringify({{
        samples,
        failClosed: {{
          wide: midpoint(4.4).outcome,
          close: midpoint(3.6).outcome,
          closeReason: String(midpoint(3.6).reason ?? ""),
          sxsRejected,
        }},
      }}));
    """
    result = subprocess.run(
        [node_binary(), "--input-type=module", "--eval", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    require(MODULE.is_file(), "strong-field module is missing")
    probe = run_node_probe()

    worst_metric = worst_null = 0.0
    for (mass, chi, position), sample in zip(CASES, probe["samples"]):
        label = f"M={mass} chi={chi} x={position}"
        require(sample["regularized"] is False, f"{label}: unexpectedly regularized")
        reference = kerr_schild(mass, chi, position)
        for i in range(4):
            for j in range(4):
                close(sample["metric"][i][j], reference["metric"][i][j], 1e-12, f"{label} g[{i}][{j}]")
                worst_metric = max(worst_metric, abs(sample["metric"][i][j] - reference["metric"][i][j]))
        close(sample["lapse"], reference["lapse"], 1e-12, f"{label} lapse")
        for i in range(3):
            close(sample["shift"][i], reference["shift"][i], 1e-12, f"{label} shift[{i}]")
            for j in range(3):
                close(
                    sample["gammaInverse"][i][j],
                    reference["gammaInverse"][i][j],
                    1e-12,
                    f"{label} gamma^[{i}][{j}]",
                )
        # The oracle's energy E = alpha q - beta.p must make (-E, p) null with
        # respect to the independent closed-form inverse metric.
        for momentum, energy in zip(MOMENTA, sample["energies"]):
            covector = [-energy, *momentum]
            norm = sum(
                reference["inverse"][i][j] * covector[i] * covector[j]
                for i in range(4)
                for j in range(4)
            )
            scale = sum(abs(reference["inverse"][i][j] * covector[i] * covector[j]) for i in range(4) for j in range(4))
            worst_null = max(worst_null, abs(norm) / scale)
    require(worst_null < 1e-13, f"oracle photon energy is not null in the closed-form metric ({worst_null:.3e})")

    # Spin parity at a mirror-symmetric point: g_0y is odd, the lapse even.
    plus, minus = probe["samples"][2], probe["samples"][3]
    require(plus["metric"][0][2] < 0.0 < minus["metric"][0][2], "Kerr spin sign is wrong")
    close(plus["metric"][0][2], -minus["metric"][0][2], 1e-14, "Kerr odd spin term")
    close(plus["lapse"], minus["lapse"], 1e-14, "Kerr even lapse")

    fail_closed = probe["failClosed"]
    require(fail_closed["wide"] == "valid", "wide binary midpoint must be a valid Lorentzian sample")
    require(
        fail_closed["close"] == "unresolved",
        f"non-Lorentzian midpoint must fail closed, got {fail_closed['close']}",
    )
    require(fail_closed["sxsRejected"], "SXS horizon-centroid adapters must be rejected")

    print(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "kerr_schild_closed_form_cases": len(CASES),
                    "max_metric_abs_error": worst_metric,
                    "max_relative_null_norm": worst_null,
                    "kerr_spin_parity": True,
                    "non_lorentzian_midpoint_unresolved": fail_closed["closeReason"],
                    "sxs_gauge_centroids_rejected": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", "") or ""
        print(f"strong-field verification failed: {error} {detail}".strip(), file=sys.stderr)
        raise SystemExit(1)
