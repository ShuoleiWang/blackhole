#!/usr/bin/env python3
"""Independent acceptance checks for the strong-field CPU spacetime oracle."""

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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    require(
        math.isfinite(actual) and abs(actual - expected) <= tolerance,
        f"{label}: expected {expected:.17g}, received {actual:.17g}",
    )


def run_node_probe() -> dict:
    candidates = [
        os.environ.get("NODE_BINARY"),
        shutil.which("node"),
    ]
    node = next(
        (
            candidate
            for candidate in candidates
            if candidate and pathlib.Path(candidate).is_file()
        ),
        None,
    )
    require(
        node is not None,
        "Node.js is required; install it or set NODE_BINARY to its executable",
    )
    module_uri = MODULE.as_uri()
    source = f"""
      import {{
        evaluateKerrSchild3p1,
        nullHamiltonianResidual,
      }} from {json.dumps(module_uri)};
      const schwarzschild = evaluateKerrSchild3p1({{
        massM: 1,
        positionM: [10, 0, 0],
      }});
      const kerrPlus = evaluateKerrSchild3p1({{
        massM: 1,
        positionM: [8, 0, 0],
        dimensionlessSpin: [0, 0, 0.7],
      }});
      const kerrMinus = evaluateKerrSchild3p1({{
        massM: 1,
        positionM: [8, 0, 0],
        dimensionlessSpin: [0, 0, -0.7],
      }});
      const residual = nullHamiltonianResidual(
        kerrPlus,
        [0.7, -0.2, 1.1],
      );
      console.log(JSON.stringify({{
        schwarzschild: {{
          lapse: schwarzschild.lapse,
          shiftX: schwarzschild.shift[0],
          gammaInvXX: schwarzschild.inverseSpatialMetric[0][0],
          g00: schwarzschild.covariantMetric[0][0],
        }},
        kerr: {{
          plusG0Y: kerrPlus.covariantMetric[0][2],
          minusG0Y: kerrMinus.covariantMetric[0][2],
          plusLapse: kerrPlus.lapse,
          minusLapse: kerrMinus.lapse,
        }},
        nullResidual: residual.normalized,
      }}));
    """
    result = subprocess.run(
        [node, "--input-type=module", "--eval", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    require(MODULE.is_file(), "strong-field module is missing")
    probe = run_node_probe()

    two_h = 2.0 / 10.0
    expected_lapse = 1.0 / math.sqrt(1.0 + two_h)
    expected_shift = two_h / (1.0 + two_h)
    expected_gamma_inv_xx = 1.0 / (1.0 + two_h)
    schwarzschild = probe["schwarzschild"]
    close(schwarzschild["lapse"], expected_lapse, 2e-14, "Schwarzschild lapse")
    close(schwarzschild["shiftX"], expected_shift, 2e-14, "Schwarzschild shift")
    close(
        schwarzschild["gammaInvXX"],
        expected_gamma_inv_xx,
        2e-14,
        "Schwarzschild inverse spatial metric",
    )
    close(schwarzschild["g00"], -1.0 + two_h, 2e-14, "Schwarzschild g00")

    kerr = probe["kerr"]
    require(kerr["plusG0Y"] < 0 < kerr["minusG0Y"], "Kerr spin sign is wrong")
    close(kerr["plusG0Y"], -kerr["minusG0Y"], 2e-14, "Kerr odd spin term")
    close(kerr["plusLapse"], kerr["minusLapse"], 2e-14, "Kerr even lapse")
    require(probe["nullResidual"] < 2e-14, "null Hamiltonian residual is too large")

    source = MODULE.read_text(encoding="utf-8")
    for required_boundary in (
        "not constraint-solved NR",
        "usesSxsGaugeCentroids === false",
        'outcome: "unresolved"',
        "STRONG_FIELD_UNIFORM_ABI",
    ):
        require(
            required_boundary in source,
            f"missing fail-closed/scientific boundary: {required_boundary}",
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "checks": {
                    "schwarzschild_closed_form": True,
                    "kerr_spin_parity": True,
                    "null_hamiltonian": True,
                    "sxs_gauge_centroids_rejected": True,
                    "unresolved_fail_closed": True,
                },
                "normalizedNullResidual": probe["nullResidual"],
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
        print(f"strong-field verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
