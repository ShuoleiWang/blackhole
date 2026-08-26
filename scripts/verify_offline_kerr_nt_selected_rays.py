#!/usr/bin/env python3
"""Verify selected exact-Kerr/NT product rays with an independent RK4 oracle.

The product is authenticated by the structural verifier first.  A small,
deterministic set of screen points is then traced twice by the independent
fixed-step Boyer--Lindquist Hamiltonian oracle (``h`` and ``h/2``) and compared
with the production sampler's public per-ray result.

This is high-accuracy selected-ray calibration, not a full-frame proof.  It is
not NR, GRMHD, a caustic-complete ray bundle, returning-radiation transport,
polarization, or a solved atmosphere.  The independent spectral calculation
shares only the Page--Thorne radial flux scalar and says so in its report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.kerr_nt_replay import reconstruct_kerr_nt_sampler
from offline.kerr_selected_oracle import (
    FixedRk4Options,
    KerrSelectedOracleError,
    SELECTED_RAY_ORACLE_IMPLEMENTATION_ID,
    configuration_from_sampler_descriptor,
    selected_ray_observed_intensities_nu,
    trace_selected_ray_refined,
)
from scripts.verify_offline_spectral_frame import (
    DEFAULT_SCHEMA,
    validate_scientific_spectral_frame,
)


class SelectedRayVerificationError(RuntimeError):
    """A selected-ray product calibration failed closed."""


def _fail(message: str) -> None:
    raise SelectedRayVerificationError(message)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be a finite number")
    return float(value)


def _relative_difference(first: float, second: float) -> float:
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return 0.0
    return abs(first - second) / scale


def _representative_screen_points(
    frame: Mapping[str, Any],
    maximum_points: int,
) -> tuple[tuple[float, float], ...]:
    try:
        width = frame["widthPixels"]
        height = frame["heightPixels"]
        bounds = frame["screenBounds"]
        x_min = _finite(bounds["xMin"], "$.frame.screenBounds.xMin")
        x_max = _finite(bounds["xMax"], "$.frame.screenBounds.xMax")
        y_min = _finite(bounds["yMin"], "$.frame.screenBounds.yMin")
        y_max = _finite(bounds["yMax"], "$.frame.screenBounds.yMax")
    except (KeyError, TypeError) as error:
        _fail(f"malformed frame descriptor: {error}")
    if type(width) is not int or width < 1 or type(height) is not int or height < 1:
        _fail("frame dimensions must be positive integers")
    if maximum_points < 1:
        _fail("maximum_points must be positive")
    candidates = (
        (width // 2, height // 2),
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    )
    indices: list[tuple[int, int]] = []
    for candidate in candidates:
        if candidate not in indices:
            indices.append(candidate)
        if len(indices) >= maximum_points:
            break
    return tuple(
        (
            x_min + (x_index + 0.5) * (x_max - x_min) / width,
            y_min + (y_index + 0.5) * (y_max - y_min) / height,
        )
        for x_index, y_index in indices
    )


def _validate_explicit_points(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        if len(value) != 2:
            _fail(f"screen point {index} must contain x and y")
        point = (
            _finite(value[0], f"screen point {index} x"),
            _finite(value[1], f"screen point {index} y"),
        )
        if point not in points:
            points.append(point)
    if not points:
        _fail("at least one explicit screen point is required")
    return tuple(points)


def verify_selected_rays(
    manifest_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_SCHEMA,
    screen_points: Sequence[Sequence[float]] = (),
    maximum_default_points: int = 5,
    step_m: float = 0.005,
    maximum_affine_length_m: float | None = None,
    maximum_steps: int = 200_000,
    maximum_disk_radius_refinement_m: float = 5.0e-4,
    maximum_relative_g_refinement: float = 2.0e-5,
    maximum_hamiltonian_residual: float = 1.0e-7,
    maximum_relative_carter_drift: float = 1.0e-7,
    maximum_production_relative_g_difference: float = 5.0e-5,
    maximum_production_relative_intensity_difference: float = 5.0e-4,
) -> dict[str, Any]:
    """Authenticate a product and calibrate a bounded set of exact rays."""

    path = Path(manifest_path).absolute()
    before = path.read_bytes()
    before_sha256 = hashlib.sha256(before).hexdigest()
    structural = validate_scientific_spectral_frame(path, schema_path)
    if path.read_bytes() != before:
        _fail("manifest changed during structural authentication")
    try:
        manifest = json.loads(before)
        descriptor = manifest["sampler"]["descriptor"]
        frame = manifest["frame"]
        frequencies = tuple(
            _finite(value, "$.observerFrequencyBinsHz")
            for value in manifest["observerFrequencyBinsHz"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        _fail(f"authenticated manifest cannot be interpreted: {error}")
    if not isinstance(descriptor, dict) or not isinstance(frame, dict):
        _fail("sampler and frame descriptors must be objects")
    if not frequencies or any(value <= 0.0 for value in frequencies):
        _fail("observer frequency bins must be positive and non-empty")

    configuration = configuration_from_sampler_descriptor(descriptor)
    production_sampler = reconstruct_kerr_nt_sampler(descriptor)
    if screen_points:
        selected = _validate_explicit_points(screen_points)
    else:
        selected = _representative_screen_points(frame, maximum_default_points)

    if maximum_affine_length_m is None:
        try:
            maximum_affine_length_m = _finite(
                descriptor["rayOptions"]["fine"]["maximumAffineLength"],
                "$.sampler.descriptor.rayOptions.fine.maximumAffineLength",
            )
        except (KeyError, TypeError) as error:
            _fail(f"missing production affine-length bound: {error}")
    options = FixedRk4Options(
        step_m=step_m,
        maximum_affine_length_m=maximum_affine_length_m,
        maximum_steps=maximum_steps,
    )
    non_negative_limits = (
        maximum_disk_radius_refinement_m,
        maximum_relative_g_refinement,
        maximum_hamiltonian_residual,
        maximum_relative_carter_drift,
        maximum_production_relative_g_difference,
        maximum_production_relative_intensity_difference,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in non_negative_limits):
        raise ValueError("selected-ray comparison limits must be finite and non-negative")

    reports: list[dict[str, Any]] = []
    maximum_radius_refinement = 0.0
    maximum_g_refinement = 0.0
    maximum_production_g = 0.0
    maximum_production_intensity = 0.0
    for index, (screen_x, screen_y) in enumerate(selected):
        refinement = trace_selected_ray_refined(
            configuration,
            screen_x,
            screen_y,
            options,
        )
        coarse = refinement.coarse
        oracle = refinement.fine
        if not refinement.outcome_agrees:
            _fail(
                f"selected ray {index} changes outcome between h and h/2: "
                f"{coarse.outcome!r} versus {oracle.outcome!r}"
            )
        if oracle.outcome == "unresolved":
            _fail(f"selected ray {index} exhausted the independent affine bound")
        for label, result in (("coarse", coarse), ("fine", oracle)):
            if result.maximum_hamiltonian_residual > maximum_hamiltonian_residual:
                _fail(
                    f"selected ray {index} {label} Hamiltonian residual "
                    f"{result.maximum_hamiltonian_residual:.6e} exceeds the limit"
                )
            if result.maximum_relative_carter_drift > maximum_relative_carter_drift:
                _fail(
                    f"selected ray {index} {label} Carter drift "
                    f"{result.maximum_relative_carter_drift:.6e} exceeds the limit"
                )
        if refinement.disk_radius_difference_m is not None:
            maximum_radius_refinement = max(
                maximum_radius_refinement,
                refinement.disk_radius_difference_m,
            )
            if refinement.disk_radius_difference_m > maximum_disk_radius_refinement_m:
                _fail(
                    f"selected ray {index} disk radius h/h2 difference "
                    f"{refinement.disk_radius_difference_m:.6e} m exceeds the limit"
                )
        if refinement.relative_g_difference is not None:
            maximum_g_refinement = max(
                maximum_g_refinement, refinement.relative_g_difference
            )
            if refinement.relative_g_difference > maximum_relative_g_refinement:
                _fail(
                    f"selected ray {index} g h/h2 difference "
                    f"{refinement.relative_g_difference:.6e} exceeds the limit"
                )

        production = production_sampler.sample(screen_x, screen_y, frequencies)
        expected_source = {
            "disk": "disk",
            "captured": "captured-boundary",
            "escaped": "escaped-boundary",
        }[oracle.outcome]
        if production.visible_source != expected_source:
            _fail(
                f"selected ray {index} source mismatch: independent "
                f"{expected_source!r}, production {production.visible_source!r}"
            )
        production_g_difference = None
        production_intensity_difference = None
        independent_intensities: tuple[float, ...] | None = None
        if oracle.outcome == "disk":
            if production.frequency_shift_g is None or oracle.frequency_shift_g is None:
                _fail(f"selected ray {index} disk comparison lacks g")
            production_g_difference = _relative_difference(
                oracle.frequency_shift_g, production.frequency_shift_g
            )
            maximum_production_g = max(
                maximum_production_g, production_g_difference
            )
            if production_g_difference > maximum_production_relative_g_difference:
                _fail(
                    f"selected ray {index} independent/production g difference "
                    f"{production_g_difference:.6e} exceeds the limit"
                )
            independent_intensities = selected_ray_observed_intensities_nu(
                configuration, oracle, frequencies
            )
            differences = tuple(
                _relative_difference(independent, production_value)
                for independent, production_value in zip(
                    independent_intensities,
                    production.specific_intensities_nu,
                )
            )
            production_intensity_difference = max(differences, default=0.0)
            maximum_production_intensity = max(
                maximum_production_intensity,
                production_intensity_difference,
            )
            if (
                production_intensity_difference
                > maximum_production_relative_intensity_difference
            ):
                _fail(
                    f"selected ray {index} independent/production I_nu difference "
                    f"{production_intensity_difference:.6e} exceeds the limit"
                )

        reports.append(
            {
                "screen": {"x": screen_x, "y": screen_y},
                "outcome": oracle.outcome,
                "coarseSteps": coarse.steps,
                "fineSteps": oracle.steps,
                "affineLengthM": oracle.affine_length_m,
                "terminalRadiusM": oracle.terminal_radius_m,
                "diskRadiusIndependentM": oracle.disk_radius_m,
                "diskRadiusProductionPersisted": False,
                "diskRadiusHvsHalfDifferenceM": (
                    refinement.disk_radius_difference_m
                ),
                "frequencyShiftIndependent": oracle.frequency_shift_g,
                "frequencyShiftProduction": production.frequency_shift_g,
                "relativeGIndependentVsProduction": production_g_difference,
                "relativeGHvsHalfDifference": refinement.relative_g_difference,
                "emissionAngleCosineIndependent": oracle.emission_angle_cosine,
                "independentSpecificIntensitiesNu": independent_intensities,
                "productionSpecificIntensitiesNu": (
                    production.specific_intensities_nu
                ),
                "maximumRelativeIntensityIndependentVsProduction": (
                    production_intensity_difference
                ),
                "constants": {
                    "energy": oracle.constants.energy,
                    "angularMomentumZ": oracle.constants.angular_momentum_z,
                    "carterQ": oracle.constants.carter_q,
                    "carterK": oracle.constants.carter_k,
                },
                "maximumHamiltonianResidual": (
                    oracle.maximum_hamiltonian_residual
                ),
                "maximumRelativeCarterDrift": (
                    oracle.maximum_relative_carter_drift
                ),
            }
        )

    if path.read_bytes() != before:
        _fail("manifest changed during selected-ray verification")
    final_structural = validate_scientific_spectral_frame(path, schema_path)
    if final_structural != structural or path.read_bytes() != before:
        _fail("authenticated product evidence changed during selected-ray verification")
    return {
        "id": manifest["id"],
        "status": "selected-exact-kerr-nt-rays-calibrated",
        "manifestSha256": before_sha256,
        "structuralContractVerified": True,
        "selectedRayCalibrationVerified": True,
        "oracleImplementationId": SELECTED_RAY_ORACLE_IMPLEMENTATION_ID,
        "independentGeodesicIntegrator": True,
        "independentEventLocator": True,
        "productionTracerCalledByOracle": False,
        "productionSamplerUsedOnlyAsComparator": True,
        "spectralIndependence": "shared-page-thorne-radial-scalar",
        "fullFramePhysicsProof": False,
        "isNumericalRelativitySolver": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isSolvedAtmosphere": False,
        "includesReturningRadiation": False,
        "includesPolarization": False,
        "diskRadiusComparisonBoundary": (
            "h-vs-h/2 independent radius only; production SpectralRaySample and "
            "the public pixel ABI do not persist per-ray intersection radius"
        ),
        "stepM": options.step_m,
        "halfStepM": 0.5 * options.step_m,
        "rayCount": len(reports),
        "maximumDiskRadiusHvsHalfDifferenceM": maximum_radius_refinement,
        "maximumRelativeGHvsHalfDifference": maximum_g_refinement,
        "maximumRelativeGIndependentVsProduction": maximum_production_g,
        "maximumRelativeIntensityIndependentVsProduction": (
            maximum_production_intensity
        ),
        "rays": reports,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--screen-point",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=[],
        help="explicit selected screen point; repeat as needed",
    )
    parser.add_argument("--maximum-default-points", type=int, default=5)
    parser.add_argument("--step-m", type=float, default=0.005)
    parser.add_argument("--maximum-affine-length-m", type=float, default=None)
    parser.add_argument("--maximum-steps", type=int, default=200_000)
    parser.add_argument(
        "--maximum-disk-radius-refinement-m", type=float, default=5.0e-4
    )
    parser.add_argument("--maximum-relative-g-refinement", type=float, default=2.0e-5)
    parser.add_argument("--maximum-hamiltonian-residual", type=float, default=1.0e-7)
    parser.add_argument("--maximum-relative-carter-drift", type=float, default=1.0e-7)
    parser.add_argument(
        "--maximum-production-relative-g-difference", type=float, default=5.0e-5
    )
    parser.add_argument(
        "--maximum-production-relative-intensity-difference",
        type=float,
        default=5.0e-4,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        report = verify_selected_rays(
            arguments.manifest,
            schema_path=arguments.schema,
            screen_points=arguments.screen_point,
            maximum_default_points=arguments.maximum_default_points,
            step_m=arguments.step_m,
            maximum_affine_length_m=arguments.maximum_affine_length_m,
            maximum_steps=arguments.maximum_steps,
            maximum_disk_radius_refinement_m=(
                arguments.maximum_disk_radius_refinement_m
            ),
            maximum_relative_g_refinement=(
                arguments.maximum_relative_g_refinement
            ),
            maximum_hamiltonian_residual=(
                arguments.maximum_hamiltonian_residual
            ),
            maximum_relative_carter_drift=(
                arguments.maximum_relative_carter_drift
            ),
            maximum_production_relative_g_difference=(
                arguments.maximum_production_relative_g_difference
            ),
            maximum_production_relative_intensity_difference=(
                arguments.maximum_production_relative_intensity_difference
            ),
        )
    except (
        KerrSelectedOracleError,
        SelectedRayVerificationError,
        OSError,
        ValueError,
    ) as error:
        print(f"selected-ray verification failed: {error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
