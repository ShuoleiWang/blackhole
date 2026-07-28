from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

from scripts.generate_schwarzschild_transfer_map import (
    CRITICAL_IMPACT_M,
    ESCAPE_RADIUS_M,
    OBSERVER_RADIUS_M,
    OUTCOME_CAPTURED,
    OUTCOME_ESCAPED,
    OUTCOME_UNRESOLVED,
    classify_impact_parameter,
    critical_shadow_radius_rad,
    generate_dataset,
    lapse_squared,
)
from scripts.verify_nr_contract import validate_contract
from scripts.verify_schwarzschild_transfer_map import (
    DEFAULT_MANIFEST,
    _expected_outcome_for_impact,
    validate_stationary_physics,
)


class SchwarzschildTransferMapTests(unittest.TestCase):
    def test_bundled_map_passes_contract_and_stationary_physics(self) -> None:
        contract = validate_contract(DEFAULT_MANIFEST)
        physics = validate_stationary_physics(DEFAULT_MANIFEST)

        self.assertEqual(contract["status"], "protocol-conformant")
        self.assertEqual(contract["records"], 1024 * 576)
        self.assertEqual(
            physics.escaped + physics.captured + physics.unresolved,
            physics.records,
        )
        self.assertEqual(physics.records, 1024 * 576)
        self.assertEqual(physics.unresolved, 0)
        self.assertLess(physics.max_direction_norm_error, 1.0e-6)
        self.assertLess(physics.max_independent_direction_error_rad, 1.5e-6)
        self.assertGreaterEqual(physics.direction_probe_count, 8)
        self.assertLess(physics.max_independent_lookback_error_m, 2.0e-4)
        self.assertEqual(physics.lookback_probe_count, 4)
        self.assertLess(physics.max_axis_mapping_error, 1.0e-12)
        self.assertLess(physics.max_null_residual, 5.0e-12)
        self.assertLess(physics.max_projection_error_px, 0.25)

    def test_generator_is_deterministic_and_tiled(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_root,
            tempfile.TemporaryDirectory() as second_root,
        ):
            first = Path(first_root) / "map"
            second = Path(second_root) / "map"
            with contextlib.redirect_stdout(io.StringIO()):
                first_report = generate_dataset(
                    first,
                    width=32,
                    height=18,
                    tile_height=8,
                )
                second_report = generate_dataset(
                    second,
                    width=32,
                    height=18,
                    tile_height=8,
                )

            self.assertEqual(first_report.chunks, 3)
            self.assertEqual(first_report.records, 32 * 18)
            self.assertEqual(
                first_report.escaped
                + first_report.captured
                + first_report.unresolved,
                32 * 18,
            )
            self.assertEqual(first_report.records, second_report.records)

            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second) for path in second.rglob("*") if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative_path in first_files:
                self.assertEqual(
                    (first / relative_path).read_bytes(),
                    (second / relative_path).read_bytes(),
                    relative_path.as_posix(),
                )
            self.assertEqual(
                validate_contract(first / "manifest.json")["records"],
                32 * 18,
            )

    def test_finite_observer_shadow_and_frequency_are_analytic(self) -> None:
        expected_shadow_radius = math.asin(
            CRITICAL_IMPACT_M
            * math.sqrt(lapse_squared(OBSERVER_RADIUS_M))
            / OBSERVER_RADIUS_M
        )
        expected_frequency_shift = math.sqrt(
            lapse_squared(ESCAPE_RADIUS_M)
            / lapse_squared(OBSERVER_RADIUS_M)
        )

        self.assertAlmostEqual(
            critical_shadow_radius_rad(),
            expected_shadow_radius,
            places=15,
        )
        self.assertAlmostEqual(
            math.degrees(2.0 * expected_shadow_radius),
            14.548010,
            places=6,
        )
        self.assertGreater(expected_frequency_shift, 1.0)
        self.assertAlmostEqual(expected_frequency_shift, 1.024951860, places=9)

    def test_exact_photon_separatrix_is_not_given_a_false_termination(self) -> None:
        self.assertEqual(
            classify_impact_parameter(math.nextafter(CRITICAL_IMPACT_M, 0.0)),
            OUTCOME_CAPTURED,
        )
        self.assertEqual(
            classify_impact_parameter(CRITICAL_IMPACT_M),
            OUTCOME_UNRESOLVED,
        )
        self.assertEqual(
            classify_impact_parameter(
                math.nextafter(CRITICAL_IMPACT_M, math.inf)
            ),
            OUTCOME_ESCAPED,
        )

    def test_exact_photon_sphere_separatrix_is_unresolved(self) -> None:
        self.assertEqual(
            _expected_outcome_for_impact(
                math.nextafter(CRITICAL_IMPACT_M, -math.inf)
            ),
            OUTCOME_CAPTURED,
        )
        self.assertEqual(
            _expected_outcome_for_impact(CRITICAL_IMPACT_M),
            OUTCOME_UNRESOLVED,
        )
        self.assertEqual(
            _expected_outcome_for_impact(
                math.nextafter(CRITICAL_IMPACT_M, math.inf)
            ),
            OUTCOME_ESCAPED,
        )


if __name__ == "__main__":
    unittest.main()
