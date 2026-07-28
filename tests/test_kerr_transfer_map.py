from __future__ import annotations

import contextlib
import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.generate_kerr_transfer_map import (
    ESCAPE_RADIUS_M,
    OBSERVER_RADIUS_M,
    OUTCOME_CAPTURED,
    OUTCOME_ESCAPED,
    OUTCOME_UNRESOLVED,
    SPIN_A_M,
    VERTICAL_FOV_RAD,
    capture_radius_m,
    equatorial_shadow_intercepts,
    generate_dataset as generate_kerr_dataset,
    horizon_radius_m,
    solve_ray,
)
from scripts.generate_schwarzschild_transfer_map import (
    generate_dataset as generate_schwarzschild_dataset,
)
from scripts.verify_kerr_transfer_map import (
    DEFAULT_MANIFEST,
    EXPECTED_CAPTURE_RADIUS_M,
    EXPECTED_SHADOW_TOP_X,
    EXPECTED_SHADOW_TOP_Y,
    EXPECTED_SHADOW_X_MAX,
    EXPECTED_SHADOW_X_MIN,
    _analytic_shadow_curve,
    _angular_separation,
    validate_kerr_physics,
)
from scripts.verify_nr_contract import validate_contract


RECORD = struct.Struct("<7fBBH")


def _records(dataset: Path) -> list[tuple[float | int, ...]]:
    unpacked: list[tuple[float | int, ...]] = []
    for chunk in sorted((dataset / "chunks").glob("*.bin")):
        unpacked.extend(RECORD.iter_unpack(chunk.read_bytes()))
    return unpacked


class KerrTransferMapTests(unittest.TestCase):
    def test_bundled_map_passes_contract_and_independent_kerr_physics(self) -> None:
        contract = validate_contract(DEFAULT_MANIFEST)
        physics = validate_kerr_physics(DEFAULT_MANIFEST)

        self.assertEqual(contract["status"], "protocol-conformant")
        self.assertEqual(contract["records"], 1024 * 576)
        self.assertEqual(physics.records, 1024 * 576)
        self.assertEqual(physics.unresolved, 0)
        self.assertEqual(physics.analytic_mask_mismatches, 0)
        self.assertGreaterEqual(physics.direction_probe_count, 6)
        self.assertLess(physics.max_direction_norm_error, 1.0e-6)
        self.assertLess(
            physics.max_independent_direction_error_rad,
            2.0e-6,
        )
        self.assertLess(
            physics.max_fixed_step_refinement_rad,
            8.0e-7,
        )
        self.assertLess(
            physics.max_independent_separation_residual,
            1.0e-8,
        )
        self.assertLess(physics.max_null_residual, 1.0e-8)
        self.assertLess(physics.max_projection_error_px, 0.25)
        self.assertLess(physics.max_axis_mapping_error, 1.0e-12)
        self.assertLess(
            physics.max_kerr_observer_identity_error,
            1.0e-11,
        )

    def test_generator_is_deterministic_tiled_and_protocol_conformant(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_root,
            tempfile.TemporaryDirectory() as second_root,
        ):
            first = Path(first_root) / "map"
            second = Path(second_root) / "map"
            with contextlib.redirect_stdout(io.StringIO()):
                first_report = generate_kerr_dataset(
                    first,
                    width=16,
                    height=8,
                    tile_height=3,
                    jobs=1,
                )
                second_report = generate_kerr_dataset(
                    second,
                    width=16,
                    height=8,
                    tile_height=3,
                    jobs=1,
                )

            self.assertEqual(first_report.chunks, 3)
            self.assertEqual(first_report.records, 16 * 8)
            self.assertEqual(first_report.unresolved, 0)
            self.assertEqual(first_report.records, second_report.records)
            first_files = sorted(
                path.relative_to(first)
                for path in first.rglob("*")
                if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second)
                for path in second.rglob("*")
                if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                    relative.as_posix(),
                )
            self.assertEqual(
                validate_contract(first / "manifest.json")["records"],
                16 * 8,
            )

    def test_finite_zamo_shadow_and_spin_reversal_are_analytic(self) -> None:
        curve = _analytic_shadow_curve(SPIN_A_M)
        mirror = _analytic_shadow_curve(-SPIN_A_M)
        self.assertAlmostEqual(
            curve.x_min, EXPECTED_SHADOW_X_MIN, delta=2.0e-6
        )
        self.assertAlmostEqual(
            curve.x_max, EXPECTED_SHADOW_X_MAX, delta=2.0e-6
        )
        self.assertAlmostEqual(
            curve.top_x, EXPECTED_SHADOW_TOP_X, delta=2.0e-6
        )
        self.assertAlmostEqual(
            curve.top_y, EXPECTED_SHADOW_TOP_Y, delta=2.0e-6
        )
        self.assertAlmostEqual(mirror.x_min, -curve.x_max, delta=2.0e-6)
        self.assertAlmostEqual(mirror.x_max, -curve.x_min, delta=2.0e-6)
        self.assertAlmostEqual(mirror.top_x, -curve.top_x, delta=2.0e-6)
        self.assertAlmostEqual(mirror.top_y, curve.top_y, delta=2.0e-6)

        self.assertAlmostEqual(
            horizon_radius_m(),
            1.727165982913406,
            places=15,
        )
        self.assertAlmostEqual(
            capture_radius_m(),
            EXPECTED_CAPTURE_RADIUS_M,
            places=15,
        )

    def test_exact_equatorial_separatrices_are_not_false_terminations(self) -> None:
        focal_pixels = 576 / (2.0 * math.tan(0.5 * VERTICAL_FOV_RAD))
        for intercept in equatorial_shadow_intercepts():
            self.assertEqual(
                solve_ray(intercept, 0.0, focal_pixels).outcome,
                OUTCOME_UNRESOLVED,
            )

    def test_a_zero_regresses_schwarzschild_outcomes_directions_and_shift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            kerr = base / "kerr-a0"
            schwarzschild = base / "schwarzschild"
            with contextlib.redirect_stdout(io.StringIO()):
                generate_kerr_dataset(
                    kerr,
                    width=16,
                    height=8,
                    tile_height=4,
                    spin=0.0,
                    jobs=1,
                )
                generate_schwarzschild_dataset(
                    schwarzschild,
                    width=16,
                    height=8,
                    tile_height=4,
                )
            kerr_records = _records(kerr)
            schwarzschild_records = _records(schwarzschild)
            self.assertEqual(len(kerr_records), len(schwarzschild_records))
            for kerr_record, schwarzschild_record in zip(
                kerr_records, schwarzschild_records
            ):
                self.assertEqual(kerr_record[7], schwarzschild_record[7])
                if kerr_record[7] == OUTCOME_ESCAPED:
                    self.assertLess(
                        max(
                            abs(float(kerr_record[index]) - float(schwarzschild_record[index]))
                            for index in range(3)
                        ),
                        2.0e-6,
                    )
                    self.assertAlmostEqual(
                        float(kerr_record[3]),
                        float(schwarzschild_record[3]),
                        places=7,
                    )

                terminal_radius = (
                    ESCAPE_RADIUS_M
                    if kerr_record[7] == OUTCOME_ESCAPED
                    else capture_radius_m(0.0)
                )
                bl_to_ks_endpoint_offset = 2.0 * math.log(
                    (terminal_radius - 2.0)
                    / (OBSERVER_RADIUS_M - 2.0)
                )
                expected_ks_lookback = (
                    float(schwarzschild_record[4])
                    - bl_to_ks_endpoint_offset
                )
                self.assertLess(
                    abs(float(kerr_record[4]) - expected_ks_lookback),
                    2.0e-3,
                )

    def test_opposite_spin_maps_are_horizontal_mirrors(self) -> None:
        width = 24
        height = 12
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            positive = base / "positive"
            negative = base / "negative"
            with contextlib.redirect_stdout(io.StringIO()):
                generate_kerr_dataset(
                    positive,
                    width=width,
                    height=height,
                    tile_height=6,
                    spin=SPIN_A_M,
                    jobs=1,
                )
                generate_kerr_dataset(
                    negative,
                    width=width,
                    height=height,
                    tile_height=6,
                    spin=-SPIN_A_M,
                    jobs=1,
                )
            positive_records = _records(positive)
            negative_records = _records(negative)
            for y in range(height):
                for x in range(width):
                    first = positive_records[y * width + x]
                    second = negative_records[y * width + (width - 1 - x)]
                    self.assertEqual(first[7], second[7])
                    if first[7] == OUTCOME_ESCAPED:
                        expected_direction = (
                            -float(first[0]),
                            float(first[1]),
                            float(first[2]),
                        )
                        self.assertLess(
                            _angular_separation(
                                expected_direction,
                                tuple(float(second[index]) for index in range(3)),
                            ),
                            2.0e-6,
                        )
                        self.assertAlmostEqual(
                            float(first[3]),
                            float(second[3]),
                            places=6,
                        )

    def test_manifest_pins_remnant_spin_source_and_kerr_semantics(self) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        roles = {
            artifact["role"]: artifact
            for artifact in manifest["provenance"]["sourceArtifacts"]
        }
        self.assertIn("remnant-spin-source", roles)
        self.assertEqual(
            roles["remnant-spin-source"]["uri"],
            "assets/scenes/binary-sxs-bbh-0001-v2.json",
        )
        self.assertEqual(
            manifest["escapeBoundary"]["surface"]["kind"],
            "constant-Kerr-r-oblate-worldtube",
        )
        self.assertEqual(
            manifest["escapeBoundary"]["referenceObserver"]["kind"],
            "Boyer-Lindquist-ZAMO",
        )
        self.assertEqual(
            manifest["projection"]["verticalFieldOfViewRad"],
            VERTICAL_FOV_RAD,
        )


if __name__ == "__main__":
    unittest.main()
