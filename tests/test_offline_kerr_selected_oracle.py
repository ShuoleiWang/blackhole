from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from offline.kerr_selected_oracle import (
    FixedRk4Options,
    KerrSelectedOracleError,
    configuration_from_sampler_descriptor,
    selected_ray_observed_intensities_nu,
    trace_selected_ray,
    trace_selected_ray_refined,
)
import scripts.render_offline_kerr_nt_frame as renderer


class KerrSelectedRayOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = Path(tempfile.gettempdir())
        cls.kerr_plan = cls._plan(temporary, 0.7, "kerr")
        cls.schwarzschild_plan = cls._plan(temporary, 0.0, "schwarzschild")
        cls.kerr_descriptor = cls.kerr_plan.sampler.descriptor()
        cls.schwarzschild_descriptor = cls.schwarzschild_plan.sampler.descriptor()
        cls.kerr = configuration_from_sampler_descriptor(cls.kerr_descriptor)
        cls.schwarzschild = configuration_from_sampler_descriptor(
            cls.schwarzschild_descriptor
        )
        cls.options = FixedRk4Options(
            step_m=0.02,
            maximum_affine_length_m=150.0,
            maximum_steps=20_000,
        )

    @staticmethod
    def _plan(root: Path, spin: float, label: str):
        arguments = renderer.parse_args(
            [
                str(root / f"unused-selected-ray-{label}-product"),
                "--cache",
                str(root / f"unused-selected-ray-{label}-cache"),
                "--spin",
                str(spin),
                "--frequency-hz",
                "5e14",
                "--ray-maximum-affine-length",
                "150",
            ]
        )
        return renderer.build_render_plan(arguments)

    def test_schwarzschild_and_kerr_selected_disk_rays_converge(self) -> None:
        for configuration in (self.schwarzschild, self.kerr):
            with self.subTest(spin=configuration.dimensionless_spin):
                refinement = trace_selected_ray_refined(
                    configuration,
                    0.5,
                    -0.5,
                    self.options,
                )
                self.assertTrue(refinement.outcome_agrees)
                self.assertEqual(refinement.fine.outcome, "disk")
                self.assertIsNotNone(refinement.disk_radius_difference_m)
                self.assertLess(refinement.disk_radius_difference_m, 1.0e-8)
                self.assertIsNotNone(refinement.relative_g_difference)
                self.assertLess(refinement.relative_g_difference, 1.0e-10)
                self.assertLess(
                    refinement.fine.maximum_hamiltonian_residual,
                    1.0e-10,
                )
                self.assertLess(
                    refinement.fine.maximum_relative_carter_drift,
                    1.0e-10,
                )

    def test_kerr_approaching_and_receding_disk_sides_differ(self) -> None:
        receding = trace_selected_ray(self.kerr, -0.5, -0.5, self.options)
        approaching = trace_selected_ray(self.kerr, 0.5, -0.5, self.options)
        self.assertEqual(receding.outcome, "disk")
        self.assertEqual(approaching.outcome, "disk")
        self.assertIsNotNone(receding.frequency_shift_g)
        self.assertIsNotNone(approaching.frequency_shift_g)
        self.assertLess(receding.frequency_shift_g, approaching.frequency_shift_g)
        self.assertLess(receding.frequency_shift_g, 1.0)
        self.assertGreater(approaching.frequency_shift_g, 1.0)

    def test_first_opaque_disk_capture_and_escape_are_distinct(self) -> None:
        disk = trace_selected_ray(self.kerr, 0.5, -0.5, self.options)
        captured = trace_selected_ray(self.kerr, 0.0, 0.0, self.options)
        escaped = trace_selected_ray(self.kerr, 0.0, 0.5, self.options)
        self.assertEqual(disk.outcome, "disk")
        self.assertGreaterEqual(disk.disk_radius_m, self.kerr.isco_radius_m)
        self.assertLessEqual(disk.disk_radius_m, self.kerr.disk_outer_radius_m)
        self.assertEqual(captured.outcome, "captured")
        self.assertAlmostEqual(
            captured.terminal_radius_m,
            self.kerr.capture_radius_m,
            places=11,
        )
        self.assertEqual(escaped.outcome, "escaped")
        self.assertAlmostEqual(
            escaped.terminal_radius_m,
            self.kerr.escape_radius_m,
            places=10,
        )
        frequencies = (5.0e14,)
        self.assertEqual(
            self.kerr_plan.sampler.sample(0.0, 0.0, frequencies).visible_source,
            "captured-boundary",
        )
        self.assertEqual(
            self.kerr_plan.sampler.sample(0.0, 0.5, frequencies).visible_source,
            "escaped-boundary",
        )

    def test_independent_transfer_matches_production_public_sample(self) -> None:
        refinement = trace_selected_ray_refined(
            self.kerr,
            0.5,
            -0.5,
            self.options,
        )
        oracle = refinement.fine
        frequencies = (5.0e14,)
        independent_intensity = selected_ray_observed_intensities_nu(
            self.kerr,
            oracle,
            frequencies,
        )
        production = self.kerr_plan.sampler.sample(0.5, -0.5, frequencies)
        self.assertEqual(production.visible_source, "disk")
        self.assertAlmostEqual(
            oracle.frequency_shift_g,
            production.frequency_shift_g,
            places=11,
        )
        self.assertAlmostEqual(
            independent_intensity[0] / production.specific_intensities_nu[0],
            1.0,
            places=10,
        )

    def test_unsupported_or_internally_inconsistent_descriptor_fails_closed(self) -> None:
        wrong_implementation = copy.deepcopy(self.kerr_descriptor)
        wrong_implementation["implementationId"] = "approximate-kerr/v1"
        with self.assertRaisesRegex(KerrSelectedOracleError, "only .* is supported"):
            configuration_from_sampler_descriptor(wrong_implementation)

        wrong_screen = copy.deepcopy(self.kerr_descriptor)
        wrong_screen["screenConvention"]["screenX"] = "opposite-sign"
        with self.assertRaisesRegex(KerrSelectedOracleError, "screen convention"):
            configuration_from_sampler_descriptor(wrong_screen)

        wrong_isco = copy.deepcopy(self.kerr_descriptor)
        wrong_isco["disk"]["iscoRadiusM"] *= 1.01
        with self.assertRaisesRegex(KerrSelectedOracleError, "ISCO disagrees"):
            configuration_from_sampler_descriptor(wrong_isco)

        wrong_angular_hash = copy.deepcopy(self.kerr_descriptor)
        wrong_angular_hash["angularEmission"]["descriptorSha256"] = "0" * 64
        with self.assertRaisesRegex(KerrSelectedOracleError, "hash does not bind"):
            configuration_from_sampler_descriptor(wrong_angular_hash)


if __name__ == "__main__":
    unittest.main()
