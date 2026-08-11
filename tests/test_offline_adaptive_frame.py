from __future__ import annotations

import math
import unittest
from dataclasses import replace

from offline.adaptive_frame import (
    AdaptivePixelOptions,
    AdaptiveSamplingError,
    RayConvergenceAudit,
    SpectralRaySample,
    integrate_spectral_pixel,
    pinhole_solid_angle,
)


def _sample(
    intensity: float,
    *,
    source: str = "sky",
    topology: str | None = None,
    shift: float | None = None,
    direction: tuple[float, float, float] | None = None,
    converged: bool = True,
) -> SpectralRaySample:
    return SpectralRaySample(
        specific_intensities_nu=(intensity,),
        absolute_errors_nu=(0.0,),
        visible_source=source,
        topology_signature=topology or source,
        frequency_shift_g=shift,
        escape_direction=direction,
        ray_converged=converged,
        convergence_audit=RayConvergenceAudit(
            accepted_steps=1,
            ray_gate_passed=converged,
            source_gate_passed=converged,
            transfer_gate_passed=converged,
        ),
    )


def _options(**overrides: object) -> AdaptivePixelOptions:
    values: dict[str, object] = {
        "minimum_depth": 0,
        "maximum_depth": 3,
        "maximum_ray_evaluations": 2_000,
        "radiance_absolute_tolerances": (1.0e-12,),
        "radiance_relative_tolerance": 1.0e-6,
        "unresolved_solid_angle_fraction_tolerance": 0.0,
        "weighted_log_g_tolerance": 1.0,
        "weighted_direction_tolerance_rad": 1.0,
        "radiance_guard_ceilings": (10.0,),
    }
    values.update(overrides)
    return AdaptivePixelOptions(**values)  # type: ignore[arg-type]


class OfflineAdaptiveFrameTests(unittest.TestCase):
    def test_constant_specific_intensity_preserves_mean_off_axis(self) -> None:
        for bounds in ((-0.1, 0.1, -0.1, 0.1), (1.2, 1.4, -0.7, -0.4)):
            result = integrate_spectral_pixel(
                lambda _x, _y, _frequencies: _sample(3.25),
                (4.0e14,),
                x_min=bounds[0],
                x_max=bounds[1],
                y_min=bounds[2],
                y_max=bounds[3],
                options=_options(),
            )
            self.assertTrue(result.converged)
            self.assertEqual(result.sample_count, 13)
            self.assertAlmostEqual(result.mean_specific_intensities_nu[0], 3.25, places=6)
            self.assertAlmostEqual(
                result.integrated_specific_intensity_nu_sr[0],
                3.25 * result.pixel_solid_angle_sr,
                delta=1.0e-7 * result.pixel_solid_angle_sr,
            )

    def test_parent_child_equal_mean_does_not_hide_guard_spread(self) -> None:
        def field(x_value: float, y_value: float, _frequencies: tuple[float, ...]):
            del y_value
            return _sample(1.0 if 0.115 < x_value < 0.135 else 0.0)

        result = integrate_spectral_pixel(
            field,
            (5.0e14,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(maximum_depth=2),
        )
        self.assertEqual(result.maximum_depth_reached, 2)
        self.assertGreater(result.sample_count, 13)

    def test_mixed_topology_at_maximum_depth_is_explicitly_unresolved(self) -> None:
        def edge(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            return _sample(
                2.0 if x_value < 0.07337 else 0.0,
                source="emitter" if x_value < 0.07337 else "capture",
            )

        result = integrate_spectral_pixel(
            edge,
            (6.0e14,),
            x_min=-0.2,
            x_max=0.2,
            y_min=-0.2,
            y_max=0.2,
            options=_options(maximum_depth=2),
        )
        self.assertFalse(result.converged)
        self.assertGreater(result.unresolved_solid_angle_sr, 0.0)
        self.assertGreater(result.estimated_absolute_errors_nu_sr[0], 0.0)

    def test_parent_guard_evidence_cannot_disappear_after_refinement(self) -> None:
        def narrow(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            inside = 0.49 < x_value < 0.51
            return _sample(
                1.0 if inside else 0.0,
                source="narrow-source" if inside else "sky",
            )

        result = integrate_spectral_pixel(
            narrow,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(maximum_depth=1),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.unresolved_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertGreater(result.estimated_absolute_errors_nu_sr[0], 0.0)

    def test_guard_only_g_and_direction_do_not_claim_zero_area_bounds(self) -> None:
        def guard_only(
            x_value: float,
            y_value: float,
            _frequencies: tuple[float, ...],
        ) -> SpectralRaySample:
            if (x_value, y_value) == (0.125, 0.125):
                return _sample(1.0, source="disk", shift=1.25)
            if (x_value, y_value) == (0.5, 0.125):
                return _sample(
                    1.0,
                    source="escaped-boundary",
                    direction=(1.0, 0.0, 0.0),
                )
            if (x_value, y_value) == (0.875, 0.125):
                return _sample(
                    1.0,
                    source="escaped-boundary",
                    direction=(0.0, 1.0, 0.0),
                )
            return _sample(0.0, source="captured-boundary")

        result = integrate_spectral_pixel(
            guard_only,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                maximum_depth=0,
                radiance_absolute_tolerances=(10.0,),
            ),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.unresolved_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertEqual(result.frequency_shift_solid_angle_sr, 0.0)
        self.assertIsNone(result.minimum_frequency_shift_g)
        self.assertIsNone(result.maximum_frequency_shift_g)
        self.assertEqual(result.escape_direction_solid_angle_sr, 0.0)
        self.assertEqual(result.maximum_escape_direction_span_rad, 0.0)

    def test_narrow_off_axis_solid_angle_avoids_corner_cancellation(self) -> None:
        half_width = 0.5e-7
        result = pinhole_solid_angle(
            1.3 - half_width,
            1.3 + half_width,
            -0.5 - half_width,
            -0.5 + half_width,
        )
        local_jacobian = (1.0 + 1.3**2 + 0.5**2) ** -1.5
        expected = local_jacobian * (2.0 * half_width) ** 2
        self.assertAlmostEqual(result / expected, 1.0, delta=2.0e-8)

    def test_sampling_and_per_ray_errors_are_added(self) -> None:
        def uncertain(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            sample = _sample(x_value * x_value)
            return SpectralRaySample(
                specific_intensities_nu=sample.specific_intensities_nu,
                absolute_errors_nu=(0.5,),
                visible_source=sample.visible_source,
                topology_signature=sample.topology_signature,
                ray_converged=True,
                convergence_audit=RayConvergenceAudit(
                    accepted_steps=1,
                    ray_gate_passed=True,
                    source_gate_passed=True,
                    transfer_gate_passed=True,
                ),
            )

        result = integrate_spectral_pixel(
            uncertain,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                maximum_depth=0,
                radiance_absolute_tolerances=(10.0,),
            ),
        )
        solid_angle = result.pixel_solid_angle_sr
        sampled_spread = solid_angle * (0.875**2 - 0.125**2)
        self.assertGreaterEqual(
            result.estimated_absolute_errors_nu_sr[0],
            sampled_spread + 0.5 * solid_angle,
        )

    def test_global_g_and_direction_variation_survive_leaf_merge(self) -> None:
        def split(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            left = x_value < 0.0
            return _sample(
                1.0,
                source="diagnostic-ray",
                topology="same-visible-disk",
                shift=1.0 if left else 2.0,
                direction=(1.0, 0.0, 0.0) if left else (-1.0, 0.0, 0.0),
            )

        result = integrate_spectral_pixel(
            split,
            (1.0,),
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            options=_options(
                maximum_depth=1,
                weighted_log_g_tolerance=0.01,
                weighted_direction_tolerance_rad=0.01,
            ),
        )
        self.assertFalse(result.converged)
        self.assertAlmostEqual(result.weighted_log_g_variation, math.log(2.0))
        self.assertAlmostEqual(result.maximum_escape_direction_span_rad, math.pi)
        self.assertAlmostEqual(
            result.weighted_escape_direction_variation_rad,
            math.pi,
        )

    def test_smooth_direction_field_survives_required_refinement(self) -> None:
        def smooth(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            angle = 0.8 * x_value
            return _sample(
                1.0,
                source="escaped-boundary",
                direction=(math.cos(angle), math.sin(angle), 0.0),
            )

        result = integrate_spectral_pixel(
            smooth,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                minimum_depth=1,
                maximum_depth=1,
                weighted_direction_tolerance_rad=10.0,
            ),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.unresolved_solid_angle_sr, 0.0)
        self.assertEqual(
            result.escape_direction_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertGreater(result.maximum_escape_direction_span_rad, 0.6)

    def test_resolved_mixed_topology_preserves_partial_direction_coverage(self) -> None:
        def split(
            x_value: float,
            y_value: float,
            _frequencies: tuple[float, ...],
        ):
            if x_value < 0.5:
                return _sample(0.0, source="captured-boundary")
            angle = 0.4 * y_value
            return _sample(
                1.0,
                source="escaped-boundary",
                direction=(math.cos(angle), math.sin(angle), 0.0),
            )

        result = integrate_spectral_pixel(
            split,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                maximum_depth=1,
                weighted_direction_tolerance_rad=10.0,
            ),
        )
        source_areas = dict(result.source_solid_angles_sr)
        self.assertTrue(result.converged)
        self.assertEqual(result.unresolved_solid_angle_sr, 0.0)
        self.assertAlmostEqual(
            result.escape_direction_solid_angle_sr,
            source_areas["escaped-boundary"],
        )
        self.assertLess(
            result.escape_direction_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )

    def test_parent_only_g_and_direction_evidence_becomes_unresolved(self) -> None:
        def parent_only(
            x_value: float,
            y_value: float,
            _frequencies: tuple[float, ...],
        ):
            if x_value == 0.5:
                lower = y_value < 0.5
                return _sample(
                    1.0,
                    shift=1.0 if lower else 2.0,
                    direction=(1.0, 0.0, 0.0) if lower else (-1.0, 0.0, 0.0),
                )
            return _sample(1.0)

        result = integrate_spectral_pixel(
            parent_only,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                maximum_depth=1,
                weighted_log_g_tolerance=0.01,
                weighted_direction_tolerance_rad=0.01,
            ),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.unresolved_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertEqual(
            result.frequency_shift_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertEqual(
            result.escape_direction_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )

    def test_partially_lost_parent_g_and_direction_evidence_is_unresolved(self) -> None:
        def partial(
            x_value: float,
            _y_value: float,
            _frequencies: tuple[float, ...],
        ):
            if x_value == 0.5:
                return _sample(
                    1.0,
                    shift=100.0,
                    direction=(1.0, 0.0, 0.0),
                )
            if x_value < 0.2:
                return _sample(
                    1.0,
                    shift=1.0,
                    direction=(0.0, 1.0, 0.0),
                )
            return _sample(1.0)

        result = integrate_spectral_pixel(
            partial,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                minimum_depth=1,
                maximum_depth=1,
                weighted_log_g_tolerance=10.0,
                weighted_direction_tolerance_rad=10.0,
            ),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.unresolved_solid_angle_sr,
            result.pixel_solid_angle_sr,
        )
        self.assertEqual(result.minimum_frequency_shift_g, 1.0)
        self.assertEqual(result.maximum_frequency_shift_g, 100.0)

    def test_subnormal_pixel_solid_angle_fails_closed(self) -> None:
        with self.assertRaisesRegex(AdaptiveSamplingError, "subnormal"):
            integrate_spectral_pixel(
                lambda _x, _y, _frequencies: _sample(1.0),
                (1.0,),
                x_min=0.0,
                x_max=8.0e-162,
                y_min=0.0,
                y_max=8.0e-162,
                options=_options(),
            )

    def test_extreme_finite_frequency_shifts_use_log_difference(self) -> None:
        minimum_shift = math.ulp(0.0)

        def extreme(x_value: float, _y: float, _frequencies: tuple[float, ...]):
            return _sample(
                1.0,
                source="diagnostic-ray",
                shift=minimum_shift if x_value < 0.5 else 1.0e308,
            )

        result = integrate_spectral_pixel(
            extreme,
            (1.0,),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            options=_options(
                maximum_depth=0,
                weighted_log_g_tolerance=2_000.0,
            ),
        )
        self.assertTrue(math.isfinite(result.weighted_log_g_variation))
        self.assertGreater(result.weighted_log_g_variation, 1_400.0)

    def test_unconverged_ray_and_radiance_ceiling_fail_closed(self) -> None:
        with self.assertRaises(AdaptiveSamplingError):
            integrate_spectral_pixel(
                lambda _x, _y, _frequencies: _sample(1.0, converged=False),
                (1.0,),
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=1.0,
                options=_options(),
            )
        with self.assertRaises(AdaptiveSamplingError):
            integrate_spectral_pixel(
                lambda _x, _y, _frequencies: _sample(11.0),
                (1.0,),
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=1.0,
                options=_options(),
            )

    def test_budget_exhaustion_is_not_an_invalid_black_pixel(self) -> None:
        with self.assertRaises(AdaptiveSamplingError):
            integrate_spectral_pixel(
                lambda x_value, _y, _frequencies: _sample(x_value),
                (1.0,),
                x_min=0.0,
                x_max=1.0,
                y_min=0.0,
                y_max=1.0,
                options=_options(
                    maximum_depth=3,
                    maximum_ray_evaluations=5,
                    radiance_relative_tolerance=0.0,
                ),
            )

    def test_shift_direction_and_source_diagnostics_are_retained(self) -> None:
        direction = (0.0, 0.0, 1.0)
        result = integrate_spectral_pixel(
            lambda x_value, _y, _frequencies: _sample(
                1.0,
                source="diagnostic-ray",
                shift=1.0 + 0.01 * x_value,
                direction=direction,
            ),
            (1.0,),
            x_min=0.0,
            x_max=0.1,
            y_min=0.0,
            y_max=0.1,
            options=_options(),
        )
        self.assertEqual(result.source_solid_angles_sr[0][0], "diagnostic-ray")
        self.assertLess(
            result.minimum_frequency_shift_g,
            result.maximum_frequency_shift_g,
        )
        self.assertEqual(result.maximum_escape_direction_span_rad, 0.0)
        self.assertAlmostEqual(
            math.fsum(area for _source, area in result.source_solid_angles_sr),
            result.pixel_solid_angle_sr,
            places=14,
        )

    def test_tiny_pixel_source_coverage_cannot_be_omitted(self) -> None:
        result = integrate_spectral_pixel(
            lambda _x, _y, _frequencies: _sample(1.0),
            (1.0,),
            x_min=0.0,
            x_max=1.0e-8,
            y_min=0.0,
            y_max=1.0e-8,
            options=_options(),
        )
        self.assertLess(result.pixel_solid_angle_sr, 2.0e-16)
        with self.assertRaisesRegex(ValueError, "do not cover"):
            replace(result, source_solid_angles_sr=())

    def test_exact_pinhole_solid_angle_is_additive(self) -> None:
        whole = pinhole_solid_angle(-0.8, 1.3, -0.6, 0.9)
        pieces = math.fsum(
            pinhole_solid_angle(x0, x1, y0, y1)
            for x0, x1 in ((-0.8, 0.2), (0.2, 1.3))
            for y0, y1 in ((-0.6, 0.1), (0.1, 0.9))
        )
        self.assertAlmostEqual(whole, pieces, places=14)

    def test_strict_validation(self) -> None:
        self.assertRaises(ValueError, AdaptivePixelOptions, maximum_depth=-1)
        self.assertRaises(
            ValueError,
            AdaptivePixelOptions,
            radiance_absolute_tolerances=(0.0,),
            radiance_guard_ceilings=(0.0,),
        )
        self.assertRaises(
            ValueError,
            SpectralRaySample,
            (1.0,),
            (0.0, 0.0),
            "sky",
            "sky",
        )


if __name__ == "__main__":
    unittest.main()
