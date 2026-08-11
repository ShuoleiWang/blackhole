from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest.mock import Mock, call, patch

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    HamiltonianState,
    RayPathSegment,
    RayTraceOptions,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventError,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_bl_zamo_tetrad,
    kerr_oblate_event_to_ks_cartesian,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_transfer import (
    KERR_DISK_TRANSFER_SCIENTIFIC_STATUS,
    KerrDiskTransferError,
    transfer_kerr_disk_spectrum,
)


SOLAR_MASS_KG = 1.98847e30


class OfflineKerrDiskTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        self.disk = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
        )

    def synthetic_ray(self, outcome: str = "escaped") -> RayTraceResult:
        start = HamiltonianState(
            event=(0.0, 30.0, 0.0, 5.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        midpoint = HamiltonianState(
            event=(-0.5, 20.0, 0.0, 2.5),
            covector=start.covector,
        )
        end = HamiltonianState(
            event=(-1.0, 10.0, 0.0, 1.0),
            covector=start.covector,
        )
        segment = RayPathSegment(
            start=start,
            midpoint=midpoint,
            end=end,
            affine_length=1.0,
            midpoint_null_residual=0.0,
        )
        return RayTraceResult(
            outcome=outcome,
            terminal_state=end,
            affine_length=1.0,
            accepted_steps=1,
            rejected_steps=0,
            maximum_null_residual=0.0,
            maximum_metric_interpolation_error=0.0,
            segments=(segment,),
            terminal_target_id=f"test-{outcome}-boundary",
        )

    def crossing(
        self,
        radius_m: float,
        *,
        ray_affine_length: float,
        coordinate_time_m: float = -0.5,
        phi_ks_rad: float = 0.0,
        bracket_affine_width: float = 0.0,
    ) -> RecordedSurfaceCrossing:
        event = kerr_oblate_event_to_ks_cartesian(
            coordinate_time_m=coordinate_time_m,
            radius_m=radius_m,
            theta_rad=0.5 * math.pi,
            phi_ks_rad=phi_ks_rad,
            spin_a_m=self.metric.spin_a_m,
        )
        state = HamiltonianState(
            event=event,
            covector=self.synthetic_ray().segments[0].start.covector,
        )
        return RecordedSurfaceCrossing(
            state=state,
            ray_affine_length=ray_affine_length,
            segment_index=0,
            segment_affine_length=ray_affine_length,
            orientation=-1,
            surface_value=0.0,
            bracket_affine_width=bracket_affine_width,
            iterations=0,
        )

    def test_first_valid_crossing_is_opaque_and_owns_emitter_event(self) -> None:
        first = self.crossing(
            8.0,
            ray_affine_length=0.25,
            coordinate_time_m=-13.5,
            phi_ks_rad=0.61,
            bracket_affine_width=2.0e-9,
        )
        second = self.crossing(
            12.0,
            ray_affine_length=0.75,
            coordinate_time_m=-28.0,
            phi_ks_rad=-0.8,
            bracket_affine_width=3.0e-9,
        )
        background = Mock(return_value=99.0)
        with (
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                return_value=(first, second),
            ) as locator,
            patch(
                "offline.kerr_disk_transfer.observer_to_emitter_frequency_shift_g",
                return_value=1.25,
            ) as frequency_shift,
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                self.synthetic_ray(),
                (1.0, 0.0, 0.0, 0.0),
                (3.0e14, 6.0e14),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=background,
            )

        self.assertEqual(result.source_kind, "disk")
        self.assertIs(result.crossing, first)
        self.assertEqual(result.disk_radius_m, 8.0)
        self.assertIsNotNone(result.emitter)
        self.assertAlmostEqual(result.emitter.event[0], first.state.event[0])
        self.assertAlmostEqual(result.emitter.phi_ks_rad, 0.61, places=14)
        self.assertEqual(result.ray_boundary_outcome, "escaped")
        self.assertEqual(result.ray_boundary_target_id, "test-escaped-boundary")
        self.assertEqual(
            tuple(
                (entry.orientation, entry.radial_region)
                for entry in result.crossing_signature
            ),
            ((-1, "opaque-annulus"), (-1, "opaque-annulus")),
        )
        self.assertEqual(result.first_opaque_crossing_index, 0)
        self.assertEqual(
            result.crossing_bracket_affine_widths,
            (2.0e-9, 3.0e-9),
        )
        frequency_shift.assert_called_once()
        locator.assert_called_once()
        self.assertIs(frequency_shift.call_args.args[3], first.state)
        self.assertIs(frequency_shift.call_args.args[4], result.emitter)
        background.assert_not_called()

    def test_out_of_order_locator_results_fail_before_first_hit_selection(self) -> None:
        farther = self.crossing(8.0, ray_affine_length=0.75)
        nearer = self.crossing(9.0, ray_affine_length=0.25)
        background = Mock(return_value=99.0)
        frequency_shift = Mock(return_value=1.0)
        with (
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                return_value=(farther, nearer),
            ),
            patch(
                "offline.kerr_disk_transfer.observer_to_emitter_frequency_shift_g",
                frequency_shift,
            ),
        ):
            with self.assertRaisesRegex(
                KerrDiskTransferError,
                "not ordered",
            ):
                transfer_kerr_disk_spectrum(
                    self.disk,
                    self.synthetic_ray(),
                    (1.0, 0.0, 0.0, 0.0),
                    (5.0e14,),
                    outer_radius_m=20.0,
                    escaped_observer_specific_intensity_nu=background,
                )
        frequency_shift.assert_not_called()
        background.assert_not_called()

    def test_inside_isco_is_transparent_and_reveals_later_disk_hit(self) -> None:
        gap = self.crossing(
            0.95 * self.disk.isco_radius_m,
            ray_affine_length=0.2,
        )
        disk_hit = self.crossing(8.0, ray_affine_length=0.7)
        background = Mock(return_value=99.0)
        with (
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                return_value=(gap, disk_hit),
            ),
            patch(
                "offline.kerr_disk_transfer.observer_to_emitter_frequency_shift_g",
                return_value=1.0,
            ),
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                self.synthetic_ray(),
                (1.0, 0.0, 0.0, 0.0),
                (5.0e14,),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=background,
            )

        self.assertIs(result.crossing, disk_hit)
        self.assertAlmostEqual(result.disk_radius_m, 8.0, places=14)
        self.assertEqual(
            tuple(entry.radial_region for entry in result.crossing_signature),
            ("inside-isco", "opaque-annulus"),
        )
        self.assertEqual(result.first_opaque_crossing_index, 1)
        background.assert_not_called()

    def test_liouville_g_cubed_and_emitter_frequency_oracle(self) -> None:
        crossing = self.crossing(8.0, ray_affine_length=0.5)
        with (
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                return_value=(crossing,),
            ),
            patch(
                "offline.kerr_disk_transfer.observer_to_emitter_frequency_shift_g",
                return_value=2.0,
            ),
            patch.object(
                StationaryNovikovThorneDisk,
                "emitted_specific_intensity_nu",
                side_effect=(3.0, 5.0),
            ) as emitted_intensity,
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                self.synthetic_ray(),
                (1.0, 0.0, 0.0, 0.0),
                (8.0e14, 12.0e14),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=(
                    lambda _state, _frequency, _target: 0.0
                ),
            )

        self.assertEqual(result.frequency_shift_g, 2.0)
        self.assertEqual(result.emitted_frequencies_hz, (4.0e14, 6.0e14))
        self.assertEqual(result.emitted_specific_intensities_nu, (3.0, 5.0))
        self.assertEqual(result.observed_specific_intensities_nu, (24.0, 40.0))
        self.assertEqual(len(emitted_intensity.call_args_list), 2)
        for actual, expected_frequency in zip(
            emitted_intensity.call_args_list,
            (4.0e14, 6.0e14),
        ):
            self.assertAlmostEqual(actual.args[0], 8.0, places=14)
            self.assertEqual(actual.args[1], expected_frequency)

    def test_declared_flux_normalized_angular_law_scales_local_intensity(
        self,
    ) -> None:
        crossing = self.crossing(8.0, ray_affine_length=0.5)
        law = FluxConservingLinearLimbDarkening()
        multiplier = law.intensity_multiplier(0.8)
        with (
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                return_value=(crossing,),
            ),
            patch(
                "offline.kerr_disk_transfer.observer_to_emitter_frequency_shift_g",
                return_value=2.0,
            ),
            patch(
                "offline.kerr_disk_transfer.equatorial_emission_angle_cosine",
                return_value=0.8,
            ) as emission_angle,
            patch.object(
                StationaryNovikovThorneDisk,
                "emitted_specific_intensity_nu",
                return_value=3.0,
            ),
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                self.synthetic_ray(),
                (1.0, 0.0, 0.0, 0.0),
                (8.0e14,),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=(
                    lambda _state, _frequency, _target: 0.0
                ),
                angular_emission_law=law,
            )

        self.assertEqual(result.emission_angle_cosine, 0.8)
        self.assertEqual(result.angular_emission_multiplier, multiplier)
        self.assertEqual(
            result.emitted_specific_intensities_nu,
            (3.0 * multiplier,),
        )
        self.assertEqual(
            result.observed_specific_intensities_nu,
            (8.0 * 3.0 * multiplier,),
        )
        emission_angle.assert_called_once()

    def test_no_valid_hit_uses_escaped_background(self) -> None:
        outside = self.crossing(25.0, ray_affine_length=0.5)
        background = Mock(
            side_effect=lambda _state, frequency, _target: frequency / 1e14
        )
        ray = self.synthetic_ray()
        with patch(
            "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
            return_value=(outside,),
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                ray,
                (1.0, 0.0, 0.0, 0.0),
                (1.0e14, 2.0e14),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=background,
            )

        self.assertEqual(result.source_kind, "escaped-boundary")
        self.assertEqual(result.observed_specific_intensities_nu, (1.0, 2.0))
        self.assertIsNone(result.crossing)
        self.assertEqual(
            tuple(entry.radial_region for entry in result.crossing_signature),
            ("outside-outer-radius",),
        )
        self.assertIsNone(result.first_opaque_crossing_index)
        self.assertEqual(
            background.call_args_list,
            [
                call(ray.terminal_state, 1.0e14, "test-escaped-boundary"),
                call(ray.terminal_state, 2.0e14, "test-escaped-boundary"),
            ],
        )

    def test_no_valid_hit_forces_captured_boundary_black(self) -> None:
        background = Mock(return_value=99.0)
        with patch(
            "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
            return_value=(),
        ):
            result = transfer_kerr_disk_spectrum(
                self.disk,
                self.synthetic_ray("captured"),
                (1.0, 0.0, 0.0, 0.0),
                (1.0e14, 2.0e14),
                outer_radius_m=20.0,
                escaped_observer_specific_intensity_nu=background,
            )

        self.assertEqual(result.source_kind, "captured-boundary")
        self.assertEqual(result.observed_specific_intensities_nu, (0.0, 0.0))
        background.assert_not_called()

    def test_bad_or_incomplete_rays_fail_before_surface_location(self) -> None:
        ray = self.synthetic_ray()
        malformed = (
            replace(
                ray,
                outcome="unresolved",
                failure_reason="affine budget exhausted",
                terminal_target_id=None,
            ),
            replace(ray, failure_reason="injected failure"),
            replace(ray, terminal_target_id=None),
            replace(ray, segments=()),
            replace(ray, accepted_steps=2),
            replace(ray, affine_length=2.0),
            replace(
                ray,
                terminal_state=HamiltonianState(
                    event=(-2.0, 1.0, 0.0, 0.0),
                    covector=ray.terminal_state.covector,
                ),
            ),
        )
        locator = Mock(return_value=())
        with patch(
            "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
            locator,
        ):
            for invalid in malformed:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        transfer_kerr_disk_spectrum(
                            self.disk,
                            invalid,
                            (1.0, 0.0, 0.0, 0.0),
                            (1.0e14,),
                            outer_radius_m=20.0,
                            escaped_observer_specific_intensity_nu=(
                                lambda _state, _frequency, _target: 0.0
                            ),
                        )
        locator.assert_not_called()

    def test_surface_locator_failure_propagates_fail_closed(self) -> None:
        failure = SurfaceEventError("injected surface failure")
        background = Mock(return_value=1.0)
        with patch(
            "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
            side_effect=failure,
        ):
            with self.assertRaises(SurfaceEventError) as raised:
                transfer_kerr_disk_spectrum(
                    self.disk,
                    self.synthetic_ray(),
                    (1.0, 0.0, 0.0, 0.0),
                    (1.0e14,),
                    outer_radius_m=20.0,
                    escaped_observer_specific_intensity_nu=background,
                )
        self.assertIs(raised.exception, failure)
        background.assert_not_called()

    def test_real_kerr_camera_rays_show_approaching_receding_shift(self) -> None:
        termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
        )
        observer = kerr_bl_zamo_tetrad(
            self.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
        )

        results = []
        for screen_x in (-0.5, 0.5):
            initial = kerr_zamo_camera_ray(
                self.metric,
                observer_radius_m=30.0,
                theta_rad=1.1,
                screen_x=screen_x,
                screen_y=-0.5,
            )
            ray = trace_null_geodesic(
                self.metric,
                initial,
                termination=termination,
                options=ray_options,
            )
            self.assertEqual(ray.outcome, "escaped", ray.failure_reason)
            results.append(
                transfer_kerr_disk_spectrum(
                    self.disk,
                    ray,
                    observer.four_velocity,
                    (5.0e14,),
                    outer_radius_m=25.0,
                    escaped_observer_specific_intensity_nu=(
                        lambda _state, _frequency, _target: 0.0
                    ),
                    surface_options=surface_options,
                    frequency_null_residual_limit=2.0e-7,
                )
            )

        receding, approaching = results
        self.assertEqual(receding.source_kind, "disk")
        self.assertEqual(approaching.source_kind, "disk")
        self.assertLess(receding.frequency_shift_g, 1.0)
        self.assertGreater(approaching.frequency_shift_g, 1.0)
        self.assertGreater(
            approaching.frequency_shift_g,
            receding.frequency_shift_g,
        )
        self.assertGreater(receding.observed_specific_intensities_nu[0], 0.0)
        self.assertGreater(approaching.observed_specific_intensities_nu[0], 0.0)

    def test_scientific_status_forbids_unsupported_claims(self) -> None:
        self.assertEqual(
            KERR_DISK_TRANSFER_SCIENTIFIC_STATUS["observable"],
            "scalar observer-frame specific intensity I_nu",
        )
        self.assertFalse(
            KERR_DISK_TRANSFER_SCIENTIFIC_STATUS["includesPolarization"]
        )
        self.assertFalse(
            KERR_DISK_TRANSFER_SCIENTIFIC_STATUS["includesReturningRadiation"]
        )
        self.assertIn(
            "already-observer-frame",
            KERR_DISK_TRANSFER_SCIENTIFIC_STATUS["escapedBoundary"],
        )


if __name__ == "__main__":
    unittest.main()
