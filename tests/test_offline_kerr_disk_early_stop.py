from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from offline.disk_atmosphere import (
    FluxConservingLinearLimbDarkening,
    equatorial_emission_angle_cosine,
)
from offline.geodesic import (
    RayTraceOptions,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_bl_zamo_tetrad,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import (
    StationaryNovikovThorneDisk,
    observer_to_emitter_frequency_shift_g,
)
from offline.kerr_disk_early_stop import (
    KERR_DISK_OPAQUE_HIT_OUTCOME,
    KerrDiskAnnulusSurface,
    transfer_early_stopped_kerr_disk_spectrum,
)
from offline.kerr_disk_frame import (
    KerrDiskRaySampler,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_disk_transfer import transfer_kerr_disk_spectrum
from offline.novikov_thorne import RETROGRADE


SOLAR_MASS_KG = 1.98847e30


class OfflineKerrDiskEarlyStopTests(unittest.TestCase):
    def make_components(self):
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
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
        surface = KerrDiskAnnulusSurface(disk, 25.0)
        observer = kerr_bl_zamo_tetrad(
            metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
        )
        background = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=1.0,
            reference_frequency_hz=1.0e14,
            spectral_index=1.0,
        )
        return (
            metric,
            termination,
            disk,
            ray_options,
            surface_options,
            surface,
            observer,
            background,
        )

    @staticmethod
    def camera(metric, screen_x: float, screen_y: float):
        return kerr_zamo_camera_ray(
            metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
            screen_x=screen_x,
            screen_y=screen_y,
        )

    def test_real_disk_hit_matches_full_path_oracle_and_removes_hidden_steps(
        self,
    ) -> None:
        (
            metric,
            termination,
            disk,
            ray_options,
            surface_options,
            surface,
            observer,
            background,
        ) = self.make_components()
        initial = self.camera(metric, 0.5, -0.5)
        full = trace_null_geodesic(
            metric,
            initial,
            termination=termination,
            options=ray_options,
        )
        early = trace_null_geodesic(
            metric,
            initial,
            termination=termination,
            interior_surface=surface,
            surface_options=surface_options,
            options=ray_options,
        )

        self.assertEqual(full.outcome, "escaped", full.failure_reason)
        self.assertEqual(early.outcome, KERR_DISK_OPAQUE_HIT_OUTCOME)
        self.assertLessEqual(early.accepted_steps, math.ceil(0.35 * full.accepted_steps))
        self.assertLess(early.affine_length, full.affine_length)
        self.assertEqual(early.segments[-1].end, early.terminal_state)
        self.assertIsNotNone(early.interior_surface_trace)
        assert early.interior_surface_trace is not None
        self.assertTrue(early.interior_surface_trace.topology_converged)
        self.assertEqual(
            early.interior_surface_trace.verification_subdivisions_per_step,
            2 * early.interior_surface_trace.base_subdivisions_per_step,
        )

        keywords = {
            "escaped_observer_specific_intensity_nu": background,
            "surface_options": surface_options,
            "frequency_null_residual_limit": 2.0e-7,
            "conserved_quantity_tolerance": 1.0e-7,
            "emitter_event_tolerance_m": 1.0e-8 * metric.mass_m,
            "angular_emission_law": FluxConservingLinearLimbDarkening(),
        }
        legacy = transfer_kerr_disk_spectrum(
            disk,
            full,
            observer.four_velocity,
            (5.0e14,),
            outer_radius_m=surface.outer_radius_m,
            **keywords,
        )
        with patch(
            "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
            side_effect=AssertionError("early transfer must not run the path locator"),
        ) as locator:
            visible = transfer_early_stopped_kerr_disk_spectrum(
                surface,
                early,
                observer.four_velocity,
                (5.0e14,),
                **keywords,
            )
        locator.assert_not_called()
        self.assertTrue(visible.terminated_at_opaque_disk)
        self.assertIsNone(visible.ray_boundary_outcome)
        self.assertIsNone(visible.ray_boundary_target_id)
        self.assertAlmostEqual(visible.disk_radius_m, legacy.disk_radius_m, delta=1e-11)
        self.assertAlmostEqual(
            visible.frequency_shift_g,
            legacy.frequency_shift_g,
            delta=1e-13,
        )
        self.assertAlmostEqual(
            visible.observed_specific_intensities_nu[0],
            legacy.observed_specific_intensities_nu[0],
            delta=1e-18,
        )
        self.assertEqual(
            visible.isotropic_emitted_specific_intensities_nu,
            legacy.isotropic_emitted_specific_intensities_nu,
        )

    def test_transparent_isco_and_outer_crossings_continue_to_true_boundary(
        self,
    ) -> None:
        (
            metric,
            termination,
            _disk,
            ray_options,
            surface_options,
            surface,
            observer,
            background,
        ) = self.make_components()
        cases = (
            ((0.1, 0.0), "captured", "inside-isco"),
            ((0.0, 0.5), "escaped", "outside-outer-radius"),
        )
        for (screen_x, screen_y), outcome, region in cases:
            with self.subTest(outcome=outcome, region=region):
                ray = trace_null_geodesic(
                    metric,
                    self.camera(metric, screen_x, screen_y),
                    termination=termination,
                    interior_surface=surface,
                    surface_options=surface_options,
                    options=ray_options,
                )
                self.assertEqual(ray.outcome, outcome, ray.failure_reason)
                result = transfer_early_stopped_kerr_disk_spectrum(
                    surface,
                    ray,
                    observer.four_velocity,
                    (1.0e14,),
                    escaped_observer_specific_intensity_nu=background,
                    surface_options=surface_options,
                    angular_emission_law=FluxConservingLinearLimbDarkening(),
                )
                self.assertFalse(result.terminated_at_opaque_disk)
                self.assertEqual(result.ray_boundary_outcome, outcome)
                self.assertEqual(result.crossing_signature[-1].radial_region, region)
                if outcome == "captured":
                    self.assertEqual(result.observed_specific_intensities_nu, (0.0,))
                else:
                    self.assertGreater(result.observed_specific_intensities_nu[0], 0.0)

    def test_surface_probe_resolution_is_bound_to_transfer_options(self) -> None:
        (
            metric,
            termination,
            _disk,
            ray_options,
            surface_options,
            surface,
            observer,
            background,
        ) = self.make_components()
        ray = trace_null_geodesic(
            metric,
            self.camera(metric, 0.5, -0.5),
            termination=termination,
            interior_surface=surface,
            surface_options=surface_options,
            options=ray_options,
        )
        with self.assertRaisesRegex(ValueError, "probe resolution"):
            transfer_early_stopped_kerr_disk_spectrum(
                surface,
                ray,
                observer.four_velocity,
                (5.0e14,),
                escaped_observer_specific_intensity_nu=background,
                surface_options=replace(
                    surface_options,
                    subdivisions_per_segment=8,
                ),
                angular_emission_law=FluxConservingLinearLimbDarkening(),
            )
        self.assertIsNotNone(ray.interior_surface_trace)
        assert ray.interior_surface_trace is not None
        entries = ray.interior_surface_trace.crossings
        forged_entry = replace(
            entries[-1],
            crossing=replace(entries[-1].crossing, segment_index=999),
        )
        forged_ray = replace(
            ray,
            interior_surface_trace=replace(
                ray.interior_surface_trace,
                crossings=(*entries[:-1], forged_entry),
            ),
        )
        with self.assertRaisesRegex(ValueError, "crossing diagnostics"):
            transfer_early_stopped_kerr_disk_spectrum(
                surface,
                forged_ray,
                observer.four_velocity,
                (5.0e14,),
                escaped_observer_specific_intensity_nu=background,
                surface_options=surface_options,
                angular_emission_law=FluxConservingLinearLimbDarkening(),
            )

    def test_frame_uses_early_transfer_for_high_polar_and_edge_rays(self) -> None:
        (
            metric,
            termination,
            disk,
            ray_options,
            surface_options,
            _surface,
            _observer,
            background,
        ) = self.make_components()
        sampler = KerrDiskRaySampler(
            metric=metric,
            observer_radius_m=30.0,
            observer_theta_rad=1.1,
            termination=termination,
            disk=disk,
            outer_radius_m=25.0,
            escaped_observer_spectrum=background,
            fine_options=ray_options,
            surface_options=surface_options,
            frequency_null_residual_limit=2.0e-7,
        )
        with (
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=AssertionError("frame must not use full-path transfer"),
            ) as legacy_transfer,
            patch(
                "offline.kerr_disk_transfer.locate_recorded_surface_crossings",
                side_effect=AssertionError("frame must not run full-path locator"),
            ) as locator,
        ):
            high_polar = sampler.sample(0.0, -0.9, (5.0e14,))
            near_visible_edge = sampler.sample(0.15, 0.0, (5.0e14,))
        legacy_transfer.assert_not_called()
        locator.assert_not_called()
        self.assertEqual(high_polar.visible_source, "disk")
        self.assertEqual(near_visible_edge.visible_source, "disk")
        self.assertTrue(high_polar.ray_converged)
        self.assertTrue(near_visible_edge.ray_converged)
        self.assertGreater(high_polar.convergence_audit.accepted_steps, 0)
        self.assertGreater(near_visible_edge.convergence_audit.accepted_steps, 0)

    def test_frame_rejects_forged_early_crossing_and_emitter_metric(self) -> None:
        (
            metric,
            termination,
            disk,
            ray_options,
            surface_options,
            _surface,
            _observer,
            background,
        ) = self.make_components()
        sampler = KerrDiskRaySampler(
            metric=metric,
            observer_radius_m=30.0,
            observer_theta_rad=1.1,
            termination=termination,
            disk=disk,
            outer_radius_m=25.0,
            escaped_observer_spectrum=background,
            fine_options=ray_options,
            surface_options=surface_options,
            frequency_null_residual_limit=2.0e-7,
        )
        real_transfer = transfer_early_stopped_kerr_disk_spectrum

        def copied_crossing(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            self.assertIsNotNone(result.crossing)
            return replace(result, crossing=replace(result.crossing))

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=copied_crossing,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "terminal opaque surface entry",
            ):
                sampler.sample(0.5, -0.5, (5.0e14,))

        foreign_metric = KerrKerrSchildMetric(spin_a_m=0.2)
        foreign_disk = StationaryNovikovThorneDisk(
            metric=foreign_metric,
            black_hole_mass_kg=disk.black_hole_mass_kg,
            mass_accretion_rate_kg_s=disk.mass_accretion_rate_kg_s,
        )

        def foreign_emitter(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            self.assertIsNotNone(result.crossing)
            self.assertIsNotNone(result.emitter)
            emitter = foreign_disk.emitter(
                result.disk_radius_m,
                phi_ks_rad=result.emitter.phi_ks_rad,
                coordinate_time_m=result.emitter.event[0],
            )
            crossing = replace(
                result.crossing,
                state=replace(result.crossing.state, event=emitter.event),
            )
            return replace(result, crossing=crossing, emitter=emitter)

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=foreign_emitter,
        ):
            with self.assertRaisesRegex(RuntimeError, "different Kerr metric"):
                sampler.sample(0.5, -0.5, (5.0e14,))

    def test_frame_recomputes_trace_transfer_and_boundary_physics(self) -> None:
        (
            metric,
            termination,
            disk,
            ray_options,
            surface_options,
            _surface,
            _observer,
            background,
        ) = self.make_components()
        sampler = KerrDiskRaySampler(
            metric=metric,
            observer_radius_m=30.0,
            observer_theta_rad=1.1,
            termination=termination,
            disk=disk,
            outer_radius_m=25.0,
            escaped_observer_spectrum=background,
            fine_options=ray_options,
            surface_options=surface_options,
            frequency_null_residual_limit=2.0e-7,
        )
        real_transfer = transfer_early_stopped_kerr_disk_spectrum

        def forged_boundary_topology(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            if result.source_kind != "escaped-boundary" or not result.crossing_signature:
                return result
            signature = result.crossing_signature
            forged = replace(
                signature[-1],
                orientation=-signature[-1].orientation,
            )
            return replace(
                result,
                crossing_signature=(*signature[:-1], forged),
                crossing_bracket_affine_widths=(
                    *result.crossing_bracket_affine_widths[:-1],
                    0.0,
                ),
            )

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=forged_boundary_topology,
        ):
            with self.assertRaisesRegex(RuntimeError, "crossing evidence"):
                sampler.sample(0.0, 0.5, (5.0e14,))

        def forged_g_and_angle(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            if result.source_kind != "disk":
                return result
            shift = 2.0 * result.frequency_shift_g
            angle = 0.2
            multiplier = sampler.angular_emission_law.intensity_multiplier(angle)
            emitted_frequencies = tuple(
                frequency / shift for frequency in result.observer_frequencies_hz
            )
            isotropic = tuple(
                disk.emitted_specific_intensity_nu(result.disk_radius_m, frequency)
                for frequency in emitted_frequencies
            )
            emitted = tuple(value * multiplier for value in isotropic)
            observed = tuple(shift**3 * value for value in emitted)
            return replace(
                result,
                frequency_shift_g=shift,
                emission_angle_cosine=angle,
                angular_emission_multiplier=multiplier,
                emitted_frequencies_hz=emitted_frequencies,
                isotropic_emitted_specific_intensities_nu=isotropic,
                emitted_specific_intensities_nu=emitted,
                observed_specific_intensities_nu=observed,
            )

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=forged_g_and_angle,
        ):
            with self.assertRaisesRegex(RuntimeError, "frequency shift"):
                sampler.sample(0.5, -0.5, (5.0e14,))

        retrograde_disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=disk.black_hole_mass_kg,
            mass_accretion_rate_kg_s=disk.mass_accretion_rate_kg_s,
            orientation=RETROGRADE,
        )

        def forged_retrograde_emitter(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            if result.source_kind != "disk":
                return result
            ray = args[1]
            emitter = retrograde_disk.emitter(
                result.disk_radius_m,
                phi_ks_rad=result.emitter.phi_ks_rad,
                coordinate_time_m=result.emitter.event[0],
            )
            shift = observer_to_emitter_frequency_shift_g(
                metric,
                ray.segments[0].start,
                sampler._observer_tetrad.four_velocity,
                result.crossing.state,
                emitter,
                null_residual_limit=sampler.frequency_null_residual_limit,
                conserved_quantity_tolerance=sampler.conserved_quantity_tolerance,
                emitter_event_tolerance_m=(
                    sampler._resolved_emitter_event_tolerance_m
                ),
            )
            angle = equatorial_emission_angle_cosine(
                metric,
                result.crossing.state,
                emitter,
                null_residual_limit=sampler.frequency_null_residual_limit,
                emitter_event_tolerance_m=(
                    sampler._resolved_emitter_event_tolerance_m
                ),
            )
            multiplier = sampler.angular_emission_law.intensity_multiplier(angle)
            emitted_frequencies = tuple(
                frequency / shift for frequency in result.observer_frequencies_hz
            )
            isotropic = tuple(
                disk.emitted_specific_intensity_nu(result.disk_radius_m, frequency)
                for frequency in emitted_frequencies
            )
            emitted = tuple(value * multiplier for value in isotropic)
            observed = tuple(shift**3 * value for value in emitted)
            return replace(
                result,
                emitter=emitter,
                frequency_shift_g=shift,
                emission_angle_cosine=angle,
                angular_emission_multiplier=multiplier,
                emitted_frequencies_hz=emitted_frequencies,
                isotropic_emitted_specific_intensities_nu=isotropic,
                emitted_specific_intensities_nu=emitted,
                observed_specific_intensities_nu=observed,
            )

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=forged_retrograde_emitter,
        ):
            with self.assertRaisesRegex(RuntimeError, "Novikov-Thorne disk"):
                sampler.sample(0.5, -0.5, (5.0e14,))

        def forged_escape(*args, **kwargs):
            result = real_transfer(*args, **kwargs)
            if result.source_kind != "escaped-boundary":
                return result
            return replace(
                result,
                observed_specific_intensities_nu=tuple(
                    42.0 for _frequency in result.observer_frequencies_hz
                ),
            )

        with patch(
            "offline.kerr_disk_frame.transfer_early_stopped_kerr_disk_spectrum",
            side_effect=forged_escape,
        ):
            with self.assertRaisesRegex(RuntimeError, "bound provider"):
                sampler.sample(0.0, 0.5, (5.0e14,))


if __name__ == "__main__":
    unittest.main()
