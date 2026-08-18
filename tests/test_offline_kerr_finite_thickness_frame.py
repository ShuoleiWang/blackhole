from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest
from unittest import mock

import offline.kerr_finite_thickness_frame as frame_module
import offline.kerr_finite_thickness_replay_certificate as replay_module
from offline.adaptive_frame import RayConvergenceAudit
from offline.geodesic import (
    RayTraceOptions,
    SurfaceEventOptions,
    trace_refined_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    KerrDiskRaySampler,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_frame import (
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    KerrFiniteThicknessFrameError,
    KerrFiniteThicknessRaySampler,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    UPPER_SURFACE_ID,
    KerrFiniteThicknessMultiSurface,
)


SOLAR_MASS_KG = 1.98847e30


class OfflineKerrFiniteThicknessFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.disk = StationaryNovikovThorneDisk(
            metric=cls.metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
        )
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
            subdivisions_per_segment=4,
        )
        cls.sampler = cls.make_sampler.__func__(  # type: ignore[attr-defined]
            cls,
            observer_theta_rad=1.1,
        )
        cls.cached_initial = kerr_zamo_camera_ray(
            cls.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        cls.cached_refinement = trace_refined_null_geodesic(
            cls.metric,
            cls.cached_initial,
            termination=cls.termination,
            multi_interior_surface=cls.surface,
            surface_options=cls.surface_options,
            fine_options=cls.ray_options,
            record_coarse_path=True,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
        )

    @classmethod
    def make_sampler(
        cls,
        *,
        observer_theta_rad: float,
        observer_radius_m: float = 30.0,
        surface: KerrFiniteThicknessMultiSurface | None = None,
        escaped=None,
        termination: KerrOblateTermination | None = None,
    ) -> KerrFiniteThicknessRaySampler:
        return KerrFiniteThicknessRaySampler(
            metric=cls.metric,
            observer_radius_m=observer_radius_m,
            termination=cls.termination if termination is None else termination,
            surface=cls.surface if surface is None else surface,
            disk=cls.disk,
            escaped_observer_spectrum=(
                DarkEscapedObserverSpectrum() if escaped is None else escaped
            ),
            fine_options=cls.ray_options,
            surface_options=cls.surface_options,
            observer_theta_rad=observer_theta_rad,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance_m=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
            specific_intensity_relative_tolerance=1.0e-3,
        )

    def test_descriptor_declares_model_and_limits_without_rate_invention(self) -> None:
        descriptor = self.sampler.descriptor()
        self.assertEqual(descriptor["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(
            descriptor["rayOptions"]["independentFineCoarseTraces"]
        )
        self.assertTrue(
            descriptor["finiteThicknessSurface"]
            ["heightRateIsIndependentOfThermalRate"]
        )
        self.assertEqual(
            descriptor["finiteThicknessSurface"]["surfaceIds"],
            sorted((LOWER_SURFACE_ID, UPPER_SURFACE_ID)),
        )
        self.assertEqual(
            descriptor["frequencyTransfer"]["boundaryValueToleranceM"],
            self.ray_options.event_value_tolerance,
        )
        self.assertEqual(
            descriptor["frequencyTransfer"]
            ["requestedRecordedPathAbsoluteTolerance"],
            self.sampler.recorded_path_absolute_tolerance,
        )
        self.assertEqual(
            descriptor["frequencyTransfer"]
            ["fineRecordedPathAbsoluteTolerance"],
            self.ray_options.absolute_tolerance,
        )
        self.assertEqual(
            descriptor["frequencyTransfer"]
            ["coarseRecordedPathAbsoluteTolerance"],
            self.ray_options.absolute_tolerance
            * self.sampler.coarse_tolerance_multiplier,
        )
        self.assertEqual(
            descriptor["frequencyTransfer"]["recordedPathTolerancePolicy"],
            "per trace max(configured minimum, producing ray local tolerance)",
        )
        self.assertEqual(
            descriptor["convergencePolicy"]["actual"]
            ["specificIntensityRelativeTolerance"],
            1.0e-3,
        )
        self.assertEqual(
            descriptor["convergencePolicy"]["maxima"]
            ["specificIntensityAbsoluteTolerance"],
            0.0,
        )
        self.assertLessEqual(
            descriptor["traceAccuracyPolicy"]["actual"]["fineMaximumStep"],
            descriptor["traceAccuracyPolicy"]["maxima"]["fineMaximumStep"],
        )
        self.assertEqual(
            descriptor["observer"]["materialClearance"]["status"],
            "outside-certified",
        )
        self.assertGreater(
            descriptor["finiteThicknessSurface"]
            ["maximumPhotosphereOblateRadiusM"],
            descriptor["finiteThicknessSurface"]["outerRadiusOverMass"],
        )
        self.assertEqual(
            descriptor["finiteThicknessSurface"]
            ["thinnessGateMaximumHOverRho"],
            self.calibration.thinness_gate_maximum_h_over_rho,
        )
        self.assertFalse(
            descriptor["scientificStatus"]
            ["isGeneralRelativisticMagnetohydrodynamics"]
        )
        self.assertFalse(
            descriptor["scientificStatus"]
            ["isCompleteGeneralRelativisticRadiativeTransfer"]
        )
        self.assertFalse(
            descriptor["scientificStatus"]["includesReturningRadiation"]
        )
        json.dumps(descriptor, allow_nan=False, sort_keys=True)
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["includesReturningRadiation"] = True

    def test_capture_target_name_is_bound_to_the_kerr_horizon_radius(self) -> None:
        exact_horizon = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=50.0,
            offset_m=0.0,
        )
        exact_sampler = self.make_sampler(
            observer_theta_rad=1.1,
            termination=exact_horizon,
        )
        self.assertEqual(
            exact_sampler.termination.capture_radius_m.hex(),
            self.metric.outer_horizon_radius_m.hex(),
        )

        mislabeled_event = replace(
            self.termination,
            capture_target_id="analytic-kerr-event-horizon",
        )
        with self.assertRaisesRegex(
            ValueError,
            "event-horizon capture target must use the exact Kerr",
        ):
            self.make_sampler(
                observer_theta_rad=1.1,
                termination=mislabeled_event,
            )

        mislabeled_stretched = replace(
            exact_horizon,
            capture_target_id="analytic-kerr-stretched-horizon",
        )
        with self.assertRaisesRegex(
            ValueError,
            "stretched-horizon capture target must lie strictly outside",
        ):
            self.make_sampler(
                observer_theta_rad=1.1,
                termination=mislabeled_stretched,
            )

    def test_observer_outer_worldtube_and_tolerance_bypasses_fail_closed(self) -> None:
        outer_point = self.calibration.photosphere_point(
            self.calibration.outer_radius_over_mass,
            "upper",
        )
        with self.assertRaisesRegex(ValueError, "observer lies on or inside"):
            self.make_sampler(
                observer_theta_rad=outer_point.theta_rad,
                observer_radius_m=outer_point.radius_over_mass,
            )

        forged_escape = KerrOblateTermination(
            spin_a_m=self.metric.spin_a_m,
            capture_radius_m=self.termination.capture_radius_m,
            escape_radius_m=0.5
            * (
                self.calibration.outer_radius_over_mass
                + outer_point.radius_over_mass
            ),
        )
        with self.assertRaisesRegex(ValueError, "maximum finite-photosphere"):
            self.make_sampler(
                observer_theta_rad=1.1,
                observer_radius_m=(
                    self.calibration.outer_radius_over_mass + 1.0e-3
                ),
                termination=forged_escape,
            )

        convergence_bypasses = (
            {"coarse_tolerance_multiplier": 1.0e6},
            {"terminal_event_tolerance_m": 1.0},
            {"terminal_covector_tolerance": 1.0},
            {"disk_radius_absolute_tolerance_m": 1.0},
            {"disk_radius_relative_tolerance": 1.0},
            {"frequency_shift_relative_tolerance": 1.0},
            {"emission_cosine_absolute_tolerance": 1.0},
            {"specific_intensity_absolute_tolerance": 1.0},
            {"specific_intensity_relative_tolerance": 1.0},
            {"escape_direction_tolerance_rad": 1.0},
        )
        for mutation in convergence_bypasses:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "policy maximum"):
                    replace(self.sampler, **mutation)

        with self.assertRaisesRegex(ValueError, "trace policy maximum"):
            replace(
                self.sampler,
                fine_options=replace(
                    self.ray_options,
                    relative_tolerance=1.0e-3,
                ),
            )
        with self.assertRaisesRegex(ValueError, "transfer policy maximum"):
            replace(
                self.sampler,
                recorded_path_relative_tolerance=2.0e-7,
            )
        with self.assertRaisesRegex(
            ValueError,
            "resolved fine/coarse recorded-path",
        ):
            replace(
                self.sampler,
                fine_options=replace(
                    self.ray_options,
                    absolute_tolerance=2.0e-9,
                    relative_tolerance=2.0e-9,
                ),
                coarse_tolerance_multiplier=64.0,
            )

        class ForeignCalibration(StationaryKerrFiniteThicknessCalibration):
            pass

        foreign_surface = KerrFiniteThicknessMultiSurface(
            self.metric,
            ForeignCalibration(
                dimensionless_spin=0.7,
                eddington_scaled_mass_accretion_rate=0.05,
                outer_radius_over_mass=25.0,
            ),
        )
        with self.assertRaisesRegex(TypeError, "surface calibration must be the exact"):
            self.make_sampler(
                observer_theta_rad=1.1,
                surface=foreign_surface,
            )

    def test_real_upper_and_lower_samples_are_reflection_symmetric(self) -> None:
        samples = []
        for theta, screen_y, expected_surface_id in (
            (1.1, -0.5, UPPER_SURFACE_ID),
            (math.pi - 1.1, 0.5, LOWER_SURFACE_ID),
        ):
            sample = self.make_sampler(observer_theta_rad=theta).sample(
                0.5,
                screen_y,
                (3.0e14, 5.0e14),
            )
            self.assertEqual(sample.visible_source, "disk")
            self.assertTrue(sample.ray_converged)
            self.assertGreater(sample.frequency_shift_g, 0.0)
            self.assertIn(expected_surface_id, sample.topology_signature)
            self.assertTrue(
                all(value > 0.0 for value in sample.specific_intensities_nu)
            )
            self.assertTrue(sample.convergence_audit.ray_gate_passed)
            self.assertTrue(sample.convergence_audit.source_gate_passed)
            self.assertTrue(sample.convergence_audit.transfer_gate_passed)
            samples.append(sample)

        upper, lower = samples
        self.assertAlmostEqual(
            upper.frequency_shift_g,
            lower.frequency_shift_g,
            places=13,
        )
        self.assertAlmostEqual(
            upper.convergence_audit.disk_radius_difference_m,
            lower.convergence_audit.disk_radius_difference_m,
            places=13,
        )
        for upper_value, lower_value in zip(
            upper.specific_intensities_nu,
            lower.specific_intensities_nu,
        ):
            self.assertAlmostEqual(upper_value, lower_value, places=18)

    def test_capture_is_black_and_escape_uses_closed_observer_spectrum(self) -> None:
        background = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=2.5,
            reference_frequency_hz=5.0e14,
            spectral_index=-1.0,
        )
        cases = (
            (0.2, 0.0, 0.0, "captured-boundary", (0.0, 0.0)),
            (0.4, 3.0, 3.0, "escaped-boundary", (5.0, 2.5)),
        )
        for theta, screen_x, screen_y, expected_source, expected_spectrum in cases:
            sample = self.make_sampler(
                observer_theta_rad=theta,
                escaped=background,
            ).sample(
                screen_x,
                screen_y,
                (2.5e14, 5.0e14),
            )
            with self.subTest(expected_source=expected_source):
                self.assertEqual(sample.visible_source, expected_source)
                self.assertIsNone(sample.frequency_shift_g)
                for actual, expected in zip(
                    sample.specific_intensities_nu,
                    expected_spectrum,
                ):
                    self.assertAlmostEqual(actual, expected, places=12)
                if expected_source == "captured-boundary":
                    self.assertIsNone(sample.escape_direction)
                    self.assertTrue(
                        all(
                            value == 0.0 and math.copysign(1.0, value) > 0.0
                            for value in sample.specific_intensities_nu
                        )
                    )
                else:
                    self.assertIsNotNone(sample.escape_direction)

    def test_dotm_to_zero_converges_to_thin_disk_frame_oracle(self) -> None:
        thin_height_surface = KerrFiniteThicknessMultiSurface(
            self.metric,
            StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.7,
                eddington_scaled_mass_accretion_rate=1.0e-6,
                outer_radius_over_mass=25.0,
            ),
        )
        finite_sample = self.make_sampler(
            observer_theta_rad=1.1,
            surface=thin_height_surface,
        ).sample(0.5, -0.5, (5.0e14,))
        thin_sample = KerrDiskRaySampler(
            metric=self.metric,
            observer_radius_m=30.0,
            termination=self.termination,
            disk=self.disk,
            outer_radius_m=25.0,
            escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
            fine_options=self.ray_options,
            surface_options=self.surface_options,
            observer_theta_rad=1.1,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance_m=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
            specific_intensity_relative_tolerance=1.0e-3,
        ).sample(0.5, -0.5, (5.0e14,))
        comparisons = (
            (finite_sample.frequency_shift_g, thin_sample.frequency_shift_g),
            (
                finite_sample.specific_intensities_nu[0],
                thin_sample.specific_intensities_nu[0],
            ),
        )
        for actual, expected in comparisons:
            self.assertLess(
                abs(actual - expected) / max(abs(expected), 1.0e-300),
                8.0e-7,
            )

    def test_stale_refinement_and_topology_wrappers_fail_closed(self) -> None:
        stale_diagnostics = replace(
            self.cached_refinement,
            terminal_event_difference=0.0,
            converged=True,
        )
        with mock.patch.object(
            frame_module,
            "trace_refined_null_geodesic",
            return_value=stale_diagnostics,
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessFrameError,
                "diagnostics are stale",
            ):
                self.sampler.sample(0.5, -0.5, (5.0e14,))

        coarse = self.cached_refinement.coarse
        trace = coarse.multi_surface_trace
        terminal = trace.crossings[-1]
        forged_crossing = replace(
            terminal.crossing,
            orientation=-terminal.crossing.orientation,
        )
        forged_entry = replace(terminal, crossing=forged_crossing)
        forged_trace = replace(trace, crossings=(*trace.crossings[:-1], forged_entry))
        forged_coarse = replace(coarse, multi_surface_trace=forged_trace)
        forged_refinement = replace(
            self.cached_refinement,
            coarse=forged_coarse,
            converged=True,
        )
        with mock.patch.object(
            frame_module,
            "trace_refined_null_geodesic",
            return_value=forged_refinement,
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessFrameError,
                "diagnostics are stale",
            ):
                self.sampler.sample(0.5, -0.5, (5.0e14,))

        forged_probe_trace = replace(
            trace,
            maximum_probe_event_difference=(
                2.0 * self.sampler.terminal_event_tolerance_m
            ),
        )
        forged_probe_coarse = replace(
            coarse,
            multi_surface_trace=forged_probe_trace,
        )
        forged_probe_refinement = replace(
            self.cached_refinement,
            coarse=forged_probe_coarse,
        )
        with mock.patch.object(
            frame_module,
            "trace_refined_null_geodesic",
            return_value=forged_probe_refinement,
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessFrameError,
                "multi-surface convergence diagnostics",
            ):
                self.sampler.sample(0.5, -0.5, (5.0e14,))

    def test_mutated_transfer_wrapper_and_failed_audit_are_rejected(self) -> None:
        real_transfer = (
            frame_module._transfer_kerr_finite_thickness_spectrum_certified
        )

        def mutate_transfer(*args, **kwargs):
            result, certificate = real_transfer(*args, **kwargs)
            if result.frequency_shift_g is not None:
                object.__setattr__(
                    result,
                    "frequency_shift_g",
                    1.001 * result.frequency_shift_g,
                )
            return result, certificate

        with mock.patch.object(
            frame_module,
            "trace_refined_null_geodesic",
            return_value=self.cached_refinement,
        ), mock.patch.object(
            frame_module,
            "_transfer_kerr_finite_thickness_spectrum_certified",
            side_effect=mutate_transfer,
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessFrameError,
                "self-revalidation",
            ):
                self.sampler.sample(0.5, -0.5, (5.0e14,))

        clean = self.sampler.sample(0.5, -0.5, (5.0e14,))
        failed_audit = replace(
            clean.convergence_audit,
            transfer_gate_passed=False,
        )
        self.assertIsInstance(failed_audit, RayConvergenceAudit)
        with self.assertRaisesRegex(ValueError, "failed audit gate"):
            replace(clean, convergence_audit=failed_audit)

    def test_frame_replays_geometry_once_for_each_fine_and_coarse_ray(
        self,
    ) -> None:
        with mock.patch.object(
            frame_module,
            "trace_refined_null_geodesic",
            return_value=self.cached_refinement,
        ), mock.patch.object(
            replay_module,
            "trace_null_geodesic",
            wraps=replay_module.trace_null_geodesic,
        ) as replay:
            sample = self.sampler.sample(0.5, -0.5, (5.0e14,))

        self.assertTrue(sample.convergence_audit.ray_gate_passed)
        self.assertTrue(sample.convergence_audit.source_gate_passed)
        self.assertTrue(sample.convergence_audit.transfer_gate_passed)
        self.assertEqual(replay.call_count, 2)


if __name__ == "__main__":
    unittest.main()
