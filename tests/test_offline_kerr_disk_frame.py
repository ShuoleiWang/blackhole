from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import inspect
import json
import math
import unittest
from unittest.mock import patch

import offline.kerr_disk_frame as kerr_disk_frame_module
from offline.disk_atmosphere import (
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
)
from offline.geodesic import (
    HamiltonianState,
    RayPathSegment,
    RayRefinementResult,
    RayTraceOptions,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_oblate_event_to_ks_cartesian,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    KerrDiskFrameError,
    KerrDiskRaySampler,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_disk_transfer import (
    KerrDiskCrossingSignatureEntry,
    KerrDiskSpectrumResult,
)


SOLAR_MASS_KG = 1.98847e30


@dataclass(frozen=True)
class TaggedEscapedSpectrum:
    scale: float = 1.0
    frequency_frame: str = "observer"

    def __call__(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
        boundary_target_id: str,
    ) -> float:
        return self.scale * observer_frequency_hz / 1.0e14

    def descriptor(self) -> dict[str, object]:
        return {
            "frequencyFrame": self.frequency_frame,
            "implementationId": "test-tagged-observer-spectrum/v1",
            "scale": self.scale,
        }


@dataclass(frozen=True)
class SpoofedBuiltInAngularLaw:
    multiplier: float = 2.0

    def intensity_multiplier(self, emission_angle_cosine: float) -> float:
        del emission_angle_cosine
        return self.multiplier

    def descriptor(self) -> dict[str, object]:
        return dict(FluxConservingLinearLimbDarkening().descriptor())


class OfflineKerrDiskFrameTests(unittest.TestCase):
    def make_sampler(self, **changes) -> KerrDiskRaySampler:
        metric = changes.pop(
            "metric",
            KerrKerrSchildMetric(spin_a_m=0.7),
        )
        termination = changes.pop(
            "termination",
            KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=50.0,
                offset_m=0.02,
            ),
        )
        disk = changes.pop(
            "disk",
            StationaryNovikovThorneDisk(
                metric=metric,
                black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
                mass_accretion_rate_kg_s=1.0e22,
            ),
        )
        fine_options = changes.pop(
            "fine_options",
            RayTraceOptions(
                absolute_tolerance=5.0e-10,
                relative_tolerance=5.0e-10,
                initial_step=0.05,
                maximum_step=0.25,
                maximum_affine_length=300.0,
                null_residual_limit=2.0e-7,
                record_path=True,
            ),
        )
        surface_options = changes.pop(
            "surface_options",
            SurfaceEventOptions(
                absolute_tolerance=5.0e-10,
                relative_tolerance=5.0e-10,
                null_residual_limit=2.0e-7,
            ),
        )
        arguments = {
            "metric": metric,
            "observer_radius_m": 30.0,
            "observer_theta_rad": 1.1,
            "termination": termination,
            "disk": disk,
            "outer_radius_m": 25.0,
            "escaped_observer_spectrum": PowerLawEscapedObserverSpectrum(
                reference_specific_intensity_nu=1.0,
                reference_frequency_hz=1.0e14,
                spectral_index=1.0,
            ),
            "fine_options": fine_options,
            "surface_options": surface_options,
            "frequency_null_residual_limit": 2.0e-7,
        }
        arguments.update(changes)
        return KerrDiskRaySampler(**arguments)

    def test_exactly_edge_on_observer_is_rejected_for_zero_thickness_disk(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "exactly edge-on"):
            self.make_sampler(observer_theta_rad=0.5 * math.pi)

    def test_capture_worldtube_cannot_swallow_the_opaque_disk(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        termination = KerrOblateTermination(
            spin_a_m=metric.spin_a_m,
            capture_radius_m=26.0,
            escape_radius_m=50.0,
        )
        with self.assertRaisesRegex(ValueError, "strictly inside the disk ISCO"):
            self.make_sampler(metric=metric, termination=termination)

    def synthetic_ray(
        self,
        sampler: KerrDiskRaySampler,
        *,
        outcome: str = "escaped",
        terminal_phi_ks_rad: float = 0.2,
    ) -> RayTraceResult:
        initial = kerr_zamo_camera_ray(
            sampler.metric,
            observer_radius_m=sampler.observer_radius_m,
            theta_rad=sampler.observer_theta_rad,
            phi_ks_rad=sampler.observer_phi_ks_rad,
            coordinate_time_m=sampler.observer_coordinate_time_m,
            screen_x=0.0,
            screen_y=0.0,
        )
        radius = (
            sampler.termination.escape_radius_m
            if outcome == "escaped"
            else sampler.termination.capture_radius_m
        )
        terminal = HamiltonianState(
            event=kerr_oblate_event_to_ks_cartesian(
                coordinate_time_m=-10.0,
                radius_m=radius,
                theta_rad=1.2,
                phi_ks_rad=terminal_phi_ks_rad,
                spin_a_m=sampler.metric.spin_a_m,
            ),
            covector=initial.covector,
        )
        midpoint = HamiltonianState(
            event=tuple(
                0.5 * (initial.event[index] + terminal.event[index])
                for index in range(4)
            ),  # type: ignore[arg-type]
            covector=initial.covector,
        )
        segment = RayPathSegment(
            start=initial,
            midpoint=midpoint,
            end=terminal,
            affine_length=1.0,
            midpoint_null_residual=0.0,
        )
        target = (
            sampler.termination.escape_target_id
            if outcome == "escaped"
            else sampler.termination.capture_target_id
        )
        return RayTraceResult(
            outcome=outcome,
            terminal_state=terminal,
            affine_length=1.0,
            accepted_steps=1,
            rejected_steps=0,
            maximum_null_residual=0.0,
            maximum_metric_interpolation_error=0.0,
            segments=(segment,),
            terminal_target_id=target,
        )

    def refinement(
        self,
        sampler: KerrDiskRaySampler,
        *,
        fine_outcome: str = "escaped",
        coarse_outcome: str | None = None,
        fine_phi: float = 0.2,
        coarse_phi: float = 0.2,
        outcome_agrees: bool = True,
        terminal_target_agrees: bool = True,
        converged: bool = True,
    ) -> RayRefinementResult:
        coarse_outcome = coarse_outcome or fine_outcome
        fine = self.synthetic_ray(
            sampler,
            outcome=fine_outcome,
            terminal_phi_ks_rad=fine_phi,
        )
        coarse = self.synthetic_ray(
            sampler,
            outcome=coarse_outcome,
            terminal_phi_ks_rad=coarse_phi,
        )
        return RayRefinementResult(
            fine=fine,
            coarse=coarse,
            outcome_agrees=outcome_agrees,
            terminal_event_difference=0.0,
            terminal_covector_difference=0.0,
            discretizations_differ=True,
            terminal_target_agrees=terminal_target_agrees,
            converged=converged,
        )

    def boundary_transfer(
        self,
        sampler: KerrDiskRaySampler,
        frequencies: tuple[float, ...],
        intensities: tuple[float, ...],
        *,
        outcome: str = "escaped",
        signature: tuple[KerrDiskCrossingSignatureEntry, ...] = (),
        bracket_widths: tuple[float, ...] | None = None,
    ) -> KerrDiskSpectrumResult:
        bracket_widths = bracket_widths or tuple(0.0 for _entry in signature)
        return KerrDiskSpectrumResult(
            observer_frequencies_hz=frequencies,
            observed_specific_intensities_nu=intensities,
            source_kind=f"{outcome}-boundary",  # type: ignore[arg-type]
            ray_boundary_outcome=outcome,  # type: ignore[arg-type]
            ray_boundary_target_id=(
                sampler.termination.escape_target_id
                if outcome == "escaped"
                else sampler.termination.capture_target_id
            ),
            crossing_signature=signature,
            crossing_bracket_affine_widths=bracket_widths,
        )

    def disk_transfer(
        self,
        sampler: KerrDiskRaySampler,
        frequencies: tuple[float, ...],
        *,
        frequency_shift_g: float = 1.0,
        signature: tuple[KerrDiskCrossingSignatureEntry, ...] | None = None,
        first_opaque_index: int = 0,
        boundary_outcome: str = "escaped",
        radius: float = 8.0,
        bracket_widths: tuple[float, ...] | None = None,
        emission_angle_cosine: float = 0.5,
    ) -> KerrDiskSpectrumResult:
        signature = signature or (
            KerrDiskCrossingSignatureEntry(-1, "opaque-annulus"),
        )
        bracket_widths = bracket_widths or tuple(0.0 for _entry in signature)
        emitter = sampler.disk.emitter(radius)
        crossing = RecordedSurfaceCrossing(
            state=HamiltonianState(
                event=emitter.event,
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            ray_affine_length=0.5,
            segment_index=0,
            segment_affine_length=0.5,
            orientation=signature[first_opaque_index].orientation,
            surface_value=0.0,
            bracket_affine_width=bracket_widths[first_opaque_index],
            iterations=0,
        )
        emitted_frequencies = tuple(
            frequency / frequency_shift_g for frequency in frequencies
        )
        isotropic_emitted = tuple(
            sampler.disk.emitted_specific_intensity_nu(radius, frequency)
            for frequency in emitted_frequencies
        )
        angular_multiplier = sampler.angular_emission_law.intensity_multiplier(
            emission_angle_cosine
        )
        emitted = tuple(
            intensity * angular_multiplier for intensity in isotropic_emitted
        )
        observed = tuple(
            frequency_shift_g**3 * intensity for intensity in emitted
        )
        return KerrDiskSpectrumResult(
            observer_frequencies_hz=frequencies,
            observed_specific_intensities_nu=observed,
            source_kind="disk",
            ray_boundary_outcome=boundary_outcome,  # type: ignore[arg-type]
            ray_boundary_target_id=(
                sampler.termination.escape_target_id
                if boundary_outcome == "escaped"
                else sampler.termination.capture_target_id
            ),
            crossing=crossing,
            disk_radius_m=radius,
            emitter=emitter,
            frequency_shift_g=frequency_shift_g,
            emitted_frequencies_hz=emitted_frequencies,
            isotropic_emitted_specific_intensities_nu=isotropic_emitted,
            emitted_specific_intensities_nu=emitted,
            emission_angle_cosine=emission_angle_cosine,
            angular_emission_multiplier=angular_multiplier,
            emitter_event_tolerance_m=(
                1.0e-8 * sampler.metric.mass_m
                if sampler.emitter_event_tolerance_m is None
                else sampler.emitter_event_tolerance_m
            ),
            crossing_signature=signature,
            crossing_bracket_affine_widths=bracket_widths,
            first_opaque_crossing_index=first_opaque_index,
        )

    def test_sampler_calls_one_refinement_and_returns_fine_escape_sample(self) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14, 2.0e14)
        refinement = self.refinement(sampler)
        fine = self.boundary_transfer(sampler, frequencies, (1.0, 2.0))
        coarse = self.boundary_transfer(
            sampler,
            frequencies,
            (1.00001, 1.99998),
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ) as trace,
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine, coarse),
            ) as transfer,
        ):
            sample = sampler.sample(0.0, 0.0, frequencies)

        self.assertEqual(sample.specific_intensities_nu, (1.0, 2.0))
        self.assertEqual(
            sample.absolute_errors_nu,
            (abs(1.0 - 1.00001), abs(2.0 - 1.99998)),
        )
        self.assertEqual(sample.visible_source, "escaped-boundary")
        self.assertIsNone(sample.frequency_shift_g)
        self.assertIsNotNone(sample.escape_direction)
        topology = json.loads(sample.topology_signature)
        self.assertEqual(topology["terminal"]["outcome"], "escaped")
        self.assertEqual(trace.call_count, 1)
        self.assertTrue(trace.call_args.kwargs["record_coarse_path"])
        self.assertEqual(transfer.call_count, 2)
        self.assertTrue(sample.convergence_audit.ray_gate_passed)
        self.assertTrue(sample.convergence_audit.source_gate_passed)
        self.assertTrue(sample.convergence_audit.transfer_gate_passed)
        self.assertEqual(sample.convergence_audit.accepted_steps, 1)
        self.assertEqual(sample.convergence_audit.rejected_steps, 0)
        self.assertEqual(
            trace.call_args.args[1],
            refinement.fine.segments[0].start,
        )

    def test_disk_topology_hides_crossings_and_terminal_behind_first_opaque(
        self,
    ) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14,)
        signature = (
            KerrDiskCrossingSignatureEntry(1, "inside-isco"),
            KerrDiskCrossingSignatureEntry(-1, "opaque-annulus"),
            KerrDiskCrossingSignatureEntry(1, "outside-outer-radius"),
        )
        transfer = self.disk_transfer(
            sampler,
            frequencies,
            signature=signature,
            first_opaque_index=1,
            boundary_outcome="captured",
            bracket_widths=(0.1, 0.2, 99.0),
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=self.refinement(
                    sampler,
                    fine_outcome="captured",
                ),
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(transfer, transfer),
            ),
        ):
            sample = sampler.sample(0.0, 0.0, frequencies)

        topology = json.loads(sample.topology_signature)
        self.assertEqual(len(topology["crossings"]), 2)
        self.assertNotIn("terminal", topology)
        self.assertEqual(topology["firstOpaqueCrossingIndex"], 1)
        self.assertEqual(sample.visible_source, "disk")
        self.assertEqual(sample.frequency_shift_g, 1.0)
        self.assertIsNone(sample.escape_direction)
        self.assertEqual(
            sample.convergence_audit.surface_bracket_affine_width,
            0.2,
        )

    def test_disk_result_cross_fields_are_jointly_bound(self) -> None:
        sampler = self.make_sampler()
        result = self.disk_transfer(sampler, (1.0e14,))
        with self.assertRaisesRegex(ValueError, "radius disagrees"):
            replace(result, disk_radius_m=9.0)
        with self.assertRaisesRegex(ValueError, "radius disagrees"):
            replace(result, emitter=sampler.disk.emitter(10.0))
        other_emitter = sampler.disk.emitter(10.0)
        assert result.crossing is not None
        with self.assertRaisesRegex(ValueError, "crossing event disagrees"):
            replace(
                result,
                crossing=replace(
                    result.crossing,
                    state=replace(
                        result.crossing.state,
                        event=other_emitter.event,
                    ),
                ),
            )
        changed_angle = 0.2
        with self.assertRaisesRegex(ValueError, "angular emission"):
            replace(
                result,
                emission_angle_cosine=changed_angle,
                angular_emission_multiplier=(
                    sampler.angular_emission_law.intensity_multiplier(
                        changed_angle
                    )
                ),
            )

    def test_visible_disk_convergence_ignores_hidden_terminal_divergence(self) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14,)
        refinement = self.refinement(
            sampler,
            fine_outcome="captured",
            coarse_outcome="escaped",
            outcome_agrees=False,
            terminal_target_agrees=False,
            converged=False,
        )
        visible_prefix = (
            KerrDiskCrossingSignatureEntry(1, "inside-isco"),
            KerrDiskCrossingSignatureEntry(-1, "opaque-annulus"),
        )
        fine = self.disk_transfer(
            sampler,
            frequencies,
            signature=(
                *visible_prefix,
                KerrDiskCrossingSignatureEntry(1, "outside-outer-radius"),
            ),
            first_opaque_index=1,
            boundary_outcome="captured",
            bracket_widths=(1.0e-7, 2.0e-7, 99.0),
        )
        coarse = self.disk_transfer(
            sampler,
            frequencies,
            signature=visible_prefix,
            first_opaque_index=1,
            boundary_outcome="escaped",
            bracket_widths=(1.0e-7, 2.0e-7),
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine, coarse),
            ),
        ):
            sample = sampler.sample(0.0, 0.0, frequencies)

        self.assertEqual(sample.visible_source, "disk")
        self.assertTrue(sample.ray_converged)
        self.assertEqual(sample.convergence_audit.terminal_event_difference_m, 0.0)
        self.assertEqual(
            sample.convergence_audit.terminal_covector_relative_difference,
            0.0,
        )
        self.assertEqual(
            sample.convergence_audit.surface_bracket_affine_width,
            2.0e-7,
        )

    def test_visible_source_and_full_crossing_mismatches_fail_closed(self) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14,)
        refinement = self.refinement(sampler)
        disk = self.disk_transfer(sampler, frequencies)
        escaped = self.boundary_transfer(sampler, frequencies, (1.0,))
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(disk, escaped),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "visible sources"):
                sampler.sample(0.0, 0.0, frequencies)

        inside = (KerrDiskCrossingSignatureEntry(1, "inside-isco"),)
        outside = (
            KerrDiskCrossingSignatureEntry(1, "outside-outer-radius"),
        )
        fine = self.boundary_transfer(
            sampler,
            frequencies,
            (1.0,),
            signature=inside,
        )
        coarse = self.boundary_transfer(
            sampler,
            frequencies,
            (1.0,),
            signature=outside,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine, coarse),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "topologies"):
                sampler.sample(0.0, 0.0, frequencies)

    def test_g_intensity_and_escape_direction_thresholds_fail_closed(self) -> None:
        frequencies = (1.0e14,)
        sampler = self.make_sampler(
            frequency_shift_relative_tolerance=1.0e-3,
            specific_intensity_relative_tolerance=1.0e-3,
            escape_direction_tolerance_rad=1.0e-3,
        )
        refinement = self.refinement(sampler)
        fine_disk = self.disk_transfer(
            sampler,
            frequencies,
            frequency_shift_g=1.0,
        )
        coarse_disk = self.disk_transfer(
            sampler,
            frequencies,
            frequency_shift_g=1.01,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine_disk, coarse_disk),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "frequency shifts"):
                sampler.sample(0.0, 0.0, frequencies)

        fine_angle = self.disk_transfer(
            sampler,
            frequencies,
            emission_angle_cosine=0.2,
        )
        coarse_angle = self.disk_transfer(
            sampler,
            frequencies,
            emission_angle_cosine=0.3,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine_angle, coarse_angle),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "emission angles"):
                sampler.sample(0.0, 0.0, frequencies)

        fine_escape = self.boundary_transfer(sampler, frequencies, (1.0,))
        coarse_escape = self.boundary_transfer(sampler, frequencies, (1.01,))
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine_escape, coarse_escape),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "intensity bin"):
                sampler.sample(0.0, 0.0, frequencies)

        divergent_direction = self.refinement(
            sampler,
            fine_phi=0.0,
            coarse_phi=0.1,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=divergent_direction,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine_escape, fine_escape),
            ),
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "escape directions"):
                sampler.sample(0.0, 0.0, frequencies)

    def test_boundary_separatrix_disagreement_fails_after_visible_transfer(self) -> None:
        sampler = self.make_sampler()
        refinement = self.refinement(
            sampler,
            fine_outcome="captured",
            coarse_outcome="escaped",
            outcome_agrees=False,
            terminal_target_agrees=False,
            converged=False,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(
                    self.boundary_transfer(
                        sampler,
                        (1.0e14,),
                        (0.0,),
                        outcome="captured",
                    ),
                    self.boundary_transfer(
                        sampler,
                        (1.0e14,),
                        (1.0,),
                        outcome="escaped",
                    ),
                ),
            ) as transfer,
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "visible sources"):
                sampler.sample(0.0, 0.0, (1.0e14,))
        self.assertEqual(transfer.call_count, 2)

    def test_worldtube_target_mismatch_fails_before_transfer(self) -> None:
        sampler = self.make_sampler()
        refinement = self.refinement(sampler)
        wrong_target = replace(
            refinement,
            fine=replace(refinement.fine, terminal_target_id="wrong-target"),
            coarse=replace(refinement.coarse, terminal_target_id="wrong-target"),
            terminal_target_agrees=True,
            converged=True,
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=wrong_target,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
            ) as transfer,
        ):
            with self.assertRaisesRegex(KerrDiskFrameError, "worldtube"):
                sampler.sample(0.0, 0.0, (1.0e14,))
        transfer.assert_not_called()

    def test_convergence_audit_records_numerical_high_water_marks(self) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14,)
        baseline = self.refinement(sampler)
        refinement = replace(
            baseline,
            fine=replace(
                baseline.fine,
                accepted_steps=3,
                rejected_steps=2,
                maximum_null_residual=2.0e-8,
                maximum_metric_interpolation_error=3.0e-9,
            ),
            coarse=replace(
                baseline.coarse,
                accepted_steps=5,
                rejected_steps=4,
                maximum_null_residual=4.0e-8,
                maximum_metric_interpolation_error=1.0e-8,
            ),
            terminal_event_difference=7.0e-7,
            terminal_covector_difference=8.0e-8,
        )
        signature = (
            KerrDiskCrossingSignatureEntry(1, "inside-isco"),
            KerrDiskCrossingSignatureEntry(-1, "opaque-annulus"),
        )
        fine = self.disk_transfer(
            sampler,
            frequencies,
            radius=8.0,
            frequency_shift_g=1.0,
            signature=signature,
            first_opaque_index=1,
            bracket_widths=(1.0e-6, 2.0e-6),
        )
        coarse = self.disk_transfer(
            sampler,
            frequencies,
            radius=8.00001,
            frequency_shift_g=1.000001,
            signature=signature,
            first_opaque_index=1,
            bracket_widths=(3.0e-6, 4.0e-6),
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=refinement,
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(fine, coarse),
            ),
        ):
            sample = sampler.sample(0.0, 0.0, frequencies)

        audit = sample.convergence_audit
        self.assertEqual(audit.maximum_null_residual, 4.0e-8)
        self.assertEqual(audit.maximum_metric_interpolation_error, 1.0e-8)
        self.assertEqual(audit.terminal_event_difference_m, 0.0)
        self.assertEqual(audit.terminal_covector_relative_difference, 0.0)
        self.assertAlmostEqual(audit.disk_radius_difference_m, 1.0e-5)
        self.assertAlmostEqual(audit.relative_g_difference, 1.0e-6 / 1.000001)
        self.assertEqual(audit.surface_bracket_affine_width, 4.0e-6)
        self.assertEqual(audit.accepted_steps, 5)
        self.assertEqual(audit.rejected_steps, 4)
        self.assertTrue(audit.ray_gate_passed)
        self.assertTrue(audit.source_gate_passed)
        self.assertTrue(audit.transfer_gate_passed)

    def test_captured_visible_boundary_has_no_escape_direction(self) -> None:
        sampler = self.make_sampler()
        frequencies = (1.0e14,)
        captured = self.boundary_transfer(
            sampler,
            frequencies,
            (0.0,),
            outcome="captured",
        )
        with (
            patch(
                "offline.kerr_disk_frame.trace_refined_null_geodesic",
                return_value=self.refinement(
                    sampler,
                    fine_outcome="captured",
                ),
            ),
            patch(
                "offline.kerr_disk_frame.transfer_kerr_disk_spectrum",
                side_effect=(captured, captured),
            ),
        ):
            sample = sampler.sample(0.0, 0.0, frequencies)
        self.assertEqual(sample.visible_source, "captured-boundary")
        self.assertIsNone(sample.escape_direction)

    def test_descriptor_binds_physical_and_numerical_parameters(self) -> None:
        baseline_sampler = self.make_sampler()
        baseline = baseline_sampler.descriptor()
        encoded = json.dumps(
            baseline,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(
            baseline["escapeDirectionDiagnostic"]["frame"],
            "finite-worldtube-KS-angular-continuation-direction",
        )
        self.assertFalse(
            baseline["escapedObserverSpectrum"]["samplerAppliesAdditionalG3"]
        )

        variations = (
            self.make_sampler(observer_theta_rad=1.0),
            self.make_sampler(outer_radius_m=24.0),
            self.make_sampler(
                disk=StationaryNovikovThorneDisk(
                    metric=baseline_sampler.metric,
                    black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
                    mass_accretion_rate_kg_s=2.0e22,
                ),
                metric=baseline_sampler.metric,
                termination=baseline_sampler.termination,
            ),
            self.make_sampler(
                fine_options=replace(
                    baseline_sampler.fine_options,
                    relative_tolerance=6.0e-10,
                ),
            ),
            self.make_sampler(
                surface_options=replace(
                    baseline_sampler.surface_options,
                    subdivisions_per_segment=4,
                ),
            ),
            self.make_sampler(coarse_tolerance_multiplier=16.0),
            self.make_sampler(
                escaped_observer_spectrum=PowerLawEscapedObserverSpectrum(
                    reference_specific_intensity_nu=2.0,
                    reference_frequency_hz=1.0e14,
                    spectral_index=1.0,
                )
            ),
            self.make_sampler(
                angular_emission_law=IsotropicAngularEmission(),
            ),
            self.make_sampler(
                angular_emission_law=(
                    FluxConservingLinearLimbDarkening(2.06)
                ),
            ),
        )
        for variation in variations:
            with self.subTest(descriptor=variation.descriptor()):
                self.assertNotEqual(
                    json.dumps(
                        variation.descriptor(),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    encoded,
                )

    def test_spoofed_escaped_spectrum_cannot_reuse_a_builtin_identity(
        self,
    ) -> None:
        with self.assertRaisesRegex(TypeError, "closed built-in"):
            self.make_sampler(
                escaped_observer_spectrum=TaggedEscapedSpectrum(
                    frequency_frame="emitter"
                )
            )

    def test_zero_index_power_law_is_constant_across_extreme_frequencies(
        self,
    ) -> None:
        state = HamiltonianState(
            event=(0.0, 1.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        low_reference = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=1.0,
            reference_frequency_hz=math.ulp(0.0),
            spectral_index=0.0,
        )
        high_reference = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=1.0,
            reference_frequency_hz=1.0e308,
            spectral_index=0.0,
        )
        self.assertEqual(low_reference(state, 1.0e308, "escape"), 1.0)
        self.assertEqual(high_reference(state, math.ulp(0.0), "escape"), 1.0)

    def test_spoofed_angular_law_cannot_reuse_a_builtin_identity(self) -> None:
        with self.assertRaisesRegex(TypeError, "closed built-in"):
            self.make_sampler(
                angular_emission_law=SpoofedBuiltInAngularLaw(),
            )

    def test_real_kerr_approaching_and_receding_samples_converge(self) -> None:
        sampler = self.make_sampler()
        receding = sampler.sample(-0.5, -0.5, (5.0e14,))
        approaching = sampler.sample(0.5, -0.5, (5.0e14,))

        self.assertEqual(receding.visible_source, "disk")
        self.assertEqual(approaching.visible_source, "disk")
        self.assertLess(receding.frequency_shift_g, 1.0)
        self.assertGreater(approaching.frequency_shift_g, 1.0)
        self.assertGreater(
            approaching.frequency_shift_g,
            receding.frequency_shift_g,
        )
        self.assertIsNone(receding.escape_direction)
        self.assertIsNone(approaching.escape_direction)
        self.assertGreater(receding.specific_intensities_nu[0], 0.0)
        self.assertGreater(approaching.specific_intensities_nu[0], 0.0)
        self.assertGreaterEqual(receding.absolute_errors_nu[0], 0.0)
        self.assertGreaterEqual(approaching.absolute_errors_nu[0], 0.0)

    def test_frame_module_imports_no_private_scientific_symbol(self) -> None:
        tree = ast.parse(inspect.getsource(kerr_disk_frame_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "offline.adaptive_frame",
                "offline.geodesic",
                "offline.kerr",
                "offline.kerr_disk",
                "offline.disk_atmosphere",
                "offline.kerr_disk_transfer",
            }:
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("_"), alias.name)
if __name__ == "__main__":
    unittest.main()
