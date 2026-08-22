from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, replace

from offline.adaptive_medium_transfer import (
    AdaptiveMediumBudgetExceeded,
    AdaptiveMediumIntegrationError,
    AdaptiveMediumValidationError,
    AdaptiveScalarTransferOptions,
    FINITE_STENCIL_CAPABILITY,
    propagate_adaptive_scalar_recorded_ray,
)
from offline.geodesic import (
    HamiltonianState,
    RadialTermination,
    RayTraceOptions,
    RecordedPathSamplingOptions,
    trace_null_geodesic,
)
from offline.pipeline import DarkBoundary, propagate_recorded_ray_spectrum
from offline.radiative_transfer import StokesInvariant, TransferCoefficients
from offline.spacetime import (
    MetricSample,
    MinkowskiMetric,
    ZERO_DERIVATIVES,
)


@dataclass(frozen=True)
class ConstantBoundary:
    intensity: float
    source_id: str = "test-constant-boundary"

    def invariant_stokes(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> StokesInvariant:
        return StokesInvariant(self.intensity)


@dataclass(frozen=True)
class PolarizedBoundary:
    source_id: str = "test-polarized-boundary"

    def invariant_stokes(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> StokesInvariant:
        return StokesInvariant(1.0, 0.1, 0.0, 0.0)


@dataclass(frozen=True)
class ThinLayerMedium:
    centre: float = 0.25
    width: float = 0.1
    emissivity: float = 0.0
    absorption: float = 0.0
    source_id: str = "test-thin-layer"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        normalized = 2.0 * (state.event[1] - self.centre) / self.width
        if abs(normalized) >= 1.0:
            return TransferCoefficients()
        profile = (1.0 - normalized * normalized) ** 2
        return TransferCoefficients(
            invariant_emissivity=StokesInvariant(self.emissivity * profile),
            invariant_absorption=self.absorption * profile,
        )


@dataclass(frozen=True)
class DiscontinuousLayerMedium:
    source_id: str = "test-discontinuous-layer"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        inside = abs(state.event[1] - 0.25) < 0.05
        return TransferCoefficients(
            invariant_emissivity=StokesInvariant(2.0 if inside else 0.0),
        )


@dataclass(frozen=True)
class SmoothMedium:
    source_id: str = "test-smooth-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        x = state.event[1]
        return TransferCoefficients(
            invariant_emissivity=StokesInvariant(1.0 + x * x),
        )


@dataclass(frozen=True)
class OrderedMedium:
    hot_near_source: bool
    source_id: str = "test-ordered-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        source_half = state.event[1] >= 0.5
        is_hot = source_half == self.hot_near_source
        return TransferCoefficients(
            invariant_emissivity=StokesInvariant(3.0 if is_hot else 0.0),
            invariant_absorption=1.0,
        )


@dataclass(frozen=True)
class VacuumMedium:
    source_id: str = "test-vacuum-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        return TransferCoefficients()


@dataclass(frozen=True)
class InvalidMedium:
    mode: str
    source_id: str = "test-invalid-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        if self.mode == "nan":
            return TransferCoefficients(invariant_absorption=math.nan)
        return TransferCoefficients(invariant_faraday=(0.0, 0.0, 1.0))


class WrongFlatMetric:
    source_id = "test-wrong-flat-metric"
    time_dependent = False

    def sample(self, event):
        covariant = (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 4.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        inverse = (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.25, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        return MetricSample(
            covariant=covariant,
            inverse=inverse,
            inverse_derivatives=ZERO_DERIVATIVES,
        )


class MissingSampleMetric:
    source_id = "test-missing-sample-metric"
    time_dependent = False


class OfflineAdaptiveMediumTransferTests(unittest.TestCase):
    def recorded_minkowski_ray(self, maximum_step: float = 1.0):
        return trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            options=RayTraceOptions(
                initial_step=maximum_step,
                maximum_step=maximum_step,
                maximum_affine_length=1.0,
                record_path=True,
            ),
        )

    def tight_options(self, **changes) -> AdaptiveScalarTransferOptions:
        defaults = dict(
            absolute_tolerance=1.0e-5,
            relative_tolerance=1.0e-5,
            optical_depth_absolute_tolerance=1.0e-5,
            optical_depth_relative_tolerance=1.0e-5,
            maximum_coefficient_evaluations=100_000,
            maximum_refinement_depth=24,
            minimum_affine_step=1.0e-10,
            sampling=RecordedPathSamplingOptions(
                maximum_reintegrations=100_000,
            ),
        )
        defaults.update(changes)
        return AdaptiveScalarTransferOptions(**defaults)

    def test_narrow_layer_missed_by_legacy_midpoint_is_adaptively_resolved(
        self,
    ) -> None:
        ray = self.recorded_minkowski_ray()
        medium = ThinLayerMedium(width=0.1, emissivity=2.0)
        legacy = propagate_recorded_ray_spectrum(
            ray,
            (230.0e9,),
            medium,
            DarkBoundary(),
        )
        adaptive = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            medium,
            DarkBoundary(),
            230.0e9,
            options=self.tight_options(),
        )

        self.assertEqual(legacy.transfers[0].stokes.i, 0.0)
        expected = 2.0 * (8.0 * medium.width / 15.0)
        self.assertAlmostEqual(
            adaptive.observer_invariant_intensity,
            expected,
            delta=3.0e-6,
        )
        self.assertGreater(adaptive.diagnostics.refined_intervals, 0)
        self.assertLessEqual(
            adaptive.diagnostics.estimated_global_absolute_error,
            adaptive.diagnostics.global_error_limit,
        )
        self.assertAlmostEqual(
            adaptive.diagnostics.global_error_limit,
            self.tight_options().absolute_tolerance
            + self.tight_options().relative_tolerance
            * adaptive.diagnostics.error_scale_invariant_intensity,
        )
        self.assertIn("finite-stencil", adaptive.diagnostics.capability)
        self.assertEqual(adaptive.diagnostics.capability, FINITE_STENCIL_CAPABILITY)

    def test_result_is_stable_for_one_or_one_hundred_recorded_segments(
        self,
    ) -> None:
        one = self.recorded_minkowski_ray(1.0)
        hundred = self.recorded_minkowski_ray(0.01)
        self.assertEqual(len(one.segments), 1)
        self.assertEqual(len(hundred.segments), 100)
        options = self.tight_options(
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-6,
        )
        one_result = propagate_adaptive_scalar_recorded_ray(
            one,
            MinkowskiMetric(),
            SmoothMedium(),
            DarkBoundary(),
            1.0,
            options=options,
        )
        hundred_result = propagate_adaptive_scalar_recorded_ray(
            hundred,
            MinkowskiMetric(),
            SmoothMedium(),
            DarkBoundary(),
            1.0,
            options=options,
        )

        expected = 4.0 / 3.0
        self.assertAlmostEqual(
            one_result.observer_invariant_intensity,
            expected,
            places=5,
        )
        self.assertAlmostEqual(
            hundred_result.observer_invariant_intensity,
            expected,
            places=5,
        )
        self.assertAlmostEqual(
            one_result.observer_invariant_intensity,
            hundred_result.observer_invariant_intensity,
            delta=2.0e-6,
        )
        self.assertLessEqual(
            hundred_result.diagnostics.estimated_global_absolute_error,
            hundred_result.diagnostics.global_error_limit,
        )

    def test_vacuum_and_absorbing_slab_match_exact_solutions(self) -> None:
        ray = self.recorded_minkowski_ray()
        vacuum = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            VacuumMedium(),
            ConstantBoundary(7.5),
            1.0,
        )
        self.assertEqual(vacuum.observer_invariant_intensity, 7.5)
        self.assertEqual(vacuum.diagnostics.scalar_optical_depth, 0.0)

        slab = ThinLayerMedium(width=0.1, absorption=3.0)
        absorbed = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            slab,
            ConstantBoundary(2.0),
            1.0,
            options=self.tight_options(),
        )
        expected_optical_depth = 3.0 * (8.0 * slab.width / 15.0)
        expected = 2.0 * math.exp(-expected_optical_depth)
        self.assertAlmostEqual(
            absorbed.observer_invariant_intensity,
            expected,
            delta=4.0e-6,
        )
        self.assertAlmostEqual(
            absorbed.diagnostics.scalar_optical_depth,
            expected_optical_depth,
            delta=3.0e-6,
        )

    def test_dark_absorber_requires_independent_optical_depth_convergence(
        self,
    ) -> None:
        ray = self.recorded_minkowski_ray()
        absorber = ThinLayerMedium(
            centre=0.5,
            width=0.5,
            absorption=20.0,
        )
        result = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            absorber,
            DarkBoundary(),
            1.0,
            options=self.tight_options(),
        )

        expected_optical_depth = 16.0 / 3.0
        self.assertEqual(result.observer_invariant_intensity, 0.0)
        self.assertGreater(result.diagnostics.refined_intervals, 0)
        self.assertAlmostEqual(
            result.diagnostics.scalar_optical_depth,
            expected_optical_depth,
            delta=3.0e-4,
        )
        self.assertLessEqual(
            result.diagnostics.estimated_global_optical_depth_error,
            result.diagnostics.optical_depth_global_error_limit,
        )
        self.assertLessEqual(
            result.diagnostics.maximum_local_optical_depth_error_norm,
            1.0,
        )

    def test_zero_length_ray_requires_current_metric_and_exact_counters(
        self,
    ) -> None:
        state = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        ray = trace_null_geodesic(
            MinkowskiMetric(),
            state,
            termination=RadialTermination(0.1, 2.0),
            options=RayTraceOptions(record_path=True),
        )
        self.assertEqual(ray.outcome, "captured")
        self.assertEqual(ray.segments, ())
        valid = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            VacuumMedium(),
            DarkBoundary(),
            1.0,
        )
        self.assertEqual(valid.diagnostics.recorded_segment_count, 0)
        self.assertEqual(
            valid.diagnostics.maximum_null_residual,
            ray.maximum_null_residual,
        )

        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "segment count",
        ):
            propagate_adaptive_scalar_recorded_ray(
                replace(ray, accepted_steps=17),
                MinkowskiMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )
        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "rejected_steps",
        ):
            propagate_adaptive_scalar_recorded_ray(
                replace(ray, rejected_steps=17),
                MinkowskiMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )
        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "zero-length ray terminal state",
        ):
            propagate_adaptive_scalar_recorded_ray(
                replace(
                    ray,
                    terminal_state=HamiltonianState(
                        event=state.event,
                        covector=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
                MinkowskiMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )
        with self.assertRaisesRegex(TypeError, "MetricProvider"):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MissingSampleMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )

    def test_physical_source_to_observer_order_is_not_reversed(self) -> None:
        ray = self.recorded_minkowski_ray()
        options = self.tight_options()
        hot_then_cold = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            OrderedMedium(hot_near_source=True),
            DarkBoundary(),
            1.0,
            options=options,
        )
        cold_then_hot = propagate_adaptive_scalar_recorded_ray(
            ray,
            MinkowskiMetric(),
            OrderedMedium(hot_near_source=False),
            DarkBoundary(),
            1.0,
            options=options,
        )

        expected_hot_then_cold = (
            3.0 * (1.0 - math.exp(-0.5)) * math.exp(-0.5)
        )
        self.assertAlmostEqual(
            hot_then_cold.observer_invariant_intensity,
            expected_hot_then_cold,
            delta=3.0e-7,
        )
        self.assertLess(
            hot_then_cold.observer_invariant_intensity,
            cold_then_hot.observer_invariant_intensity,
        )
        self.assertEqual(
            hot_then_cold.diagnostics.ordering,
            "source-to-observer",
        )

    def test_work_depth_minimum_step_and_nonfinite_fail_closed(self) -> None:
        ray = self.recorded_minkowski_ray()
        medium = ThinLayerMedium(width=0.1, emissivity=2.0)
        with self.assertRaisesRegex(
            AdaptiveMediumBudgetExceeded,
            "coefficient-evaluation budget",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                medium,
                DarkBoundary(),
                1.0,
                options=self.tight_options(maximum_coefficient_evaluations=2),
            )
        with self.assertRaisesRegex(
            AdaptiveMediumIntegrationError,
            "reintegration budget",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                medium,
                DarkBoundary(),
                1.0,
                options=self.tight_options(
                    sampling=RecordedPathSamplingOptions(
                        maximum_reintegrations=1,
                    ),
                ),
            )
        with self.assertRaisesRegex(
            AdaptiveMediumBudgetExceeded,
            "refinement-depth budget",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                medium,
                DarkBoundary(),
                1.0,
                options=self.tight_options(maximum_refinement_depth=0),
            )
        with self.assertRaisesRegex(
            AdaptiveMediumBudgetExceeded,
            "minimum affine step",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                medium,
                DarkBoundary(),
                1.0,
                options=self.tight_options(minimum_affine_step=0.3),
            )
        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "coefficient evaluation failed",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                InvalidMedium("nan"),
                DarkBoundary(),
                1.0,
            )

        with self.assertRaisesRegex(
            AdaptiveMediumBudgetExceeded,
            "refinement-depth budget",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                DiscontinuousLayerMedium(),
                DarkBoundary(),
                1.0,
                options=self.tight_options(maximum_refinement_depth=12),
            )

    def test_scalar_contract_rejects_polarization(self) -> None:
        ray = self.recorded_minkowski_ray()
        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "Faraday",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                InvalidMedium("polarized"),
                DarkBoundary(),
                1.0,
            )
        with self.assertRaisesRegex(
            AdaptiveMediumValidationError,
            "polarization",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                MinkowskiMetric(),
                VacuumMedium(),
                PolarizedBoundary(),
                1.0,
            )

    def test_record_tampering_and_wrong_metric_reintegration_are_rejected(
        self,
    ) -> None:
        ray = self.recorded_minkowski_ray()
        segment = ray.segments[0]
        corrupted_midpoint = HamiltonianState(
            event=(
                segment.midpoint.event[0],
                segment.midpoint.event[1] + 1.0e-4,
                segment.midpoint.event[2],
                segment.midpoint.event[3],
            ),
            covector=segment.midpoint.covector,
        )
        corrupted_ray = replace(
            ray,
            segments=(replace(segment, midpoint=corrupted_midpoint),),
        )
        with self.assertRaisesRegex(
            AdaptiveMediumIntegrationError,
            "Hamiltonian certification",
        ):
            propagate_adaptive_scalar_recorded_ray(
                corrupted_ray,
                MinkowskiMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )
        with self.assertRaisesRegex(
            AdaptiveMediumIntegrationError,
            "Hamiltonian certification",
        ):
            propagate_adaptive_scalar_recorded_ray(
                ray,
                WrongFlatMetric(),
                VacuumMedium(),
                DarkBoundary(),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
