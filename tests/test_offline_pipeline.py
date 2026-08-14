from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from dataclasses import replace

from offline.geodesic import HamiltonianState, RayTraceOptions, trace_null_geodesic
from offline.pipeline import (
    DarkBoundary,
    VacuumMedium,
    past_directed_comoving_frequency_ratio,
    propagate_recorded_ray_spectrum,
    source_to_observer_segments,
)
from offline.radiative_transfer import (
    StokesInvariant,
    TransferCoefficients,
)
from offline.spacetime import MinkowskiMetric


@dataclass(frozen=True)
class LayeredMedium:
    source_id: str = "test-layered-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        if state.event[1] >= 1.0:
            return TransferCoefficients(
                invariant_emissivity=StokesInvariant(3.0, 0.0, 0.0, 0.0),
                invariant_absorption=1.0,
            )
        return TransferCoefficients(invariant_absorption=1.0)


@dataclass(frozen=True)
class PolarizedMedium:
    source_id: str = "test-polarized-medium"

    def coefficients(
        self,
        state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> TransferCoefficients:
        return TransferCoefficients(invariant_faraday=(0.0, 0.0, 1.0))


@dataclass
class CountingBrightBoundary:
    source_id: str = "test-counting-bright-boundary"
    calls: int = 0

    def invariant_stokes(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
    ) -> StokesInvariant:
        self.calls += 1
        return StokesInvariant(10.0)


class OfflinePipelineTests(unittest.TestCase):
    def recorded_minkowski_ray(self):
        return trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            options=RayTraceOptions(
                initial_step=1.0,
                maximum_step=1.0,
                maximum_affine_length=2.0,
                record_path=True,
            ),
        )

    def test_observer_traced_path_is_explicitly_reversed_for_transfer(self) -> None:
        ray = self.recorded_minkowski_ray()
        self.assertLess(ray.segments[0].midpoint.event[1], 1.0)
        self.assertGreater(ray.segments[-1].midpoint.event[1], 1.0)
        transfer_segments = source_to_observer_segments(
            ray,
            LayeredMedium(),
            230.0e9,
        )
        self.assertEqual(
            transfer_segments[0].coefficients.invariant_emissivity.i,
            3.0,
        )
        self.assertEqual(
            transfer_segments[-1].coefficients.invariant_emissivity.i,
            0.0,
        )

        result = propagate_recorded_ray_spectrum(
            ray,
            (230.0e9,),
            LayeredMedium(),
            DarkBoundary(),
        )
        expected = 3.0 * (1.0 - math.exp(-1.0)) * math.exp(-1.0)
        self.assertAlmostEqual(result.transfers[0].stokes.i, expected, places=13)

    def test_comoving_frequency_uses_past_directed_sign(self) -> None:
        state = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(2.0, 1.0, 0.0, 0.0),
        )
        self.assertEqual(
            past_directed_comoving_frequency_ratio(
                state,
                (1.0, 0.0, 0.0, 0.0),
            ),
            2.0,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            past_directed_comoving_frequency_ratio(
                state,
                (-1.0, 0.0, 0.0, 0.0),
            )

    def test_unrecorded_path_and_unsorted_frequencies_fail_closed(self) -> None:
        unrecorded = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            options=RayTraceOptions(maximum_affine_length=1.0),
        )
        with self.assertRaisesRegex(ValueError, "not recorded"):
            source_to_observer_segments(
                unrecorded,
                LayeredMedium(),
                230.0e9,
            )

    def test_failed_incomplete_and_unframed_polarized_paths_fail_closed(self) -> None:
        ray = self.recorded_minkowski_ray()
        with self.assertRaisesRegex(ValueError, "not usable"):
            source_to_observer_segments(
                replace(
                    ray,
                    outcome="integrator-failure",
                    failure_reason="injected",
                ),
                LayeredMedium(),
                230.0e9,
            )
        with self.assertRaisesRegex(ValueError, "segments do not cover"):
            source_to_observer_segments(
                replace(ray, affine_length=ray.affine_length + 0.5),
                LayeredMedium(),
                230.0e9,
            )
        with self.assertRaisesRegex(ValueError, "parallel-transported"):
            source_to_observer_segments(
                ray,
                PolarizedMedium(),
                230.0e9,
            )

    def test_captured_ray_requires_target_and_forces_a_dark_horizon(self) -> None:
        ray = self.recorded_minkowski_ray()
        with self.assertRaisesRegex(ValueError, "termination target"):
            source_to_observer_segments(
                replace(ray, outcome="captured"),
                LayeredMedium(),
                230.0e9,
            )

        captured = replace(
            ray,
            outcome="captured",
            terminal_target_id="analytic-capture-sphere",
        )
        boundary = CountingBrightBoundary()
        result = propagate_recorded_ray_spectrum(
            captured,
            (230.0e9,),
            VacuumMedium(),
            boundary,
        )
        self.assertEqual(boundary.calls, 0)
        self.assertEqual(result.transfers[0].stokes, StokesInvariant())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            propagate_recorded_ray_spectrum(
                self.recorded_minkowski_ray(),
                (230.0e9, 100.0e9),
                LayeredMedium(),
                DarkBoundary(),
            )


if __name__ == "__main__":
    unittest.main()
