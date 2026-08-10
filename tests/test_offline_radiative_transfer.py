from __future__ import annotations

import math
import unittest

from offline.radiative_transfer import (
    StepBudgetExceeded,
    StokesInvariant,
    TransferCoefficients,
    TransferIntegrationError,
    TransferSegment,
    TransferValidationError,
    propagate_source_to_observer,
)


class OfflineRadiativeTransferTests(unittest.TestCase):
    def test_vacuum_conserves_invariant_stokes_exactly(self) -> None:
        source = StokesInvariant(3.0, 0.4, -0.3, 0.2)
        zero_length_extreme = TransferCoefficients(
            invariant_faraday=(1.0e308, -1.0e308, 1.0e308),
        )
        result = propagate_source_to_observer(
            source,
            [
                TransferSegment(0.0, zero_length_extreme),
                TransferSegment(2.5),
                TransferSegment(100.0),
            ],
        )

        self.assertEqual(result.stokes, source)
        self.assertEqual(result.diagnostics.ordering, "source-to-observer")
        self.assertEqual(result.diagnostics.segment_count, 3)
        self.assertEqual(result.diagnostics.zero_length_segments, 1)
        self.assertEqual(result.diagnostics.exact_slab_segments, 2)
        self.assertEqual(result.diagnostics.implicit_midpoint_steps, 0)

    def test_lte_homogeneous_slab_uses_exact_formal_solution(self) -> None:
        absorption = 0.7
        source_function = 4.2
        length = 2.3
        incoming = StokesInvariant(0.35, 0.08, -0.04, 0.02)
        coefficients = TransferCoefficients(
            invariant_emissivity=StokesInvariant(
                absorption * source_function,
                0.0,
                0.0,
                0.0,
            ),
            invariant_absorption=absorption,
        )

        result = propagate_source_to_observer(
            incoming,
            [TransferSegment(length, coefficients)],
        )
        attenuation = math.exp(-absorption * length)
        expected_i = (
            incoming.i * attenuation
            + source_function * (1.0 - attenuation)
        )

        self.assertAlmostEqual(result.stokes.i, expected_i, places=14)
        self.assertAlmostEqual(result.stokes.q, incoming.q * attenuation, places=14)
        self.assertAlmostEqual(result.stokes.u, incoming.u * attenuation, places=14)
        self.assertAlmostEqual(result.stokes.v, incoming.v * attenuation, places=14)
        self.assertEqual(result.diagnostics.exact_slab_segments, 1)
        self.assertEqual(result.diagnostics.implicit_midpoint_segments, 0)

    def test_exact_slab_is_invariant_under_path_segmentation(self) -> None:
        source = StokesInvariant(0.8, 0.1, 0.02, -0.03)
        coefficients = TransferCoefficients(
            invariant_emissivity=StokesInvariant(0.9, 0.08, 0.0, 0.0),
            invariant_absorption=0.45,
        )
        whole = propagate_source_to_observer(
            source,
            [TransferSegment(5.0, coefficients)],
        )
        split = propagate_source_to_observer(
            source,
            [TransferSegment(0.125, coefficients) for _ in range(40)],
        )

        for whole_value, split_value in zip(
            whole.stokes.as_tuple(),
            split.stokes.as_tuple(),
        ):
            self.assertAlmostEqual(whole_value, split_value, places=13)
        self.assertEqual(whole.diagnostics.scalar_optical_depth, 2.25)
        self.assertAlmostEqual(split.diagnostics.scalar_optical_depth, 2.25)

    def test_subnormal_optical_depth_uses_the_exact_transparent_limit(self) -> None:
        minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
        result = propagate_source_to_observer(
            StokesInvariant(),
            [
                TransferSegment(
                    0.5,
                    TransferCoefficients(
                        invariant_emissivity=StokesInvariant(1.0),
                        invariant_absorption=minimum_subnormal,
                    ),
                )
            ],
        )
        self.assertEqual(result.stokes.i, 0.5)

    def test_segments_are_applied_source_to_observer_without_reordering(self) -> None:
        source = StokesInvariant()
        hot = TransferCoefficients(
            invariant_emissivity=StokesInvariant(3.0, 0.0, 0.0, 0.0),
            invariant_absorption=1.0,
        )
        cold = TransferCoefficients(invariant_absorption=1.0)
        hot_then_cold = propagate_source_to_observer(
            source,
            [TransferSegment(1.0, hot), TransferSegment(1.0, cold)],
        )
        cold_then_hot = propagate_source_to_observer(
            source,
            [TransferSegment(1.0, cold), TransferSegment(1.0, hot)],
        )

        self.assertLess(hot_then_cold.stokes.i, cold_then_hot.stokes.i)
        expected = 3.0 * (1.0 - math.exp(-1.0)) * math.exp(-1.0)
        self.assertAlmostEqual(hot_then_cold.stokes.i, expected, places=14)

    def test_pure_faraday_transfer_preserves_norm_and_matches_rotation_phase(self) -> None:
        source = StokesInvariant(1.5, 0.7, -0.4, 0.2)
        coefficients = TransferCoefficients(
            invariant_faraday=(0.35, -0.2, 0.9),
        )
        result = propagate_source_to_observer(
            source,
            [TransferSegment(7.0, coefficients)],
            maximum_step_matrix_norm=0.05,
        )

        self.assertAlmostEqual(result.stokes.i, source.i, places=13)
        self.assertAlmostEqual(
            result.stokes.polarization_norm,
            source.polarization_norm,
            places=11,
        )
        self.assertEqual(result.diagnostics.implicit_midpoint_segments, 1)
        self.assertGreater(result.diagnostics.implicit_midpoint_steps, 1)
        self.assertLessEqual(
            result.diagnostics.maximum_step_matrix_norm,
            0.05 + 1.0e-15,
        )
        self.assertLessEqual(
            result.diagnostics.maximum_convergence_error_norm,
            1.0,
        )

        phase_source = StokesInvariant(1.0, 1.0, 0.0, 0.0)
        phase_length = 20.0
        phase = propagate_source_to_observer(
            phase_source,
            [
                TransferSegment(
                    phase_length,
                    TransferCoefficients(
                        invariant_faraday=(0.0, 0.0, 1.0),
                    ),
                )
            ],
        )
        self.assertAlmostEqual(
            phase.stokes.q,
            math.cos(phase_length),
            delta=1.0e-5,
        )
        self.assertAlmostEqual(
            phase.stokes.u,
            math.sin(phase_length),
            delta=1.0e-5,
        )

    def test_nonfinite_and_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(TransferValidationError, "must be finite"):
            StokesInvariant(math.nan, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(TransferValidationError, "must be finite"):
            TransferCoefficients(invariant_absorption=math.inf)
        with self.assertRaisesRegex(TransferValidationError, "may not exceed"):
            StokesInvariant(0.0, 1.0e-15, 0.0, 0.0)
        with self.assertRaisesRegex(TransferValidationError, "passive transfer"):
            TransferCoefficients(
                invariant_dichroism=(1.0e-15, 0.0, 0.0),
            )
        with self.assertRaisesRegex(TransferValidationError, "non-negative"):
            TransferSegment(-1.0)
        with self.assertRaisesRegex(TransferValidationError, "must be finite"):
            TransferSegment(math.nan)
        with self.assertRaisesRegex(TransferValidationError, "positive integer"):
            propagate_source_to_observer(StokesInvariant(), [], maximum_steps=0)
        with self.assertRaisesRegex(TransferValidationError, "must be positive"):
            propagate_source_to_observer(
                StokesInvariant(),
                [],
                maximum_step_matrix_norm=0.0,
            )

        overflowing = TransferCoefficients(
            invariant_emissivity=StokesInvariant(1.0e308, 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(TransferIntegrationError, "non-finite"):
            propagate_source_to_observer(
                StokesInvariant(),
                [TransferSegment(2.0, overflowing)],
            )

    def test_step_budget_is_checked_before_coupled_integration(self) -> None:
        coefficients = TransferCoefficients(
            invariant_faraday=(0.0, 0.0, 100.0),
        )
        with self.assertRaisesRegex(StepBudgetExceeded, "needs"):
            propagate_source_to_observer(
                StokesInvariant(1.0, 0.3, 0.0, 0.0),
                [TransferSegment(10.0, coefficients)],
                maximum_steps=20,
                maximum_step_matrix_norm=0.1,
            )

        extreme = TransferCoefficients(
            invariant_faraday=(1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(StepBudgetExceeded, "needs"):
            propagate_source_to_observer(
                StokesInvariant(1.0, 0.3, 0.0, 0.0),
                [TransferSegment(1.0e308, extreme)],
                maximum_steps=10,
                maximum_step_matrix_norm=1.0e-308,
            )

    def test_local_convergence_rejects_whole_turn_faraday_alias(self) -> None:
        alias_length = 1627.447970235928
        with self.assertRaisesRegex(StepBudgetExceeded, "local transfer"):
            propagate_source_to_observer(
                StokesInvariant(1.0, 1.0, 0.0, 0.0),
                [
                    TransferSegment(
                        alias_length,
                        TransferCoefficients(
                            invariant_faraday=(0.0, 0.0, 1.0),
                        ),
                    )
                ],
                maximum_steps=25_000,
            )

    def test_coupled_tolerance_is_invariant_under_path_segmentation(self) -> None:
        source = StokesInvariant(1.0, 1.0, 0.0, 0.0)
        coefficients = TransferCoefficients(
            invariant_faraday=(0.0, 0.0, 1.0),
        )
        whole = propagate_source_to_observer(
            source,
            [TransferSegment(20.0, coefficients)],
            maximum_steps=500_000,
        )
        split = propagate_source_to_observer(
            source,
            [TransferSegment(0.2, coefficients) for _ in range(100)],
            maximum_steps=500_000,
        )
        for whole_value, split_value in zip(
            whole.stokes.as_tuple(),
            split.stokes.as_tuple(),
        ):
            self.assertAlmostEqual(whole_value, split_value, delta=1.0e-5)
        self.assertAlmostEqual(whole.stokes.q, math.cos(20.0), delta=1.0e-5)
        self.assertAlmostEqual(whole.stokes.u, math.sin(20.0), delta=1.0e-5)
        self.assertAlmostEqual(split.stokes.q, math.cos(20.0), delta=1.0e-5)
        self.assertAlmostEqual(split.stokes.u, math.sin(20.0), delta=1.0e-5)


if __name__ == "__main__":
    unittest.main()
