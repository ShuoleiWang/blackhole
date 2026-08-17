from __future__ import annotations

import dataclasses
import json
import math
import unittest
from unittest.mock import patch

from offline.returning_radiation import (
    SCIENTIFIC_STATUS,
    AnnulusPhotonFateTable,
    AxisymmetricReturningRadiationKernel,
    PhotonFateProbabilityTriple,
    ReturningRadiationConvergenceError,
    ReturningRadiationFixedPointPolicy,
    ReturningRadiationVerificationError,
    solve_absorbed_returning_radiation,
    validate_returning_radiation_solution,
    verify_returning_radiation_solution,
)


def _kernel(
    coefficients: tuple[tuple[float, ...], ...],
) -> AxisymmetricReturningRadiationKernel:
    return AxisymmetricReturningRadiationKernel(
        annulus_radii_over_mass=tuple(
            6.0 + 4.0 * index for index in range(len(coefficients))
        ),
        receiver_emitter_coefficients=coefficients,
        ray_kernel_producer_id="unit-test-external-ray-kernel/v1",
    )


class ReturningRadiationKernelTests(unittest.TestCase):
    def test_scientific_boundary_is_explicit(self) -> None:
        self.assertIn("receiver-centred", SCIENTIFIC_STATUS["classification"])
        self.assertEqual(
            SCIENTIFIC_STATUS["equation"],
            "F_out = F_0 + F_in; F_in = K F_out",
        )
        self.assertIn("g^4", SCIENTIFIC_STATUS["kernelCoefficientSemantics"])
        for key in (
            "isIndependentRayKernel",
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesScattering",
            "includesPolarization",
            "includesFiniteThickness",
            "isGeneralRelativisticMagnetohydrodynamics",
            "isCompleteKerrbb",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("complete KERRBB", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isCompleteKerrbb"] = True

    def test_kernel_descriptor_is_canonical_stable_and_receiver_first(self) -> None:
        first = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=[6, 10],
            receiver_emitter_coefficients=[[0.1, 0.2], [0.05, 0.1]],
            ray_kernel_producer_id="ray-producer/v4",
        )
        second = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(6.0, 10.0),
            receiver_emitter_coefficients=((0.1, 0.2), (0.05, 0.1)),
            ray_kernel_producer_id="ray-producer/v4",
        )
        self.assertEqual(
            first.canonical_descriptor_json,
            second.canonical_descriptor_json,
        )
        self.assertEqual(
            first.canonical_descriptor_sha256,
            second.canonical_descriptor_sha256,
        )
        descriptor = json.loads(first.canonical_descriptor_json)
        self.assertEqual(
            descriptor["coefficientIndexOrder"],
            "K[receiverAnnulus][emitterAnnulus]",
        )
        self.assertFalse(descriptor["isIndependentRayKernel"])
        self.assertIn(
            "discretised emitting-annulus weight",
            descriptor["coefficientSemantics"],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.ray_kernel_producer_id = "forged"

    def test_zero_kernel_is_exact_identity(self) -> None:
        kernel = _kernel(((0.0, 0.0), (0.0, 0.0)))
        solution = solve_absorbed_returning_radiation(kernel, (3.5, 0.25))
        self.assertEqual(solution.intrinsic_flux, (3.5, 0.25))
        self.assertEqual(solution.incident_returning_flux, (0.0, 0.0))
        self.assertEqual(solution.outgoing_flux, (3.5, 0.25))
        self.assertEqual(solution.equation_residual, (0.0, 0.0))
        self.assertEqual(solution.iterations, 0)
        self.assertTrue(solution.monotonic_fixed_point_verified)
        verify_returning_radiation_solution(kernel, (3.5, 0.25), solution)

    def test_two_by_two_matches_analytic_inverse(self) -> None:
        # (I-K)^-1 (1,2) for K below is exactly (1.625, 2.3125).
        kernel = _kernel(((0.1, 0.2), (0.05, 0.1)))
        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=512,
            absolute_residual_tolerance=0.0,
            relative_residual_tolerance=1.0e-14,
        )
        solution = solve_absorbed_returning_radiation(
            kernel,
            (1.0, 2.0),
            policy,
        )
        self.assertTrue(
            math.isclose(solution.outgoing_flux[0], 1.625, rel_tol=1.0e-14)
        )
        self.assertTrue(
            math.isclose(solution.outgoing_flux[1], 2.3125, rel_tol=1.0e-14)
        )
        for residual, tolerance in zip(
            solution.equation_residual,
            solution.residual_tolerances,
        ):
            self.assertLessEqual(abs(residual), tolerance)
        self.assertGreater(solution.iterations, 0)
        verify_returning_radiation_solution(
            kernel,
            (1.0, 2.0),
            solution,
            policy,
        )

    def test_linearity_under_positive_flux_rescaling(self) -> None:
        kernel = _kernel(((0.08, 0.03), (0.02, 0.12)))
        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=512,
            relative_residual_tolerance=1.0e-13,
        )
        base = solve_absorbed_returning_radiation(
            kernel,
            (1.25, 0.75),
            policy,
        )
        scale = 17.0
        scaled = solve_absorbed_returning_radiation(
            kernel,
            (scale * 1.25, scale * 0.75),
            policy,
        )
        self.assertEqual(base.iterations, scaled.iterations)
        for base_value, scaled_value in zip(
            base.outgoing_flux,
            scaled.outgoing_flux,
        ):
            self.assertTrue(
                math.isclose(
                    scaled_value,
                    scale * base_value,
                    rel_tol=4.0e-15,
                )
            )

    def test_nonnegative_kernel_preserves_source_monotonicity(self) -> None:
        kernel = _kernel(((0.15, 0.02), (0.1, 0.05)))
        smaller = solve_absorbed_returning_radiation(kernel, (0.5, 1.0))
        larger = solve_absorbed_returning_radiation(kernel, (0.75, 3.0))
        self.assertTrue(smaller.monotonic_fixed_point_verified)
        self.assertTrue(larger.monotonic_fixed_point_verified)
        self.assertTrue(
            all(
                small <= large
                for small, large in zip(
                    smaller.outgoing_flux,
                    larger.outgoing_flux,
                )
            )
        )

    def test_subcritical_certificate_is_not_a_maximum_row_sum_shortcut(self) -> None:
        # This nilpotent K has a maximum row sum of two but K**2=0, hence its
        # spectral radius is zero and the fixed point terminates exactly.
        kernel = _kernel(((0.0, 2.0), (0.0, 0.0)))
        solution = solve_absorbed_returning_radiation(kernel, (1.0, 3.0))
        self.assertEqual(solution.outgoing_flux, (7.0, 3.0))
        self.assertEqual(solution.incident_returning_flux, (6.0, 0.0))
        self.assertEqual(solution.iterations, 1)

    def test_near_critical_kernel_exhausts_fixed_point_budget(self) -> None:
        kernel = _kernel(((0.9999,),))
        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=8,
            relative_residual_tolerance=1.0e-12,
        )
        with self.assertRaisesRegex(
            ReturningRadiationConvergenceError,
            "fixed-point residual did not converge",
        ):
            solve_absorbed_returning_radiation(kernel, (1.0,), policy)

    def test_critical_and_supercritical_kernels_fail_closed(self) -> None:
        policy = ReturningRadiationFixedPointPolicy(maximum_iterations=12)
        for coefficient in (1.0, 1.01):
            with self.subTest(coefficient=coefficient):
                with self.assertRaisesRegex(
                    ReturningRadiationConvergenceError,
                    "subcriticality was not certified",
                ):
                    solve_absorbed_returning_radiation(
                        _kernel(((coefficient,),)),
                        (1.0,),
                        policy,
                    )
                # A zero source must not disguise a non-unique or unstable
                # operator as a certified converged physical system.
                with self.assertRaises(ReturningRadiationConvergenceError):
                    solve_absorbed_returning_radiation(
                        _kernel(((coefficient,),)),
                        (0.0,),
                        policy,
                    )

    def test_shape_order_and_numeric_gates(self) -> None:
        valid_arguments = {
            "annulus_radii_over_mass": (6.0, 10.0),
            "receiver_emitter_coefficients": ((0.1, 0.2), (0.3, 0.1)),
            "ray_kernel_producer_id": "test",
        }
        invalid_overrides = (
            {"annulus_radii_over_mass": ()},
            {"annulus_radii_over_mass": (6.0, 6.0)},
            {"annulus_radii_over_mass": (10.0, 6.0)},
            {"annulus_radii_over_mass": (6.0, math.nan)},
            {"receiver_emitter_coefficients": ((0.1, 0.2),)},
            {"receiver_emitter_coefficients": ((0.1,), (0.2,))},
            {"receiver_emitter_coefficients": ((0.1, -0.2), (0.3, 0.1))},
            {"receiver_emitter_coefficients": ((0.1, math.nan), (0.3, 0.1))},
            {"ray_kernel_producer_id": ""},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                arguments = dict(valid_arguments)
                arguments.update(override)
                with self.assertRaises((TypeError, ValueError)):
                    AxisymmetricReturningRadiationKernel(**arguments)

        kernel = _kernel(((0.1, 0.0), (0.0, 0.1)))
        for source in ((1.0,), (1.0, -1.0), (1.0, math.nan)):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    solve_absorbed_returning_radiation(kernel, source)
        with self.assertRaises(ValueError):
            ReturningRadiationFixedPointPolicy(maximum_iterations=0)
        with self.assertRaises(ValueError):
            ReturningRadiationFixedPointPolicy(
                absolute_residual_tolerance=0.0,
                relative_residual_tolerance=0.0,
            )

    def test_changed_solution_is_rejected_by_deterministic_verifier(self) -> None:
        kernel = _kernel(((0.1, 0.02), (0.01, 0.15)))
        source = (2.0, 1.0)
        solution = solve_absorbed_returning_radiation(kernel, source)
        changed_flux = dataclasses.replace(
            solution,
            outgoing_flux=(
                solution.outgoing_flux[0] + 1.0e-6,
                solution.outgoing_flux[1],
            ),
        )
        changed_iterations = dataclasses.replace(
            solution,
            iterations=solution.iterations + 1,
        )
        changed_kernel_binding = dataclasses.replace(
            solution,
            kernel_descriptor_sha256="0" * 64,
        )
        for changed in (
            changed_flux,
            changed_iterations,
            changed_kernel_binding,
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ReturningRadiationVerificationError):
                    verify_returning_radiation_solution(kernel, source, changed)
        with self.assertRaises(ReturningRadiationVerificationError):
            verify_returning_radiation_solution(
                kernel,
                (source[0] + 1.0, source[1]),
                solution,
            )

    def test_algebraic_validator_does_not_repeat_fixed_point_iteration(self) -> None:
        kernel = _kernel(((0.1, 0.02), (0.01, 0.15)))
        source = (2.0, 1.0)
        solution = solve_absorbed_returning_radiation(kernel, source)
        with patch(
            "offline.returning_radiation.solve_absorbed_returning_radiation",
            side_effect=AssertionError("algebraic validation must not solve"),
        ):
            validate_returning_radiation_solution(kernel, source, solution)
        changed = dataclasses.replace(
            solution,
            incident_returning_flux=(
                solution.incident_returning_flux[0] + 1.0e-9,
                solution.incident_returning_flux[1],
            ),
        )
        with self.assertRaises(ReturningRadiationVerificationError):
            validate_returning_radiation_solution(kernel, source, changed)

    def test_deterministic_verifier_replays_fixed_point_exactly_once(self) -> None:
        kernel = _kernel(((0.1, 0.02), (0.01, 0.15)))
        source = (2.0, 1.0)
        solution = solve_absorbed_returning_radiation(kernel, source)
        original = solve_absorbed_returning_radiation
        with patch(
            "offline.returning_radiation.solve_absorbed_returning_radiation",
            wraps=original,
        ) as solver:
            verify_returning_radiation_solution(kernel, source, solution)
        self.assertEqual(solver.call_count, 1)

    def test_binary64_sign_and_live_kernel_identity_are_exact(self) -> None:
        zero_kernel = _kernel(((0.0,),))
        zero_solution = solve_absorbed_returning_radiation(
            zero_kernel,
            (0.0,),
        )
        object.__setattr__(
            zero_solution,
            "incident_returning_flux",
            (-0.0,),
        )
        with self.assertRaises(ReturningRadiationVerificationError):
            validate_returning_radiation_solution(
                zero_kernel,
                (0.0,),
                zero_solution,
            )
        with self.assertRaises(ReturningRadiationVerificationError):
            verify_returning_radiation_solution(
                zero_kernel,
                (0.0,),
                zero_solution,
            )

        stale_kernel = _kernel(((0.0,),))
        object.__setattr__(
            stale_kernel,
            "receiver_emitter_coefficients",
            ((0.5,),),
        )
        with self.assertRaisesRegex(
            ReturningRadiationVerificationError,
            "descriptor are stale",
        ):
            solve_absorbed_returning_radiation(stale_kernel, (1.0,))

    def test_policy_numeric_subclass_cannot_change_solver_arithmetic(self) -> None:
        class LooseTolerance(float):
            def __mul__(self, other):
                del other
                return 1.0e300

        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=100,
            absolute_residual_tolerance=0.0,
            relative_residual_tolerance=1.0e-12,
        )
        object.__setattr__(
            policy,
            "relative_residual_tolerance",
            LooseTolerance(1.0e-12),
        )
        with self.assertRaisesRegex(
            ReturningRadiationVerificationError,
            "exact canonical types",
        ):
            solve_absorbed_returning_radiation(
                _kernel(((0.5,),)),
                (1.0,),
                policy,
            )


class PhotonFateProbabilityTests(unittest.TestCase):
    def test_probability_triple_sums_to_one_and_stays_separate(self) -> None:
        fate = PhotonFateProbabilityTriple(0.27, 0.04, 0.69)
        self.assertEqual(fate.probability_sum, 1.0)
        table = AnnulusPhotonFateTable(
            annulus_radii_over_mass=(6.0,),
            probability_triples=(fate,),
            ray_fate_producer_id="fate-ray-tracer/v1",
        )
        descriptor = table.canonical_descriptor()
        self.assertIs(descriptor["isLocalEnergyFluxKernel"], False)
        self.assertEqual(
            descriptor["probabilityOrder"],
            ["return", "capture", "escape"],
        )
        self.assertEqual(len(table.canonical_descriptor_sha256), 64)

    def test_probability_gates_reject_invalid_or_wrong_sum(self) -> None:
        for values in (
            (-0.1, 0.2, 0.9),
            (0.2, 0.2, 0.2),
            (1.1, 0.0, -0.1),
            (math.nan, 0.5, 0.5),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    PhotonFateProbabilityTriple(*values)
        fate = PhotonFateProbabilityTriple(0.1, 0.2, 0.7)
        with self.assertRaises(ValueError):
            AnnulusPhotonFateTable(
                annulus_radii_over_mass=(6.0, 10.0),
                probability_triples=(fate,),
                ray_fate_producer_id="test",
            )


if __name__ == "__main__":
    unittest.main()
