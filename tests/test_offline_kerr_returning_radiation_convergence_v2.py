from __future__ import annotations

import hashlib
import math
import unittest
from unittest.mock import patch

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import StationaryKerrFiniteThicknessCalibration
from offline.kerr_finite_thickness_area import KerrFiniteThicknessAreaQuadraturePolicy
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_kernel import KerrReturningRadiationKernelPolicy

from offline.kerr_returning_radiation_convergence_v2 import (
    SCIENTIFIC_STATUS,
    KerrReturningRadiationConvergenceV2Policy,
    KerrReturningRadiationConvergenceV2VerificationError,
    KerrReturningRadiationGridComparisonV2,
    KerrReturningRadiationGridSummaryV2,
    KerrAuthenticatedReturningRadiationConvergenceV2,
    authenticate_direct_kerr_returning_radiation_convergence_v2,
    compare_kerr_returning_radiation_grids_v2,
)


_COMMON_FATES = ((0.125, 0.0, 0.125, 0.75, 0.0),)


def _summary(
    grid_id: str,
    coefficients: tuple[tuple[float, ...], ...],
    g2_columns: tuple[float, ...],
    fate_fractions: tuple[tuple[float, ...], ...] = _COMMON_FATES,
    *,
    receiver_areas: tuple[float, ...] | None = None,
    emitter_areas: tuple[float, ...] = (1.0,),
    maximum_weight: float = 5.0e-4,
) -> KerrReturningRadiationGridSummaryV2:
    selected_receiver_areas = (
        (1.0,) * len(coefficients)
        if receiver_areas is None
        else receiver_areas
    )
    return KerrReturningRadiationGridSummaryV2(
        grid_id,
        selected_receiver_areas,
        emitter_areas,
        coefficients,
        g2_columns,
        fate_fractions,
        maximum_weight,
    )


def _v1_scalar_gate(first: float, second: float) -> bool:
    difference = abs(first - second)
    threshold = max(0.02, 0.05 * max(abs(first), abs(second)))
    return difference <= threshold


class OfflineKerrReturningRadiationConvergenceV2Tests(unittest.TestCase):
    @staticmethod
    def authenticated_direct_source():
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=8.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric, escape_radius_m=20.0, offset_m=0.02
        )
        policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=0.25,
            relative_tolerance=0.25,
            symmetry_absolute_tolerance=0.25,
            symmetry_relative_tolerance=0.25,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=20_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )

        def classifier(*arguments):
            source_face = arguments[6]
            source_radius = arguments[7]
            tangent = arguments[9]
            receiver_face = "upper" if tangent < math.pi else "lower"
            identity = hashlib.sha256(repr(arguments[6:]).encode()).hexdigest()
            return forward_module._DirectionTransport(
                f"return-{receiver_face}",
                receiver_face,
                source_radius,
                2.0,
                4.0,
                identity,
                receiver_face,
                source_radius,
            )

        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            return forward_module.integrate_kerr_returning_radiation_energy_kernel(
                surface,
                termination=termination,
                annulus_edges_over_mass=(
                    float(calibration.isco_radius_over_mass),
                    float(calibration.outer_radius_over_mass),
                ),
                ray_options=RayTraceOptions(),
                surface_options=SurfaceEventOptions(subdivisions_per_segment=4),
                policy=policy,
                area_policy=area_policy,
            ), classifier

    def synthetic_passing_pair(self):
        fine = _summary(
            "synthetic-fine",
            ((0.1,), (0.1,)),
            (0.2,),
            _COMMON_FATES,
        )
        comparison = _summary(
            "synthetic-comparison",
            ((0.0995,), (0.0995,)),
            (0.199,),
            ((0.125244140625, 0.0, 0.125, 0.749755859375, 0.0),),
        )
        return fine, comparison

    def test_synthetic_scale_aware_comparison_passes_and_revalidates(self) -> None:
        fine, comparison = self.synthetic_passing_pair()
        result = compare_kerr_returning_radiation_grids_v2(fine, comparison)

        self.assertTrue(result.resolution_qualified)
        self.assertTrue(result.converged)
        self.assertTrue(result.columns[0].converged)
        self.assertTrue(result.fates[0].converged)
        self.assertLessEqual(
            result.columns[0].g2_relative_difference,
            1.0e-2,
        )
        self.assertLessEqual(
            result.columns[0].column_normalized_l1_difference,
            1.0e-2,
        )
        self.assertEqual(result.columns[0].support_flip_count, 0)
        self.assertEqual(result.fates[0].support_flip_count, 0)
        result.revalidate()
        fine.revalidate()
        comparison.revalidate()
        result.policy.revalidate()
        self.assertEqual(
            result.model_descriptor_sha256,
            hashlib.sha256(
                result.canonical_descriptor_json.encode("utf-8")
            ).hexdigest(),
        )
        descriptor = result.descriptor()
        self.assertFalse(descriptor["capabilities"]["rigorousContinuumErrorBound"])
        self.assertFalse(descriptor["capabilities"]["isIndependentPhysicsOracle"])
        self.assertFalse(SCIENTIFIC_STATUS["rigorousContinuumErrorBound"])
        self.assertTrue(SCIENTIFIC_STATUS["acceptsCallerConstructedGridSummaries"])
        self.assertFalse(SCIENTIFIC_STATUS["authenticatesUpstreamKernelProvenance"])
        self.assertTrue(
            SCIENTIFIC_STATUS["requiresAuthenticatedAdapterBeforeProductCertification"]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS["requiresSameOrderedReceiverEmitterCellSemantics"]
        )
        self.assertTrue(SCIENTIFIC_STATUS["requiresRevalidationBeforeConsumption"])
        self.assertFalse(SCIENTIFIC_STATUS["descriptorAccessAloneIsAuthentication"])
        self.assertFalse(
            SCIENTIFIC_STATUS["protectsAgainstMaliciousSameProcessObjectMutation"]
        )

    def test_current_06_five_grid_summary_passes_v1_but_v2_rejects(self) -> None:
        # Reconstructed from the authenticated 2026-08-10 -06 direction cache.
        # The one-annulus upper-face coefficient equals its proper-power/g2
        # column because receiver and emitter areas are identical.
        rows = (
            (
                "full",
                0.033397157422357784,
                (
                    0.019595943359167024,
                    0.0,
                    0.006902753335079088,
                    0.97155417689212,
                    0.0019471264136339398,
                ),
                0.01274158819778768,
            ),
            (
                "half-rho",
                0.020427743205791084,
                (
                    0.015415053608898667,
                    0.0,
                    0.008299316799806526,
                    0.9762455252842511,
                    4.010430704368293e-05,
                ),
                0.022125144432577716,
            ),
            (
                "half-mu",
                0.020052526225617918,
                (
                    0.018921135908593865,
                    0.0,
                    0.004260891782790302,
                    0.9667212384580618,
                    0.010096733850554093,
                ),
                0.02175592768865436,
            ),
            (
                "half-psi",
                0.022347711468476188,
                (
                    0.01734937820191196,
                    0.0,
                    0.007360963376554916,
                    0.9741287927843336,
                    0.0011608656371994912,
                ),
                0.02548317639557536,
            ),
            (
                "phase-shifted",
                0.023909196976138844,
                (
                    0.021995814937077803,
                    0.0,
                    0.004179140859450149,
                    0.9694823560491678,
                    0.0043426881543042585,
                ),
                0.01274158819778768,
            ),
        )
        area = 2179.1979541357364
        full_name, full_k, full_fates, full_maximum_weight = rows[0]
        full = _summary(
            full_name,
            ((full_k,),),
            (full_k,),
            (full_fates,),
            receiver_areas=(area,),
            emitter_areas=(area,),
            maximum_weight=full_maximum_weight,
        )

        for name, comparison_k, comparison_fates, maximum_weight in rows[1:]:
            with self.subTest(grid=name):
                self.assertTrue(_v1_scalar_gate(full_k, comparison_k))
                self.assertTrue(_v1_scalar_gate(full_k, comparison_k))
                self.assertTrue(
                    all(
                        _v1_scalar_gate(first, second)
                        for first, second in zip(full_fates, comparison_fates)
                    )
                )
                comparison = _summary(
                    name,
                    ((comparison_k,),),
                    (comparison_k,),
                    (comparison_fates,),
                    receiver_areas=(area,),
                    emitter_areas=(area,),
                    maximum_weight=maximum_weight,
                )
                result = compare_kerr_returning_radiation_grids_v2(
                    full,
                    comparison,
                )
                self.assertFalse(result.resolution_qualified)
                self.assertFalse(result.columns[0].g2_converged)
                self.assertFalse(result.columns[0].column_l1_converged)
                self.assertFalse(result.fates[0].converged)
                self.assertFalse(result.converged)
                result.revalidate()

    def test_proper_power_passes_when_raw_k_changes_with_receiver_area(self) -> None:
        fine = _summary(
            "area-fine",
            ((0.125,), (0.25,)),
            (0.375,),
            receiver_areas=(1.0, 1.0),
        )
        comparison = _summary(
            "area-comparison",
            ((0.0625,), (0.0625,)),
            (0.375,),
            receiver_areas=(2.0, 4.0),
        )
        self.assertNotEqual(fine.coefficients, comparison.coefficients)
        self.assertEqual(fine.proper_power_matrix, comparison.proper_power_matrix)

        result = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        self.assertTrue(result.converged)
        self.assertEqual(result.columns[0].column_normalized_l1_difference, 0.0)
        self.assertEqual(
            result.columns[0].maximum_significant_cell_symmetric_relative_difference,
            0.0,
        )

    def test_sampled_zero_support_flip_fails_matrix_and_fate(self) -> None:
        support = 2.0**-12
        fine = _summary(
            "zero-fine",
            ((0.0,),),
            (0.0,),
            ((0.0, 0.0, 0.0, 1.0, 0.0),),
        )
        comparison = _summary(
            "zero-comparison",
            ((support,),),
            (support,),
            ((support, 0.0, 0.0, 1.0 - support, 0.0),),
        )

        result = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        self.assertEqual(result.columns[0].support_flip_count, 1)
        self.assertFalse(result.columns[0].support_converged)
        self.assertEqual(result.fates[0].support_flip_count, 1)
        self.assertFalse(result.fates[0].support_converged)
        self.assertFalse(result.converged)

    def test_many_tiny_cell_differences_fail_insignificant_tail(self) -> None:
        tiny_fine = 2.0**-14
        tiny_delta = 2.0**-20
        tiny_comparison = tiny_fine - tiny_delta
        tiny_count = 200
        anchor = 0.125 - tiny_count * tiny_fine
        fine_values = (anchor,) + (tiny_fine,) * tiny_count
        comparison_values = (anchor,) + (tiny_comparison,) * tiny_count
        fine_g2 = math.fsum(fine_values)
        comparison_g2 = math.fsum(comparison_values)
        fine = _summary(
            "tail-fine",
            tuple((value,) for value in fine_values),
            (fine_g2,),
        )
        comparison = _summary(
            "tail-comparison",
            tuple((value,) for value in comparison_values),
            (comparison_g2,),
        )

        result = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        column = result.columns[0]
        self.assertTrue(column.g2_converged)
        self.assertTrue(column.column_l1_converged)
        self.assertTrue(column.significant_cells_converged)
        self.assertEqual(column.insignificant_cell_count, tiny_count)
        self.assertGreater(column.insignificant_tail_normalized_difference, 1.0e-3)
        self.assertFalse(column.insignificant_tail_converged)
        self.assertFalse(result.converged)

    def test_exact_builtin_types_positive_zero_limits_and_tamper_fail_closed(self) -> None:
        class AlwaysEqualFloat(float):
            def __eq__(self, _other):
                return True

        with self.assertRaises(TypeError):
            KerrReturningRadiationGridSummaryV2(
                "evil-float",
                (AlwaysEqualFloat(1.0),),
                (1.0,),
                ((0.0,),),
                (0.0,),
                ((0.0, 0.0, 0.0, 1.0, 0.0),),
                5.0e-4,
            )
        with self.assertRaisesRegex(TypeError, "built only by"):
            KerrReturningRadiationGridComparisonV2()
        with self.assertRaises(ValueError):
            _summary(
                "negative-zero",
                ((-0.0,),),
                (0.0,),
                ((0.0, 0.0, 0.0, 1.0, 0.0),),
            )
        fine, comparison = self.synthetic_passing_pair()
        with self.assertRaises(ValueError):
            compare_kerr_returning_radiation_grids_v2(
                fine,
                comparison,
                KerrReturningRadiationConvergenceV2Policy(
                    maximum_receiver_cells=1,
                ),
            )

        valid = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        forged = object.__new__(KerrReturningRadiationGridComparisonV2)
        for name in (
            "fine",
            "comparison",
            "columns",
            "fates",
            "resolution_qualified",
            "converged",
        ):
            object.__setattr__(forged, name, object.__getattribute__(valid, name))
        object.__setattr__(
            forged,
            "policy",
            KerrReturningRadiationConvergenceV2Policy(maximum_receiver_cells=1),
        )
        with self.assertRaisesRegex(ValueError, "receiver-cell limit"):
            forged.__post_init__()

        result = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        object.__setattr__(result, "converged", False)
        with self.assertRaises(KerrReturningRadiationConvergenceV2VerificationError):
            result.revalidate()

        clean = compare_kerr_returning_radiation_grids_v2(fine, comparison)
        object.__setattr__(clean, "_descriptor_sha256", "0" * 64)
        with self.assertRaises(KerrReturningRadiationConvergenceV2VerificationError):
            clean.revalidate()

    def test_authenticated_direct_five_pass_order_blocks_and_revalidation(self) -> None:
        kernel, classifier = self.authenticated_direct_source()
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            result = authenticate_direct_kerr_returning_radiation_convergence_v2(
                kernel
            )
        self.assertIs(type(result), KerrAuthenticatedReturningRadiationConvergenceV2)
        self.assertEqual(len(result.summaries), 5)
        self.assertEqual(len(result.reports), 4)
        self.assertEqual(
            tuple(item.grid_id for item in result.summaries),
            (
                "direct-full",
                "direct-half-rho",
                "direct-half-mu",
                "direct-half-psi",
                "direct-phase-shifted",
            ),
        )
        full = result.full
        self.assertEqual(
            full.receiver_areas,
            (
                *kernel.upper_annulus_areas_over_mass_squared,
                *kernel.lower_annulus_areas_over_mass_squared,
            ),
        )
        self.assertEqual(
            full.coefficients,
            (
                (
                    *kernel.upper_receiver_upper_emitter_coefficients[0],
                    *kernel.upper_receiver_lower_emitter_coefficients[0],
                ),
                (
                    *kernel.lower_receiver_upper_emitter_coefficients[0],
                    *kernel.lower_receiver_lower_emitter_coefficients[0],
                ),
            ),
        )
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            document = result.descriptor()
        self.assertEqual(document["cellSemantics"]["matrixBlockRows"], "UU|UL then LU|LL")
        self.assertFalse(document["authentication"]["isIndependentGeodesicOrPhysicsOracle"])
        self.assertFalse(document["authentication"]["isRigorousContinuumErrorBound"])
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            result.revalidate()

        object.__setattr__(result, "summaries", tuple(reversed(result.summaries)))
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            fresh_document = result.descriptor()
            self.assertEqual(fresh_document["passEvidence"][0]["passName"], "full")
            with self.assertRaises(KerrReturningRadiationConvergenceV2VerificationError):
                result.revalidate()
        with self.assertRaisesRegex(TypeError, "built only"):
            KerrAuthenticatedReturningRadiationConvergenceV2()
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            projection = kernel.coarsen_annuli(kernel.annulus_edges_over_mass)
        with self.assertRaises(TypeError):
            authenticate_direct_kerr_returning_radiation_convergence_v2(projection)
        object.__setattr__(kernel, "full_grid_sample_audit_sha256", "0" * 64)
        with patch.object(forward_module, "_trace_direction", side_effect=classifier):
            with self.assertRaises(forward_module.KerrReturningRadiationKernelVerificationError):
                result.descriptor()


if __name__ == "__main__":
    unittest.main()
