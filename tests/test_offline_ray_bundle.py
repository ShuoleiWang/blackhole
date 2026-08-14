from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from dataclasses import replace

from offline.ray_bundle import (
    RAY_BUNDLE_DIAGNOSTIC_VERSION,
    SCIENTIFIC_STATUS,
    RayBundleDiagnosticError,
    RayBundleOptions,
    RayBundleWorkBudgetExceeded,
    RayEndpointConvergenceAudit,
    RayEndpointSample,
    diagnose_screen_ray_bundle,
    pinhole_chart_solid_angle_density,
    stable_svd_2x2,
)


Coordinates = tuple[float, float]
Metadata = tuple[str, str, str]


def _passed_audit(
    *,
    maximum_null_residual: float = 0.0,
    maximum_source_coordinate_error: float = 0.0,
) -> RayEndpointConvergenceAudit:
    return RayEndpointConvergenceAudit(
        maximum_null_residual=maximum_null_residual,
        maximum_source_coordinate_error=maximum_source_coordinate_error,
        accepted_steps=3,
        rejected_steps=1,
        ray_gate_passed=True,
        source_topology_gate_passed=True,
        source_coordinate_gate_passed=True,
    )


class _AnalyticEndpointMapper:
    implementation_id = "test-analytic-endpoint-map/v1"

    def __init__(
        self,
        coordinate_map: Callable[[float, float], Coordinates],
        *,
        density: Callable[[float, float, Coordinates], float] | None = None,
        metadata: Callable[[float, float], Metadata] | None = None,
        converged: Callable[[float, float], bool] | None = None,
        audit: Callable[[float, float], RayEndpointConvergenceAudit] | None = None,
    ) -> None:
        self.coordinate_map = coordinate_map
        self.density = density
        self.metadata = metadata
        self.converged = converged
        self.audit = audit
        self.calls: list[tuple[float, float]] = []

    def map_endpoint(self, screen_x: float, screen_y: float) -> RayEndpointSample:
        self.calls.append((screen_x, screen_y))
        coordinates = self.coordinate_map(screen_x, screen_y)
        source_kind, topology, chart = (
            self.metadata(screen_x, screen_y)
            if self.metadata is not None
            else ("escaped-sky", "same-sky-image", "pinhole-source-sky/v1")
        )
        endpoint_converged = (
            self.converged(screen_x, screen_y)
            if self.converged is not None
            else True
        )
        convergence_audit = (
            self.audit(screen_x, screen_y)
            if self.audit is not None
            else _passed_audit()
        )
        source_density = (
            self.density(screen_x, screen_y, coordinates)
            if self.density is not None
            else pinhole_chart_solid_angle_density(*coordinates)
        )
        return RayEndpointSample(
            source_kind=source_kind,
            topology_signature=topology,
            source_chart_id=chart,
            source_coordinates=coordinates,
            source_solid_angle_density_sr_per_coordinate_area=source_density,
            endpoint_converged=endpoint_converged,
            convergence_audit=convergence_audit,
        )


def _options(**overrides: object) -> RayBundleOptions:
    values: dict[str, object] = {
        "finite_difference_step": 1.0e-4,
        "jacobian_absolute_tolerance": 1.0e-10,
        "jacobian_relative_tolerance": 1.0e-8,
        "minimum_singular_value_ratio": 1.0e-10,
        "maximum_endpoint_evaluations": 9,
    }
    values.update(overrides)
    return RayBundleOptions(**values)  # type: ignore[arg-type]


class OfflineRayBundleTests(unittest.TestCase):
    def test_scientific_status_forbids_sachs_and_completeness_claims(self) -> None:
        self.assertIn("finite-difference", SCIENTIFIC_STATUS["classification"])
        self.assertIs(SCIENTIFIC_STATUS["isSachsJacobiRayBundle"], False)
        self.assertIs(SCIENTIFIC_STATUS["integratesGeodesicDeviation"], False)
        self.assertIs(SCIENTIFIC_STATUS["providesWavefrontCurvature"], False)
        self.assertIs(SCIENTIFIC_STATUS["providesTimeDelayHessian"], False)
        self.assertIs(SCIENTIFIC_STATUS["isCausticComplete"], False)
        self.assertIn("Sachs/Jacobi", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isCausticComplete"] = True  # type: ignore[index]

    def test_minkowski_pinhole_identity_has_unit_magnification_off_axis(self) -> None:
        mapper = _AnalyticEndpointMapper(lambda x_value, y_value: (x_value, y_value))
        result = diagnose_screen_ray_bundle(
            mapper,
            0.37,
            -0.21,
            options=_options(),
        )
        self.assertEqual(result.diagnostic_version, RAY_BUNDLE_DIAGNOSTIC_VERSION)
        self.assertEqual(result.sample_count, 9)
        self.assertEqual(len(mapper.calls), 9)
        self.assertEqual(result.parity, 1)
        self.assertFalse(result.near_critical)
        self.assertAlmostEqual(result.jacobian[0][0], 1.0, places=11)
        self.assertAlmostEqual(result.jacobian[0][1], 0.0, places=15)
        self.assertAlmostEqual(result.jacobian[1][0], 0.0, places=15)
        self.assertAlmostEqual(result.jacobian[1][1], 1.0, places=11)
        self.assertAlmostEqual(result.determinant or 0.0, 1.0, places=11)
        self.assertAlmostEqual(result.solid_angle_magnification or 0.0, 1.0)

    def test_analytic_anisotropic_scaling(self) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (2.0 * x_value, 3.0 * y_value)
        )
        result = diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())
        self.assertAlmostEqual(result.determinant or 0.0, 6.0)
        self.assertEqual(result.parity, 1)
        self.assertAlmostEqual(result.singular_values[0], 3.0)
        self.assertAlmostEqual(result.singular_values[1], 2.0)
        self.assertAlmostEqual(result.condition_number, 1.5)
        self.assertAlmostEqual(result.solid_angle_magnification or 0.0, 1.0 / 6.0)

    def test_rotation_shear_and_negative_parity(self) -> None:
        angle = 0.73
        cosine = math.cos(angle)
        sine = math.sin(angle)
        cases = (
            (
                "rotation",
                ((cosine, -sine), (sine, cosine)),
                1,
                (1.0, 1.0),
            ),
            (
                "shear",
                ((1.0, 2.0), (0.0, 1.0)),
                1,
                (1.0 + math.sqrt(2.0), math.sqrt(2.0) - 1.0),
            ),
            (
                "parity",
                ((-1.0, 0.0), (0.0, 1.0)),
                -1,
                (1.0, 1.0),
            ),
        )
        for label, matrix, expected_parity, expected_singular in cases:
            with self.subTest(label=label):
                mapper = _AnalyticEndpointMapper(
                    lambda x_value, y_value, matrix=matrix: (
                        matrix[0][0] * x_value + matrix[0][1] * y_value,
                        matrix[1][0] * x_value + matrix[1][1] * y_value,
                    )
                )
                result = diagnose_screen_ray_bundle(
                    mapper,
                    0.0,
                    0.0,
                    options=_options(),
                )
                self.assertEqual(result.parity, expected_parity)
                self.assertAlmostEqual(
                    result.singular_values[0], expected_singular[0], places=12
                )
                self.assertAlmostEqual(
                    result.singular_values[1], expected_singular[1], places=12
                )
                self.assertAlmostEqual(
                    result.solid_angle_magnification or 0.0,
                    1.0,
                    places=12,
                )

    def test_fold_map_marks_critical_curve_without_finite_reciprocal(
        self,
    ) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value * x_value, y_value)
        )
        critical = diagnose_screen_ray_bundle(
            mapper,
            0.0,
            0.0,
            options=_options(),
        )
        self.assertTrue(critical.near_critical)
        self.assertEqual(critical.determinant, 0.0)
        self.assertEqual(critical.determinant_sign, 0)
        self.assertEqual(critical.parity, 0)
        self.assertTrue(math.isinf(critical.condition_number))
        self.assertIsNone(critical.solid_angle_magnification)
        self.assertIsNone(critical.log_solid_angle_magnification)
        with self.assertRaisesRegex(
            ValueError,
            "may not report finite magnification",
        ):
            replace(critical, solid_angle_magnification=1.0)

        positive = diagnose_screen_ray_bundle(
            mapper,
            0.25,
            0.0,
            options=_options(),
        )
        negative = diagnose_screen_ray_bundle(
            mapper,
            -0.25,
            0.0,
            options=_options(),
        )
        self.assertAlmostEqual(positive.determinant or 0.0, 0.5, places=11)
        self.assertAlmostEqual(negative.determinant or 0.0, -0.5, places=11)
        self.assertEqual(positive.parity, 1)
        self.assertEqual(negative.parity, -1)

    def test_near_rank_loss_never_fabricates_finite_magnification(self) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value, 1.0e-13 * y_value)
        )
        result = diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())
        self.assertTrue(result.near_critical)
        self.assertEqual(result.determinant_sign, 1)
        self.assertGreater(result.determinant or 0.0, 0.0)
        self.assertEqual(result.parity, 0)
        self.assertIsNone(result.solid_angle_magnification)

    def test_h_and_h_over_two_must_converge(self) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value + x_value**3, y_value)
        )
        with self.assertRaisesRegex(
            RayBundleDiagnosticError,
            "h versus h/2 Jacobian convergence gate",
        ):
            diagnose_screen_ray_bundle(
                mapper,
                0.0,
                0.0,
                options=_options(
                    finite_difference_step=0.1,
                    jacobian_absolute_tolerance=0.0,
                    jacobian_relative_tolerance=1.0e-5,
                ),
            )

        accepted = diagnose_screen_ray_bundle(
            mapper,
            0.0,
            0.0,
            options=_options(
                finite_difference_step=0.1,
                jacobian_absolute_tolerance=0.01,
                jacobian_relative_tolerance=0.0,
            ),
        )
        self.assertAlmostEqual(accepted.coarse_jacobian[0][0], 1.01)
        self.assertAlmostEqual(accepted.fine_jacobian[0][0], 1.0025)
        self.assertAlmostEqual(accepted.jacobian_difference_norm, 0.0075)
        self.assertAlmostEqual(accepted.estimated_jacobian_error_norm, 0.0025)

    def test_mixed_capture_escape_topology_fails_closed(self) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value, y_value),
            metadata=lambda x_value, _y_value: (
                ("captured-boundary", "horizon", "horizon-chart/v1")
                if x_value < 0.0
                else ("escaped-sky", "sky", "pinhole-source-sky/v1")
            ),
        )
        with self.assertRaisesRegex(
            RayBundleDiagnosticError,
            "mixed source topology or chart",
        ):
            diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())

    def test_unconverged_endpoint_fails_closed(self) -> None:
        failed_audit = RayEndpointConvergenceAudit(
            accepted_steps=3,
            ray_gate_passed=True,
            source_topology_gate_passed=True,
            source_coordinate_gate_passed=False,
        )
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value, y_value),
            converged=lambda x_value, _y_value: x_value != 5.0e-5,
            audit=lambda x_value, _y_value: (
                failed_audit if x_value == 5.0e-5 else _passed_audit()
            ),
        )
        with self.assertRaisesRegex(
            RayBundleDiagnosticError,
            "endpoint convergence gate failed",
        ):
            diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())

    def test_nan_endpoint_and_density_are_rejected(self) -> None:
        for label, mapper in (
            (
                "coordinate",
                _AnalyticEndpointMapper(lambda _x_value, y_value: (math.nan, y_value)),
            ),
            (
                "density",
                _AnalyticEndpointMapper(
                    lambda x_value, y_value: (x_value, y_value),
                    density=lambda _x, _y, _coordinates: math.nan,
                ),
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    RayBundleDiagnosticError,
                    "endpoint mapper failed",
                ):
                    diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())

    def test_work_budget_is_checked_before_mapper_invocation(self) -> None:
        mapper = _AnalyticEndpointMapper(lambda x_value, y_value: (x_value, y_value))
        with self.assertRaisesRegex(
            RayBundleWorkBudgetExceeded,
            "nine-ray endpoint stencil",
        ):
            diagnose_screen_ray_bundle(
                mapper,
                0.0,
                0.0,
                options=_options(maximum_endpoint_evaluations=8),
            )
        self.assertEqual(mapper.calls, [])

    def test_extreme_coordinate_relabeling_keeps_physical_magnification(self) -> None:
        coordinate_scale = 1.0e150
        density = 1.0 / (coordinate_scale * coordinate_scale)
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (
                coordinate_scale * x_value,
                coordinate_scale * y_value,
            ),
            density=lambda _x, _y, _coordinates: density,
        )
        result = diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())
        self.assertFalse(result.near_critical)
        self.assertEqual(result.parity, 1)
        self.assertAlmostEqual(result.singular_values[0] / coordinate_scale, 1.0)
        self.assertAlmostEqual(result.singular_values[1] / coordinate_scale, 1.0)
        self.assertAlmostEqual(result.condition_number, 1.0)
        self.assertIsNotNone(result.determinant)
        self.assertAlmostEqual(
            (result.determinant or 0.0) / (coordinate_scale * coordinate_scale),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(result.solid_angle_magnification or 0.0, 1.0)

    def test_stable_svd_handles_large_scale_and_exact_rank_loss(self) -> None:
        scaled = stable_svd_2x2(((1.0e150, 0.0), (0.0, -2.0e150)))
        self.assertEqual(scaled.determinant_sign, -1)
        self.assertAlmostEqual(scaled.singular_values[0] / 1.0e150, 2.0)
        self.assertAlmostEqual(scaled.singular_values[1] / 1.0e150, 1.0)
        self.assertAlmostEqual(scaled.condition_number, 2.0)
        singular = stable_svd_2x2(((1.0, 2.0), (2.0, 4.0)))
        self.assertEqual(singular.determinant, 0.0)
        self.assertEqual(singular.determinant_sign, 0)
        self.assertTrue(math.isinf(singular.condition_number))

    def test_mapper_audit_maxima_are_preserved(self) -> None:
        mapper = _AnalyticEndpointMapper(
            lambda x_value, y_value: (x_value, y_value),
            audit=lambda x_value, y_value: _passed_audit(
                maximum_null_residual=abs(x_value) + abs(y_value),
                maximum_source_coordinate_error=2.0 * (abs(x_value) + abs(y_value)),
            ),
        )
        result = diagnose_screen_ray_bundle(mapper, 0.0, 0.0, options=_options())
        self.assertAlmostEqual(result.maximum_null_residual, 1.0e-4)
        self.assertAlmostEqual(result.maximum_source_coordinate_error, 2.0e-4)

    def test_invalid_options_and_unrepresentable_stencil_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _options(maximum_endpoint_evaluations=True)
        with self.assertRaisesRegex(ValueError, "less than one"):
            _options(minimum_singular_value_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _options(minimum_singular_value_ratio=0.0)
        mapper = _AnalyticEndpointMapper(lambda x_value, y_value: (x_value, y_value))
        with self.assertRaisesRegex(
            RayBundleDiagnosticError,
            "not distinct",
        ):
            diagnose_screen_ray_bundle(
                mapper,
                1.0e100,
                0.0,
                options=_options(finite_difference_step=1.0e-4),
            )


if __name__ == "__main__":
    unittest.main()
