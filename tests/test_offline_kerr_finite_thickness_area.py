from __future__ import annotations

import hashlib
import json
import math
import unittest

import offline.kerr_finite_thickness_area as area_module
from offline.kerr import KerrKerrSchildMetric, kerr_bl_vector_to_ks_cartesian
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    IMPLEMENTATION_ID,
    MAXIMUM_POINT_EVALUATIONS,
    SCIENTIFIC_STATUS,
    KerrFiniteThicknessAreaConvergenceError,
    KerrFiniteThicknessAreaQuadraturePolicy,
    KerrFiniteThicknessAreaVerificationError,
    integrate_kerr_finite_thickness_annulus_area,
    kerr_finite_thickness_area_density,
    verify_kerr_finite_thickness_annulus_area,
    verify_kerr_finite_thickness_area_density,
)
from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
from offline.spacetime import bilinear, matrix_vector


class AlwaysEqualFloat(float):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class MetricSubclass(KerrKerrSchildMetric):
    pass


class OfflineKerrFiniteThicknessAreaTests(unittest.TestCase):
    @staticmethod
    def calibration(
        *,
        spin: float = 0.7,
        dotm: float = 0.08,
        outer: float = 30.0,
    ) -> StationaryKerrFiniteThicknessCalibration:
        return StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=dotm,
            outer_radius_over_mass=outer,
        )

    @staticmethod
    def metric(*, spin: float = 0.7, mass: float = 1.0) -> KerrKerrSchildMetric:
        return KerrKerrSchildMetric(mass_m=mass, spin_a_m=spin * mass)

    @staticmethod
    def policy(order: int = 16) -> KerrFiniteThicknessAreaQuadraturePolicy:
        return KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=order,
            relative_tolerance=1.0e-7,
            absolute_tolerance_over_mass_squared=1.0e-7,
            maximum_point_evaluations=MAXIMUM_POINT_EVALUATIONS,
        )

    def test_every_internal_area_order_has_a_resolved_float64_rule(self) -> None:
        legacy_noncycling = hashlib.sha256()
        formerly_cycling = {62, 121, 124, 130, 133, 142, 204, 213, 244}
        for order in range(1, 257):
            with self.subTest(order=order):
                rule = area_module._gauss_legendre_unit_interval(order)
                self.assertEqual(len(rule), order)
                self.assertLessEqual(
                    abs(math.fsum(weight for _node, weight in rule) - 1.0),
                    64.0 * math.ulp(1.0),
                )
                power = min(3, 2 * order - 1)
                moment = math.fsum(
                    weight * node**power for node, weight in rule
                )
                self.assertTrue(
                    math.isclose(moment, 1.0 / (power + 1), abs_tol=2.0e-15)
                )
                if order not in formerly_cycling:
                    for node, weight in rule:
                        legacy_noncycling.update(
                            f"{order}\0{node.hex()}\0{weight.hex()}\n".encode(
                                "ascii"
                            )
                        )
        self.assertEqual(
            legacy_noncycling.hexdigest(),
            "0d284407d44bfc84290e282fbdb58a133b7733b9d1304c7d6fbf56f98a6c2230",
        )

    def test_scientific_boundary_is_explicit_and_immutable(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertIn("stationary prescribed", SCIENTIFIC_STATUS["classification"])
        self.assertEqual(
            SCIENTIFIC_STATUS["restSpaceProjector"],
            "h_mu_nu = g_mu_nu + u_mu u_nu",
        )
        self.assertIs(SCIENTIFIC_STATUS["isStationaryPrescribedSurfaceArea"], True)
        for key in (
            "isReturningRadiationKernel",
            "isReceiverSolidAngleAreaJacobian",
            "includesReturningRadiationStressWork",
            "isHydrostaticVerticalStructureSolution",
            "includesSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("GRMHD", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isReturningRadiationKernel"] = True

    def test_direct_exact_metric_projection_recomputes_q_and_density(self) -> None:
        calibration = self.calibration()
        metric = self.metric(mass=2.3)
        rho = 8.0
        phi = 0.41
        result = kerr_finite_thickness_area_density(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=UPPER,
            phi_ks_rad=phi,
            coordinate_time_m=1.7,
        )

        # Independent test-side construction of X_,rho and X_,phi.  This does
        # not consume any stored q component or area-density value.
        emitter = KerrFiniteThicknessFaceEmitter(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=UPPER,
            phi_ks_rad=phi,
            coordinate_time_m=1.7,
        )
        point = emitter.photosphere_point
        z = point.signed_height_over_mass
        dz = calibration.photosphere_height_derivative(rho)
        dr = (rho + z * dz) / point.radius_over_mass
        dtheta = (z - rho * dz) / point.radius_over_mass**2
        x_rho = kerr_bl_vector_to_ks_cartesian(
            (0.0, metric.mass_m * dr, dtheta, 0.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=point.radius_over_mass * metric.mass_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
        )
        x_phi = kerr_bl_vector_to_ks_cartesian(
            (0.0, 0.0, 0.0, 1.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=point.radius_over_mass * metric.mass_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=phi,
        )
        g = metric.sample(emitter.event).covariant
        u_cov = matrix_vector(g, emitter.four_velocity)
        h = tuple(
            tuple(g[i][j] + u_cov[i] * u_cov[j] for j in range(4))
            for i in range(4)
        )
        q_rr = bilinear(x_rho, h, x_rho)
        q_rp = bilinear(x_rho, h, x_phi)
        q_pp = bilinear(x_phi, h, x_phi)
        expected = math.sqrt(q_rr * q_pp - q_rp * q_rp)
        self.assertEqual(result.embedding_radial_tangent_ks, x_rho)
        self.assertEqual(result.embedding_azimuthal_tangent_ks, x_phi)
        self.assertAlmostEqual(result.q_rho_rho_m2, q_rr, delta=2.0e-13)
        self.assertAlmostEqual(result.q_rho_phi_m2, q_rp, delta=2.0e-13)
        self.assertAlmostEqual(result.q_phi_phi_m2, q_pp, delta=2.0e-13)
        self.assertAlmostEqual(result.proper_area_density_m2, expected, delta=2e-13)
        self.assertAlmostEqual(
            result.proper_area_density_over_mass_squared,
            expected / metric.mass_m**2,
            delta=2.0e-13,
        )
        self.assertLess(result.maximum_tangency_residual_over_mass, 2.0e-13)
        result.revalidate()

    def test_schwarzschild_zero_thickness_limit_matches_independent_formula(self) -> None:
        # At z=0 in Schwarzschild, q_rr=r/(r-2) and
        # q_phiphi=r^2(r-2)/(r-3), so sqrt(det q)/M^2 =
        # r/sqrt(1-3/r).  A tiny positive height is required because the two
        # repository faces intentionally have no identity at exactly dotm=0.
        rho = 10.0
        expected = rho / math.sqrt(1.0 - 3.0 / rho)
        calibration = self.calibration(spin=0.0, dotm=1.0e-7, outer=20.0)
        metric = self.metric(spin=0.0, mass=2.3)
        for face in (UPPER, LOWER):
            with self.subTest(face=face):
                result = kerr_finite_thickness_area_density(
                    metric=metric,
                    calibration=calibration,
                    pseudo_cylindrical_radius_over_mass=rho,
                    face=face,
                )
                self.assertTrue(
                    math.isclose(
                        result.proper_area_density_over_mass_squared,
                        expected,
                        rel_tol=3.0e-13,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        result.proper_area_density_m2,
                        metric.mass_m**2 * expected,
                        rel_tol=3.0e-13,
                    )
                )

    def test_upper_lower_reflection_and_stationary_axisymmetry(self) -> None:
        calibration = self.calibration(spin=0.998, dotm=0.2)
        metric = self.metric(spin=0.998, mass=1.7)
        values = []
        for face in (UPPER, LOWER):
            for phi, time in ((0.0, 0.0), (1.234, 8.0)):
                result = kerr_finite_thickness_area_density(
                    metric=metric,
                    calibration=calibration,
                    pseudo_cylindrical_radius_over_mass=4.0,
                    face=face,
                    phi_ks_rad=phi,
                    coordinate_time_m=time,
                )
                self.assertGreater(result.q_rho_rho_m2, 0.0)
                self.assertGreater(result.q_phi_phi_m2, 0.0)
                self.assertGreater(result.determinant_m4, 0.0)
                self.assertGreater(result.proper_area_density_m2, 0.0)
                values.append(result.proper_area_density_over_mass_squared)
        for value in values[1:]:
            self.assertTrue(math.isclose(value, values[0], rel_tol=3.0e-14))

    def test_metric_projection_is_not_euclidean_surface_area(self) -> None:
        calibration = self.calibration()
        result = kerr_finite_thickness_area_density(
            metric=self.metric(),
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=8.0,
            face=UPPER,
        )
        euclidean = 8.0 * math.sqrt(
            1.0 + calibration.photosphere_height_derivative(8.0) ** 2
        )
        self.assertGreater(
            abs(result.proper_area_density_over_mass_squared - euclidean),
            0.05 * euclidean,
        )

    def test_annulus_budget_n_2n_and_descriptor_are_bound(self) -> None:
        calibration = self.calibration()
        metric = self.metric(mass=2.0)
        policy = self.policy(order=12)
        result = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=calibration.isco_radius_over_mass,
            outer_radius_over_mass=12.0,
            face=UPPER,
            policy=policy,
            phi_ks_rad=0.8,
        )
        self.assertEqual(result.coarse_order, 12)
        self.assertEqual(result.fine_order, 24)
        self.assertEqual(result.point_evaluations, 36)
        self.assertLessEqual(result.point_evaluations, result.maximum_point_evaluations)
        self.assertLessEqual(
            result.estimated_absolute_error_over_mass_squared,
            result.convergence_threshold_over_mass_squared,
        )
        self.assertEqual(
            result.proper_area_m2,
            metric.mass_m**2 * result.proper_area_over_mass_squared,
        )
        descriptor = result.model_descriptor()
        canonical = json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            result.model_descriptor_sha256,
        )
        self.assertEqual(descriptor["quadrature"]["requiredPointEvaluations"], 36)
        self.assertEqual(descriptor["annulus"]["face"], UPPER)
        verify_kerr_finite_thickness_annulus_area(result)

    def test_annulus_merge_conserves_proper_area(self) -> None:
        calibration = self.calibration()
        metric = self.metric()
        policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=16,
            relative_tolerance=2.0e-10,
            absolute_tolerance_over_mass_squared=2.0e-11,
        )
        inner = calibration.isco_radius_over_mass
        middle = 7.0
        outer = 12.0
        left = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=inner,
            outer_radius_over_mass=middle,
            face=LOWER,
            policy=policy,
        )
        right = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=middle,
            outer_radius_over_mass=outer,
            face=LOWER,
            policy=policy,
        )
        whole = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=inner,
            outer_radius_over_mass=outer,
            face=LOWER,
            policy=policy,
        )
        self.assertTrue(
            math.isclose(
                left.proper_area_over_mass_squared
                + right.proper_area_over_mass_squared,
                whole.proper_area_over_mass_squared,
                rel_tol=3.0e-13,
                abs_tol=3.0e-11,
            )
        )

    def test_n_2n_error_contracts_with_resolution(self) -> None:
        calibration = self.calibration()
        metric = self.metric()
        low = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=calibration.isco_radius_over_mass,
            outer_radius_over_mass=12.0,
            face=UPPER,
            policy=self.policy(order=8),
        )
        high = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=calibration.isco_radius_over_mass,
            outer_radius_over_mass=12.0,
            face=UPPER,
            policy=self.policy(order=12),
        )
        self.assertLess(
            high.estimated_absolute_error_over_mass_squared,
            low.estimated_absolute_error_over_mass_squared / 100.0,
        )
        self.assertTrue(
            math.isclose(
                low.proper_area_over_mass_squared,
                high.proper_area_over_mass_squared,
                rel_tol=1.0e-7,
            )
        )
        with self.assertRaises(KerrFiniteThicknessAreaConvergenceError):
            integrate_kerr_finite_thickness_annulus_area(
                metric=metric,
                calibration=calibration,
                inner_radius_over_mass=calibration.isco_radius_over_mass,
                outer_radius_over_mass=12.0,
                face=UPPER,
                policy=self.policy(order=4),
            )

    def test_near_isco_and_near_extremal_calibrated_spin_are_finite(self) -> None:
        calibration = self.calibration(spin=0.998, dotm=0.1, outer=20.0)
        metric = self.metric(spin=0.998, mass=1.3)
        rho = math.nextafter(calibration.isco_radius_over_mass, math.inf)
        point = kerr_finite_thickness_area_density(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=UPPER,
        )
        self.assertTrue(math.isfinite(point.proper_area_density_over_mass_squared))
        self.assertGreater(point.proper_area_density_over_mass_squared, 0.0)
        annulus = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=calibration.isco_radius_over_mass,
            outer_radius_over_mass=3.0,
            face=LOWER,
        )
        self.assertTrue(math.isfinite(annulus.proper_area_over_mass_squared))
        self.assertGreater(annulus.proper_area_over_mass_squared, 0.0)
        annulus.revalidate()

    def test_illegal_inputs_and_tampered_policy_fail_closed(self) -> None:
        calibration = self.calibration()
        metric = self.metric()
        with self.assertRaises(TypeError):
            kerr_finite_thickness_area_density(
                metric=MetricSubclass(spin_a_m=0.7),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=8.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "spin"):
            kerr_finite_thickness_area_density(
                metric=self.metric(spin=0.6),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=8.0,
                face=UPPER,
            )
        zero = self.calibration(dotm=0.0)
        with self.assertRaisesRegex(ValueError, "positive finite thickness"):
            kerr_finite_thickness_area_density(
                metric=metric,
                calibration=zero,
                pseudo_cylindrical_radius_over_mass=8.0,
                face=UPPER,
            )
        for rho, face in (
            (calibration.isco_radius_over_mass, UPPER),
            (31.0, UPPER),
            (8.0, "side"),
        ):
            with self.subTest(rho=rho, face=face):
                with self.assertRaises(ValueError):
                    kerr_finite_thickness_area_density(
                        metric=metric,
                        calibration=calibration,
                        pseudo_cylindrical_radius_over_mass=rho,
                        face=face,
                    )
        with self.assertRaises(ValueError):
            integrate_kerr_finite_thickness_annulus_area(
                metric=metric,
                calibration=calibration,
                inner_radius_over_mass=10.0,
                outer_radius_over_mass=8.0,
                face=UPPER,
            )
        with self.assertRaises((TypeError, ValueError)):
            KerrFiniteThicknessAreaQuadraturePolicy(gauss_legendre_order=True)
        with self.assertRaises(ValueError):
            KerrFiniteThicknessAreaQuadraturePolicy(relative_tolerance=1.0e-3)
        tampered_policy = self.policy(order=8)
        object.__setattr__(tampered_policy, "relative_tolerance", 1.0)
        with self.assertRaises(ValueError):
            integrate_kerr_finite_thickness_annulus_area(
                metric=metric,
                calibration=calibration,
                inner_radius_over_mass=calibration.isco_radius_over_mass,
                outer_radius_over_mass=12.0,
                face=UPPER,
                policy=tampered_policy,
            )

    def test_point_and_annulus_tampering_fail_closed(self) -> None:
        calibration = self.calibration()
        metric = self.metric()
        point = kerr_finite_thickness_area_density(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=8.0,
            face=UPPER,
        )
        object.__setattr__(
            point,
            "proper_area_density_over_mass_squared",
            point.proper_area_density_over_mass_squared + 1.0,
        )
        with self.assertRaises(KerrFiniteThicknessAreaVerificationError):
            verify_kerr_finite_thickness_area_density(point)

        point_type_attack = kerr_finite_thickness_area_density(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=8.0,
            face=UPPER,
        )
        object.__setattr__(
            point_type_attack,
            "q_rho_rho_m2",
            AlwaysEqualFloat(point_type_attack.q_rho_rho_m2),
        )
        with self.assertRaises(KerrFiniteThicknessAreaVerificationError):
            point_type_attack.revalidate()

        annulus = integrate_kerr_finite_thickness_annulus_area(
            metric=metric,
            calibration=calibration,
            inner_radius_over_mass=calibration.isco_radius_over_mass,
            outer_radius_over_mass=12.0,
            face=UPPER,
        )
        object.__setattr__(
            annulus,
            "_descriptor_sha256",
            AlwaysEqualStr(annulus.model_descriptor_sha256),
        )
        with self.assertRaises(KerrFiniteThicknessAreaVerificationError):
            annulus.revalidate()


if __name__ == "__main__":
    unittest.main()
