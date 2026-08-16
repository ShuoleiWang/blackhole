from __future__ import annotations

import math
import unittest

from offline.kerr_finite_thickness import (
    EDGE_ON_COSINE_NUMERICAL_GUARD,
    EDDINGTON_SCALING_DEFINITION,
    LOWER,
    MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO,
    MODEL_IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    UPPER,
    BoyerLindquistPhotospherePoint,
    StationaryKerrFiniteThicknessCalibration,
    validate_calibration_observer_inclination_rad,
)
from offline.novikov_thorne import PROGRADE, RETROGRADE


class KerrFiniteThicknessTests(unittest.TestCase):
    def test_scientific_status_is_strictly_phenomenological(self) -> None:
        self.assertEqual(
            SCIENTIFIC_STATUS["classification"],
            "stationary analytic Kerr finite-thickness calibration surface",
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["implementationId"],
            MODEL_IMPLEMENTATION_ID,
        )
        self.assertIn("Newtonian", SCIENTIFIC_STATUS["heightPrescription"])
        self.assertIn("assumed photosphere", SCIENTIFIC_STATUS["heightPrescription"])
        self.assertEqual(
            SCIENTIFIC_STATUS["efficiencyDefinition"],
            "eta = 1 - E_ISCO (Novikov-Thorne)",
        )
        for key in (
            "isHydrostaticVerticalStructureSolution",
            "isGeneralRelativisticMagnetohydrodynamics",
            "includesSolvedAtmosphere",
            "includesReturningRadiation",
            "includesRadialAdvection",
            "includesSelfOcclusionRayTracing",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIs(
            SCIENTIFIC_STATUS["providesSignedSelfOcclusionGeometry"],
            True,
        )
        self.assertIn("GRMHD", SCIENTIFIC_STATUS["prohibitedClaim"])
        self.assertIn(
            "no inner or outer vertical sidewall",
            SCIENTIFIC_STATUS["radialBoundary"],
        )
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["classification"] = "mutable"

    def test_eddington_ratio_is_explicit_and_has_no_guessed_si_conversion(self) -> None:
        self.assertEqual(
            EDDINGTON_SCALING_DEFINITION["parameter"],
            "dot(M) / dot(M)_Edd",
        )
        self.assertIs(EDDINGTON_SCALING_DEFINITION["isDimensionless"], True)
        self.assertIs(
            EDDINGTON_SCALING_DEFINITION[
                "physicalDotMEddDefinitionProvidedBySource"
            ],
            False,
        )
        self.assertIs(
            EDDINGTON_SCALING_DEFINITION["supportsKilogramsPerSecondConversion"],
            False,
        )
        with self.assertRaises(TypeError):
            EDDINGTON_SCALING_DEFINITION["parameter"] = "forged"

    def test_schwarzschild_figure_profile_is_reproducible_from_equations(self) -> None:
        # Figure 1 of arXiv:2004.12589 includes a*=0 and dot_m=0.1, 0.2,
        # and 0.3.  At rho=9M, Eq. (6)--(7) and E_ISCO=sqrt(8/9)
        # independently give these values.
        expected_efficiency = 1.0 - math.sqrt(8.0 / 9.0)
        expected_photosphere_heights = (
            0.962582674684058,
            1.925165349368116,
            2.8877480240521733,
        )
        for accretion_ratio, expected_height in zip(
            (0.1, 0.2, 0.3),
            expected_photosphere_heights,
        ):
            model = StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.0,
                eddington_scaled_mass_accretion_rate=accretion_ratio,
                outer_radius_over_mass=100.0,
            )
            with self.subTest(accretion_ratio=accretion_ratio):
                self.assertAlmostEqual(
                    model.isco_radius_over_mass,
                    6.0,
                    places=15,
                )
                self.assertAlmostEqual(
                    model.isco_specific_energy,
                    math.sqrt(8.0 / 9.0),
                    places=15,
                )
                self.assertAlmostEqual(
                    model.novikov_thorne_radiative_efficiency,
                    expected_efficiency,
                    places=15,
                )
                self.assertTrue(
                    math.isclose(
                        model.photosphere_height_over_mass(9.0),
                        expected_height,
                        rel_tol=2.0e-15,
                    )
                )

    def test_height_continuity_and_analytic_derivative(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.8,
            eddington_scaled_mass_accretion_rate=0.2,
            outer_radius_over_mass=100.0,
        )
        radius = 3.0 * model.isco_radius_over_mass
        step = 1.0e-5 * radius
        numerical = (
            model.photosphere_height_over_mass(radius + step)
            - model.photosphere_height_over_mass(radius - step)
        ) / (2.0 * step)
        analytic = model.photosphere_height_derivative(radius)
        self.assertTrue(math.isclose(numerical, analytic, rel_tol=5.0e-11))

        edge = model.isco_radius_over_mass
        edge_slope = model.photosphere_height_derivative(edge)
        near = math.nextafter(edge, math.inf)
        self.assertEqual(model.photosphere_height_over_mass(edge), 0.0)
        self.assertGreater(model.photosphere_height_over_mass(near), 0.0)
        self.assertGreater(edge_slope, 0.0)

    def test_upper_and_lower_boyer_lindquist_faces_are_reflections(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.9,
            eddington_scaled_mass_accretion_rate=0.15,
            outer_radius_over_mass=100.0,
        )
        rho = 4.0 * model.isco_radius_over_mass
        upper = model.photosphere_point(rho, UPPER)
        lower = model.photosphere_point(rho, LOWER)
        self.assertEqual(upper.pseudo_cylindrical_radius_over_mass, rho)
        self.assertEqual(lower.pseudo_cylindrical_radius_over_mass, rho)
        self.assertEqual(lower.signed_height_over_mass, -upper.signed_height_over_mass)
        self.assertEqual(lower.radius_over_mass, upper.radius_over_mass)
        self.assertAlmostEqual(
            lower.theta_rad,
            math.pi - upper.theta_rad,
            places=15,
        )
        for point in (upper, lower):
            self.assertAlmostEqual(
                point.radius_over_mass * math.sin(point.theta_rad),
                rho,
                places=14,
            )
            self.assertAlmostEqual(
                point.radius_over_mass * math.cos(point.theta_rad),
                point.signed_height_over_mass,
                places=14,
            )

    def test_signed_surfaces_are_zero_on_face_and_outward_positive(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.5,
            eddington_scaled_mass_accretion_rate=0.1,
            outer_radius_over_mass=100.0,
        )
        rho = 20.0
        epsilon = 1.0e-5
        for face in (UPPER, LOWER):
            point = model.photosphere_point(rho, face)
            on_surface = model.face_signed_surface_over_mass(
                radius_over_mass=point.radius_over_mass,
                theta_rad=point.theta_rad,
                face=face,
            )
            sign = 1.0 if face == UPPER else -1.0
            outward_height = point.signed_height_over_mass + sign * epsilon
            inward_height = point.signed_height_over_mass - sign * epsilon
            outward = model.face_signed_surface_over_mass(
                radius_over_mass=math.hypot(rho, outward_height),
                theta_rad=math.atan2(rho, outward_height),
                face=face,
            )
            inward = model.face_signed_surface_over_mass(
                radius_over_mass=math.hypot(rho, inward_height),
                theta_rad=math.atan2(rho, inward_height),
                face=face,
            )
            with self.subTest(face=face):
                self.assertAlmostEqual(on_surface, 0.0, places=14)
                self.assertGreater(outward, 0.0)
                self.assertLess(inward, 0.0)

    def test_surface_gradient_matches_finite_differences(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.1,
            outer_radius_over_mass=100.0,
        )
        point = model.photosphere_point(20.0, UPPER)
        radius = point.radius_over_mass
        theta = point.theta_rad
        radial_step = 1.0e-6 * radius
        angular_step = 1.0e-7
        gradient = model.face_surface_gradient_covector_bl(
            radius_over_mass=radius,
            theta_rad=theta,
            face=UPPER,
        )
        numerical_radial = (
            model.face_signed_surface_over_mass(
                radius_over_mass=radius + radial_step,
                theta_rad=theta,
                face=UPPER,
            )
            - model.face_signed_surface_over_mass(
                radius_over_mass=radius - radial_step,
                theta_rad=theta,
                face=UPPER,
            )
        ) / (2.0 * radial_step)
        numerical_polar = (
            model.face_signed_surface_over_mass(
                radius_over_mass=radius,
                theta_rad=theta + angular_step,
                face=UPPER,
            )
            - model.face_signed_surface_over_mass(
                radius_over_mass=radius,
                theta_rad=theta - angular_step,
                face=UPPER,
            )
        ) / (2.0 * angular_step)
        self.assertEqual(gradient[0], 0.0)
        self.assertEqual(gradient[3], 0.0)
        self.assertTrue(math.isclose(gradient[1], numerical_radial, rel_tol=2.0e-9))
        self.assertTrue(math.isclose(gradient[2], numerical_polar, rel_tol=2.0e-9))

    def test_unit_face_normal_is_spacelike_and_reflection_symmetric(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.998,
            eddington_scaled_mass_accretion_rate=0.3,
            outer_radius_over_mass=100.0,
        )
        rho = 3.0 * model.isco_radius_over_mass
        normals = {}
        for face in (UPPER, LOWER):
            point = model.photosphere_point(rho, face)
            normal = model.unit_face_normal_covector_bl(rho, face)
            radius = point.radius_over_mass
            sigma = radius * radius + model.dimensionless_spin**2 * math.cos(
                point.theta_rad
            ) ** 2
            delta = (
                radius * radius
                - 2.0 * radius
                + model.dimensionless_spin**2
            )
            squared_norm = (
                delta * normal[1] * normal[1] + normal[2] * normal[2]
            ) / sigma
            with self.subTest(face=face):
                self.assertAlmostEqual(squared_norm, 1.0, places=14)
                self.assertEqual(normal[0], 0.0)
                self.assertEqual(normal[3], 0.0)
            normals[face] = normal
        self.assertAlmostEqual(normals[UPPER][1], normals[LOWER][1], places=14)
        self.assertAlmostEqual(normals[UPPER][2], -normals[LOWER][2], places=14)

    def test_zero_accretion_ratio_recovers_razor_thin_limit(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.998,
            eddington_scaled_mass_accretion_rate=0.0,
            orientation=RETROGRADE,
            outer_radius_over_mass=100.0,
        )
        self.assertEqual(model.asymptotic_pressure_scale_height_over_mass, 0.0)
        self.assertEqual(model.asymptotic_photosphere_height_over_mass, 0.0)
        self.assertEqual(model.maximum_pressure_scale_height_aspect_ratio, 0.0)
        for rho in (model.isco_radius_over_mass, 20.0, 100.0):
            self.assertEqual(model.pressure_scale_height_over_mass(rho), 0.0)
            self.assertEqual(model.photosphere_height_derivative(rho), 0.0)
            upper = model.photosphere_point(rho, UPPER)
            lower = model.photosphere_point(rho, LOWER)
            self.assertEqual(upper.signed_height_over_mass, 0.0)
            self.assertEqual(lower.signed_height_over_mass, -0.0)
            self.assertEqual(upper.radius_over_mass, rho)
            self.assertEqual(lower.radius_over_mass, rho)
            self.assertEqual(upper.theta_rad, 0.5 * math.pi)
            self.assertEqual(lower.theta_rad, 0.5 * math.pi)

    def test_isco_outer_boundaries_and_no_implicit_sidewall(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.1,
            outer_radius_over_mass=60.0,
        )
        self.assertTrue(
            model.contains_pseudo_cylindrical_radius(model.isco_radius_over_mass)
        )
        self.assertTrue(model.contains_pseudo_cylindrical_radius(60.0))
        self.assertFalse(
            model.contains_pseudo_cylindrical_radius(
                math.nextafter(model.isco_radius_over_mass, 0.0)
            )
        )
        self.assertFalse(
            model.contains_pseudo_cylindrical_radius(
                math.nextafter(60.0, math.inf)
            )
        )
        self.assertEqual(
            model.photosphere_height_over_mass(model.isco_radius_over_mass),
            0.0,
        )
        self.assertGreater(model.photosphere_height_over_mass(60.0), 0.0)
        with self.assertRaises(ValueError):
            model.photosphere_height_over_mass(
                math.nextafter(model.isco_radius_over_mass, 0.0)
            )
        with self.assertRaises(ValueError):
            model.photosphere_height_over_mass(math.nextafter(60.0, math.inf))

    def test_exact_aspect_maximum_and_thinness_gate_fail_closed(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.3,
            outer_radius_over_mass=100.0,
            thinness_gate_maximum_h_over_rho=0.2,
        )
        self.assertEqual(
            model.maximum_pressure_scale_height_aspect_radius_over_mass,
            13.5,
        )
        self.assertTrue(
            math.isclose(
                model.maximum_pressure_scale_height_aspect_ratio,
                0.19428090415820629,
                rel_tol=3.0e-15,
            )
        )
        self.assertEqual(
            model.maximum_photosphere_height_aspect_ratio,
            2.0 * model.maximum_pressure_scale_height_aspect_ratio,
        )
        with self.assertRaisesRegex(ValueError, "violates the declared H/rho"):
            StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.0,
                eddington_scaled_mass_accretion_rate=0.3,
                outer_radius_over_mass=100.0,
                thinness_gate_maximum_h_over_rho=0.19,
            )
        with self.assertRaisesRegex(ValueError, "cannot loosen"):
            StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.0,
                eddington_scaled_mass_accretion_rate=0.1,
                thinness_gate_maximum_h_over_rho=(
                    MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO + 1.0e-6
                ),
            )

    def test_extreme_inputs_and_invalid_domains_fail_closed(self) -> None:
        valid_extremes = (
            (0.998, 0.3, PROGRADE),
            (0.998, 0.3, RETROGRADE),
        )
        for spin, ratio, orientation in valid_extremes:
            model = StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=spin,
                eddington_scaled_mass_accretion_rate=ratio,
                orientation=orientation,
                outer_radius_over_mass=100.0,
            )
            with self.subTest(orientation=orientation):
                self.assertTrue(
                    math.isfinite(model.maximum_photosphere_height_aspect_ratio)
                )
                self.assertLessEqual(
                    model.maximum_pressure_scale_height_aspect_ratio,
                    MAXIMUM_ALLOWED_PRESSURE_SCALE_ASPECT_RATIO,
                )

        invalid_kwargs = (
            {"dimensionless_spin": -1.0e-9},
            {"dimensionless_spin": 0.9980000001},
            {"dimensionless_spin": math.nan},
            {"eddington_scaled_mass_accretion_rate": -1.0e-9},
            {"eddington_scaled_mass_accretion_rate": 0.300000001},
            {"eddington_scaled_mass_accretion_rate": math.inf},
            {"orientation": "sideways"},
            {"outer_radius_over_mass": 1.0e6 + 1.0},
            {"outer_radius_over_mass": 1.0},
            {"thinness_gate_maximum_h_over_rho": 0.0},
        )
        for overrides in invalid_kwargs:
            kwargs = {
                "dimensionless_spin": 0.5,
                "eddington_scaled_mass_accretion_rate": 0.1,
                "outer_radius_over_mass": 100.0,
            }
            kwargs.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                StationaryKerrFiniteThicknessCalibration(**kwargs)

    def test_edge_on_observer_gate_is_explicit_and_fail_closed(self) -> None:
        self.assertEqual(validate_calibration_observer_inclination_rad(0.0), 0.0)
        safe = math.acos(2.0 * EDGE_ON_COSINE_NUMERICAL_GUARD)
        self.assertEqual(validate_calibration_observer_inclination_rad(safe), safe)
        for invalid in (
            math.acos(0.5 * EDGE_ON_COSINE_NUMERICAL_GUARD),
            0.5 * math.pi,
            -1.0e-9,
            math.nextafter(0.5 * math.pi, math.inf),
            math.nan,
            True,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_calibration_observer_inclination_rad(invalid)

    def test_surface_and_value_objects_reject_malformed_inputs(self) -> None:
        model = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.5,
            eddington_scaled_mass_accretion_rate=0.1,
            outer_radius_over_mass=100.0,
        )
        point = model.photosphere_point(20.0, UPPER)
        with self.assertRaises(ValueError):
            model.photosphere_point(20.0, "north")
        with self.assertRaises(ValueError):
            model.face_signed_surface_over_mass(
                radius_over_mass=point.radius_over_mass,
                theta_rad=0.0,
                face=UPPER,
            )
        with self.assertRaises(ValueError):
            model.face_surface_gradient_covector_bl(
                radius_over_mass=math.inf,
                theta_rad=point.theta_rad,
                face=UPPER,
            )
        with self.assertRaises(ValueError):
            BoyerLindquistPhotospherePoint(
                pseudo_cylindrical_radius_over_mass=20.0,
                signed_height_over_mass=-1.0,
                radius_over_mass=math.hypot(20.0, 1.0),
                theta_rad=math.atan2(20.0, -1.0),
                face=UPPER,
            )
        with self.assertRaises(ValueError):
            BoyerLindquistPhotospherePoint(
                pseudo_cylindrical_radius_over_mass=20.0,
                signed_height_over_mass=1.0,
                radius_over_mass=20.0,
                theta_rad=math.atan2(20.0, 1.0),
                face=UPPER,
            )


if __name__ == "__main__":
    unittest.main()
