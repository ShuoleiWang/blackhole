from __future__ import annotations

import math
import unittest

from offline.novikov_thorne import (
    PROGRADE,
    RETROGRADE,
    SCIENTIFIC_STATUS,
    circular_orbit_scalars,
    kerr_isco_radius_m,
    orbital_angular_velocity_m,
    page_thorne_flux_shape,
    specific_angular_momentum_m,
    specific_energy,
)


def schwarzschild_page_thorne_flux_shape(radius_m: float) -> float:
    """Independent closed form of the Page--Thorne integral for a=0."""

    if radius_m <= 6.0:
        return 0.0
    root_radius = math.sqrt(radius_m)
    root_isco = math.sqrt(6.0)
    root_three = math.sqrt(3.0)
    logarithm = math.log(
        ((root_radius - root_three) / (root_radius + root_three))
        / ((root_isco - root_three) / (root_isco + root_three))
    )
    integral = (
        root_radius
        - root_isco
        - 0.5 * root_three * logarithm
    )
    return (
        1.5
        * radius_m ** -2.5
        * integral
        / (radius_m - 3.0)
    )


class NovikovThorneTests(unittest.TestCase):
    def test_scientific_status_is_stationary_thin_disk_not_grmhd(self) -> None:
        self.assertEqual(
            SCIENTIFIC_STATUS["classification"],
            "stationary analytic Novikov-Thorne thin-disk scalar oracle",
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["spacetime"],
            "exact stationary Kerr in Boyer-Lindquist coordinates",
        )
        self.assertIs(
            SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"],
            False,
        )
        self.assertIs(SCIENTIFIC_STATUS["includesPhotonTransfer"], False)
        self.assertIn("GRMHD", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["classification"] = "mutable"

    def test_kerr_isco_matches_benchmark_values_and_branch_order(self) -> None:
        benchmarks = (
            (0.0, 6.0, 6.0),
            (0.5, 4.233002529530826, 7.554584714512358),
            (0.9, 2.320883041761887, 8.717352279606489),
            (0.998, 1.2369706551751847, 8.99437445480357),
        )
        for spin, expected_prograde, expected_retrograde in benchmarks:
            with self.subTest(spin=spin):
                prograde = kerr_isco_radius_m(spin, PROGRADE)
                retrograde = kerr_isco_radius_m(spin, RETROGRADE)
                self.assertAlmostEqual(prograde, expected_prograde, places=14)
                self.assertAlmostEqual(retrograde, expected_retrograde, places=14)
                self.assertLessEqual(prograde, 6.0)
                self.assertGreaterEqual(retrograde, 6.0)

    def test_schwarzschild_isco_orbit_scalars_are_exact_limits(self) -> None:
        expected_omega = 1.0 / (6.0 * math.sqrt(6.0))
        expected_energy = math.sqrt(8.0 / 9.0)
        expected_angular_momentum = 2.0 * math.sqrt(3.0)
        prograde = circular_orbit_scalars(6.0, 0.0, PROGRADE)
        retrograde = circular_orbit_scalars(6.0, 0.0, RETROGRADE)

        self.assertAlmostEqual(prograde.omega_m, expected_omega, places=15)
        self.assertAlmostEqual(
            prograde.specific_energy,
            expected_energy,
            places=15,
        )
        self.assertAlmostEqual(
            prograde.specific_angular_momentum_m,
            expected_angular_momentum,
            places=15,
        )
        self.assertEqual(retrograde.omega_m, -prograde.omega_m)
        self.assertEqual(
            retrograde.specific_angular_momentum_m,
            -prograde.specific_angular_momentum_m,
        )
        self.assertEqual(retrograde.specific_energy, prograde.specific_energy)

    def test_public_scalar_accessors_match_aggregate_and_orientations(self) -> None:
        for spin in (0.0, 0.5, 0.9, 0.998, 0.999999):
            for orientation in (PROGRADE, RETROGRADE):
                radius = 2.0 * kerr_isco_radius_m(spin, orientation)
                with self.subTest(spin=spin, orientation=orientation):
                    orbit = circular_orbit_scalars(radius, spin, orientation)
                    self.assertEqual(
                        orbital_angular_velocity_m(radius, spin, orientation),
                        orbit.omega_m,
                    )
                    self.assertEqual(
                        specific_energy(radius, spin, orientation),
                        orbit.specific_energy,
                    )
                    self.assertEqual(
                        specific_angular_momentum_m(radius, spin, orientation),
                        orbit.specific_angular_momentum_m,
                    )
                    sign = 1.0 if orientation == PROGRADE else -1.0
                    self.assertGreater(sign * orbit.omega_m, 0.0)
                    self.assertGreater(
                        sign * orbit.specific_angular_momentum_m,
                        0.0,
                    )
                    self.assertGreater(orbit.specific_energy, 0.0)
                    self.assertLess(orbit.specific_energy, 1.0)
                    self.assertGreater(
                        orbit.specific_energy
                        - orbit.omega_m * orbit.specific_angular_momentum_m,
                        0.0,
                    )

    def test_circular_orbit_first_law_dE_equals_omega_dL(self) -> None:
        for spin in (0.0, 0.5, 0.9):
            for orientation in (PROGRADE, RETROGRADE):
                radius = 2.5 * kerr_isco_radius_m(spin, orientation)
                step = 1.0e-5 * radius
                energy_derivative = (
                    specific_energy(radius + step, spin, orientation)
                    - specific_energy(radius - step, spin, orientation)
                ) / (2.0 * step)
                angular_momentum_derivative = (
                    specific_angular_momentum_m(
                        radius + step,
                        spin,
                        orientation,
                    )
                    - specific_angular_momentum_m(
                        radius - step,
                        spin,
                        orientation,
                    )
                ) / (2.0 * step)
                omega = orbital_angular_velocity_m(radius, spin, orientation)
                with self.subTest(spin=spin, orientation=orientation):
                    self.assertTrue(
                        math.isclose(
                            energy_derivative,
                            omega * angular_momentum_derivative,
                            rel_tol=2.0e-9,
                            abs_tol=2.0e-11,
                        )
                    )

    def test_orbit_scalars_recover_newtonian_far_field(self) -> None:
        radius = 1.0e8
        for spin in (0.0, 0.5, 0.9, 0.998):
            for orientation in (PROGRADE, RETROGRADE):
                sign = 1.0 if orientation == PROGRADE else -1.0
                orbit = circular_orbit_scalars(radius, spin, orientation)
                with self.subTest(spin=spin, orientation=orientation):
                    self.assertTrue(
                        math.isclose(
                            orbit.omega_m * radius**1.5,
                            sign,
                            rel_tol=2.0e-12,
                        )
                    )
                    self.assertTrue(
                        math.isclose(
                            orbit.specific_angular_momentum_m / math.sqrt(radius),
                            sign,
                            rel_tol=2.0e-8,
                        )
                    )
                    self.assertTrue(
                        math.isclose(
                            1.0 - orbit.specific_energy,
                            0.5 / radius,
                            rel_tol=2.0e-4,
                        )
                    )

    def test_page_thorne_flux_matches_independent_schwarzschild_closed_form(self) -> None:
        for radius in (6.001, 7.0, 10.0, 20.0, 100.0, 1.0e6):
            expected = schwarzschild_page_thorne_flux_shape(radius)
            actual = page_thorne_flux_shape(radius, 0.0, PROGRADE)
            with self.subTest(radius=radius):
                self.assertTrue(
                    math.isclose(
                        actual,
                        expected,
                        rel_tol=3.0e-9,
                        abs_tol=1.0e-24,
                    )
                )
                self.assertEqual(
                    page_thorne_flux_shape(radius, 0.0, RETROGRADE),
                    actual,
                )

    def test_page_thorne_flux_preserves_nonzero_spin_golden_values(self) -> None:
        # These values were generated independently from the conservation-law
        # integral before the logarithmic implementation became the main path.
        cases = (
            (0.1, PROGRADE, 1.2, 1.1379749538925641e-4),
            (0.1, RETROGRADE, 2.0, 1.2452294824963358e-4),
            (0.5, PROGRADE, 1.5, 5.317609943664694e-4),
            (0.5, RETROGRADE, 10.0, 1.972442185972739e-6),
            (0.9, PROGRADE, 2.0, 3.183955694889092e-3),
            (0.9, RETROGRADE, 3.0, 2.3604432685029735e-5),
            (0.998, PROGRADE, 10.0, 4.2695738252421716e-4),
            (0.998, RETROGRADE, 100.0, 1.7551703164248845e-9),
        )
        for spin, orientation, radius_factor, expected in cases:
            radius = radius_factor * kerr_isco_radius_m(spin, orientation)
            actual = page_thorne_flux_shape(radius, spin, orientation)
            with self.subTest(
                spin=spin,
                orientation=orientation,
                radius_factor=radius_factor,
            ):
                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=5.0e-12)
                )

    def test_page_thorne_zero_torque_edge_is_quadratic_and_nonnegative(self) -> None:
        for spin in (0.0, 0.5, 0.9, 0.998):
            for orientation in (PROGRADE, RETROGRADE):
                isco = kerr_isco_radius_m(spin, orientation)
                epsilon = 1.0e-5 * isco
                first = page_thorne_flux_shape(
                    isco + epsilon,
                    spin,
                    orientation,
                )
                second = page_thorne_flux_shape(
                    isco + 2.0 * epsilon,
                    spin,
                    orientation,
                )
                with self.subTest(spin=spin, orientation=orientation):
                    self.assertEqual(
                        page_thorne_flux_shape(0.9 * isco, spin, orientation),
                        0.0,
                    )
                    self.assertEqual(
                        page_thorne_flux_shape(isco, spin, orientation),
                        0.0,
                    )
                    self.assertGreater(first, 0.0)
                    self.assertGreater(second, first)
                    self.assertTrue(
                        math.isclose(second / first, 4.0, rel_tol=5.0e-4)
                    )

    def test_page_thorne_flux_is_finite_over_spin_branches(self) -> None:
        for spin in (0.0, 0.5, 0.9, 0.998, 0.999999):
            for orientation in (PROGRADE, RETROGRADE):
                isco = kerr_isco_radius_m(spin, orientation)
                for factor in (1.000001, 1.01, 1.5, 2.0, 10.0, 1000.0):
                    flux = page_thorne_flux_shape(
                        factor * isco,
                        spin,
                        orientation,
                    )
                    with self.subTest(
                        spin=spin,
                        orientation=orientation,
                        factor=factor,
                    ):
                        self.assertTrue(math.isfinite(flux))
                        self.assertGreaterEqual(flux, 0.0)

    def test_page_thorne_flux_recovers_universal_far_field(self) -> None:
        radius = 1.0e6
        newtonian_limit = 1.5 / radius**3
        for spin in (0.0, 0.5, 0.9, 0.998):
            for orientation in (PROGRADE, RETROGRADE):
                flux = page_thorne_flux_shape(radius, spin, orientation)
                with self.subTest(spin=spin, orientation=orientation):
                    self.assertTrue(
                        math.isclose(
                            flux,
                            newtonian_limit,
                            rel_tol=5.0e-3,
                        )
                    )

    def test_page_thorne_flux_covers_extreme_spin_and_full_radius_domain(self) -> None:
        extreme_spin = math.nextafter(1.0, 0.0)
        for orientation in (PROGRADE, RETROGRADE):
            isco = kerr_isco_radius_m(extreme_spin, orientation)
            near_edge = page_thorne_flux_shape(
                isco * (1.0 + 1.0e-12),
                extreme_spin,
                orientation,
            )
            ordinary = page_thorne_flux_shape(
                2.0 * isco,
                extreme_spin,
                orientation,
            )
            with self.subTest(orientation=orientation):
                self.assertTrue(math.isfinite(near_edge))
                self.assertGreater(near_edge, 0.0)
                self.assertTrue(math.isfinite(ordinary))
                self.assertGreater(ordinary, 0.0)

        high_spin_far_field = page_thorne_flux_shape(
            1.0e6,
            0.999999999999,
            PROGRADE,
        )
        self.assertTrue(math.isfinite(high_spin_far_field))
        self.assertTrue(
            math.isclose(
                high_spin_far_field,
                1.5e-18,
                rel_tol=5.0e-3,
            )
        )

        huge_schwarzschild = page_thorne_flux_shape(1.0e12, 0.0, PROGRADE)
        self.assertTrue(
            math.isclose(
                huge_schwarzschild,
                schwarzschild_page_thorne_flux_shape(1.0e12),
                rel_tol=2.0e-15,
            )
        )

        maximum_radius = float.fromhex("0x1.fffffffffffffp+1023")
        for orientation in (PROGRADE, RETROGRADE):
            with self.subTest(maximum_radius_orientation=orientation):
                self.assertEqual(
                    page_thorne_flux_shape(
                        maximum_radius,
                        extreme_spin,
                        orientation,
                    ),
                    0.0,
                )

    def test_strict_input_validation_and_stable_orbit_domain(self) -> None:
        bad_numbers = (True, False, None, "0.5", math.nan, math.inf, -math.inf)
        for value in bad_numbers:
            with self.subTest(spin=value):
                with self.assertRaises(ValueError):
                    kerr_isco_radius_m(value)
        for spin in (-1.0e-12, 1.0, 1.01):
            with self.subTest(spin=spin):
                with self.assertRaises(ValueError):
                    kerr_isco_radius_m(spin)

        for orientation in (None, True, 1, "co-rotating", [], {}):
            with self.subTest(orientation=repr(orientation)):
                with self.assertRaises(ValueError):
                    kerr_isco_radius_m(0.5, orientation)

        for radius in (True, False, None, "6", math.nan, math.inf, -1.0, 0.0):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    circular_orbit_scalars(radius, 0.0)
                with self.assertRaises(ValueError):
                    page_thorne_flux_shape(radius, 0.0)

        prograde_isco = kerr_isco_radius_m(0.9, PROGRADE)
        with self.assertRaisesRegex(ValueError, "at or outside"):
            circular_orbit_scalars(0.99 * prograde_isco, 0.9, PROGRADE)
        self.assertEqual(
            page_thorne_flux_shape(0.99 * prograde_isco, 0.9, PROGRADE),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
