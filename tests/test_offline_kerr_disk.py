from __future__ import annotations

from dataclasses import replace
from decimal import Inexact, ROUND_UP, Rounded, localcontext
import math
import unittest

from offline.geodesic import HamiltonianState
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_zamo_tetrad,
    kerr_zamo_camera_ray,
    stationary_axisymmetric_constants,
)
from offline.kerr_disk import (
    BOLTZMANN_CONSTANT_J_K,
    COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID,
    GRAVITATIONAL_CONSTANT_M3_KG_S2,
    KerrDiskEmitter,
    KerrDiskError,
    LIGHT_SPEED_M_S,
    PLANCK_CONSTANT_J_S,
    SCIENTIFIC_STATUS,
    STEFAN_BOLTZMANN_W_M2_K4,
    StationaryNovikovThorneDisk,
    colour_corrected_planck_specific_intensity_nu,
    observer_to_emitter_frequency_shift_g,
)
from offline.novikov_thorne import (
    PROGRADE,
    RETROGRADE,
    circular_orbit_scalars,
    page_thorne_flux_shape,
)
from offline.spacetime import MinkowskiMetric, bilinear, matrix_vector


SOLAR_MASS_KG = 1.98847e30


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


class OfflineKerrDiskTests(unittest.TestCase):
    def disk(
        self,
        *,
        metric: KerrKerrSchildMetric | None = None,
        mass_kg: float = 1.0e8 * SOLAR_MASS_KG,
        accretion_kg_s: float = 1.0e22,
        orientation: str = PROGRADE,
        colour_correction: float = 1.7,
    ) -> StationaryNovikovThorneDisk:
        return StationaryNovikovThorneDisk(
            metric=metric or KerrKerrSchildMetric(spin_a_m=0.7),
            black_hole_mass_kg=mass_kg,
            mass_accretion_rate_kg_s=accretion_kg_s,
            orientation=orientation,  # type: ignore[arg-type]
            colour_correction=colour_correction,
        )

    def test_scientific_status_is_thin_surface_not_grmhd_or_returning_radiation(self) -> None:
        self.assertEqual(
            SCIENTIFIC_STATUS["classification"],
            "stationary analytic Novikov-Thorne equatorial thin-disk surface emitter",
        )
        self.assertFalse(
            SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"]
        )
        self.assertFalse(SCIENTIFIC_STATUS["includesReturningRadiation"])
        self.assertFalse(SCIENTIFIC_STATUS["includesPhotonPathIntersection"])
        self.assertFalse(SCIENTIFIC_STATUS["includesPolarization"])
        self.assertIn("one disk face", SCIENTIFIC_STATUS["emissionModel"])
        self.assertEqual(
            SCIENTIFIC_STATUS["planckImplementationId"],
            COLOUR_CORRECTED_PLANCK_IMPLEMENTATION_ID,
        )

    def test_schwarzschild_circular_emitter_matches_closed_form(self) -> None:
        mass_m = 2.0
        radius_m = 20.0
        radius_over_mass = radius_m / mass_m
        metric = KerrKerrSchildMetric(mass_m=mass_m, spin_a_m=0.0)
        prograde = self.disk(metric=metric).emitter(radius_m)
        retrograde = self.disk(
            metric=metric,
            orientation=RETROGRADE,
        ).emitter(radius_m)

        expected_ut = 1.0 / math.sqrt(1.0 - 3.0 / radius_over_mass)
        expected_omega = 1.0 / (
            mass_m * radius_over_mass ** 1.5
        )
        self.assertAlmostEqual(prograde.four_velocity[0], expected_ut, places=13)
        self.assertAlmostEqual(
            prograde.angular_velocity_inverse_m,
            expected_omega,
            places=15,
        )
        self.assertAlmostEqual(
            retrograde.angular_velocity_inverse_m,
            -expected_omega,
            places=15,
        )
        self.assertAlmostEqual(prograde.four_velocity[1], 0.0, places=14)
        self.assertAlmostEqual(
            prograde.four_velocity[2],
            expected_ut * expected_omega * radius_m,
            places=13,
        )
        self.assertAlmostEqual(
            retrograde.four_velocity[2],
            -prograde.four_velocity[2],
            places=13,
        )

    def test_cartesian_emitter_is_unit_timelike_and_preserves_orbit_invariants(self) -> None:
        metric = KerrKerrSchildMetric(mass_m=1.7, spin_a_m=0.91)
        disk = self.disk(metric=metric)
        emitter = disk.emitter(12.0 * metric.mass_m, phi_ks_rad=0.43)
        sample = metric.sample(emitter.event)
        self.assertAlmostEqual(
            bilinear(emitter.four_velocity, sample.covariant, emitter.four_velocity),
            -1.0,
            places=12,
        )
        self.assertAlmostEqual(
            metric.oblate_radius_m(emitter.event),
            emitter.radius_m,
            places=12,
        )

        covector = matrix_vector(sample.covariant, emitter.four_velocity)
        energy, angular_momentum = stationary_axisymmetric_constants(
            HamiltonianState(emitter.event, covector)
        )
        orbit = circular_orbit_scalars(
            emitter.radius_over_mass,
            abs(metric.dimensionless_spin),
            PROGRADE,
        )
        self.assertAlmostEqual(energy, orbit.specific_energy, places=12)
        self.assertAlmostEqual(
            angular_momentum,
            orbit.specific_angular_momentum_m * metric.mass_m,
            places=11,
        )

        _time, x_m, y_m, _z_m = emitter.event
        omega = emitter.angular_velocity_inverse_m
        self.assertAlmostEqual(
            emitter.four_velocity[1] / emitter.four_velocity[0],
            -omega * y_m,
            places=13,
        )
        self.assertAlmostEqual(
            emitter.four_velocity[2] / emitter.four_velocity[0],
            omega * x_m,
            places=13,
        )
        self.assertAlmostEqual(emitter.four_velocity[3], 0.0, places=14)

    def test_signed_spin_maps_relative_orientation_to_coordinate_rotation(self) -> None:
        positive = self.disk(
            metric=KerrKerrSchildMetric(spin_a_m=0.7)
        )
        negative = self.disk(
            metric=KerrKerrSchildMetric(spin_a_m=-0.7)
        )
        positive_emitter = positive.emitter(10.0)
        negative_emitter = negative.emitter(10.0)
        self.assertAlmostEqual(
            negative_emitter.angular_velocity_inverse_m,
            -positive_emitter.angular_velocity_inverse_m,
            places=15,
        )
        self.assertAlmostEqual(
            negative_emitter.specific_angular_momentum_m,
            -positive_emitter.specific_angular_momentum_m,
            places=13,
        )
        self.assertAlmostEqual(
            negative_emitter.specific_energy,
            positive_emitter.specific_energy,
            places=14,
        )
        self.assertAlmostEqual(
            negative.thermal_state(10.0).surface_flux_w_m2,
            positive.thermal_state(10.0).surface_flux_w_m2,
            places=2,
        )

    def test_frequency_shift_uses_past_directed_local_frequencies(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        disk = self.disk(metric=metric)
        radius = 10.0
        phi = 0.31
        emitter = disk.emitter(radius, phi_ks_rad=phi)
        observer_tetrad = kerr_bl_zamo_tetrad(
            metric,
            observer_radius_m=radius,
            phi_ks_rad=phi,
        )
        observer_state = kerr_zamo_camera_ray(
            metric,
            observer_radius_m=radius,
            screen_x=0.2,
            screen_y=-0.1,
            phi_ks_rad=phi,
        )
        emitter_state = HamiltonianState(
            event=emitter.event,
            covector=observer_state.covector,
        )
        observer_frequency = math.fsum(
            observer_tetrad.four_velocity[index]
            * observer_state.covector[index]
            for index in range(4)
        )
        emitter_frequency = math.fsum(
            emitter.four_velocity[index] * emitter_state.covector[index]
            for index in range(4)
        )
        expected = observer_frequency / emitter_frequency
        actual = observer_to_emitter_frequency_shift_g(
            metric,
            observer_state,
            observer_tetrad.four_velocity,
            emitter_state,
            emitter,
        )
        self.assertAlmostEqual(observer_frequency, 1.0, places=13)
        self.assertAlmostEqual(actual, expected, places=14)
        self.assertGreater(actual, 0.0)
        for scale in (1.0e-300, 1.0e-150, 1.0e150, 1.0e300):
            scaled_observer = HamiltonianState(
                observer_state.event,
                tuple(value * scale for value in observer_state.covector),
            )
            scaled_emitter = HamiltonianState(
                emitter_state.event,
                tuple(value * scale for value in emitter_state.covector),
            )
            self.assertAlmostEqual(
                observer_to_emitter_frequency_shift_g(
                    metric,
                    scaled_observer,
                    observer_tetrad.four_velocity,
                    scaled_emitter,
                    emitter,
                ),
                actual,
                places=13,
            )

        self.assertAlmostEqual(
            observer_to_emitter_frequency_shift_g(
                metric,
                emitter_state,
                emitter.four_velocity,
                emitter_state,
                emitter,
            ),
            1.0,
            places=15,
        )

    def test_frequency_shift_rejects_non_null_unrelated_or_future_directed_states(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.5)
        disk = self.disk(metric=metric)
        radius = 10.0
        phi = 0.2
        emitter = disk.emitter(radius, phi_ks_rad=phi)
        tetrad = kerr_bl_zamo_tetrad(
            metric,
            observer_radius_m=radius,
            phi_ks_rad=phi,
        )
        first = kerr_zamo_camera_ray(
            metric,
            observer_radius_m=radius,
            screen_x=0.0,
            screen_y=0.0,
            phi_ks_rad=phi,
        )
        second = kerr_zamo_camera_ray(
            metric,
            observer_radius_m=radius,
            screen_x=0.6,
            screen_y=0.1,
            phi_ks_rad=phi,
        )
        with self.assertRaisesRegex(ValueError, "not conserved"):
            observer_to_emitter_frequency_shift_g(
                metric,
                first,
                tetrad.four_velocity,
                HamiltonianState(emitter.event, second.covector),
                emitter,
            )
        for scale in (1.0e-8, 1.0e-100, 1.0e-300):
            scaled_first = HamiltonianState(
                first.event,
                tuple(value * scale for value in first.covector),
            )
            scaled_second = HamiltonianState(
                emitter.event,
                tuple(value * scale for value in second.covector),
            )
            with self.assertRaisesRegex(ValueError, "not conserved"):
                observer_to_emitter_frequency_shift_g(
                    metric,
                    scaled_first,
                    tetrad.four_velocity,
                    scaled_second,
                    emitter,
                )

        non_null = HamiltonianState(
            emitter.event,
            (
                first.covector[0] + 0.1,
                *first.covector[1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "null residual"):
            observer_to_emitter_frequency_shift_g(
                metric,
                non_null,
                tetrad.four_velocity,
                non_null,
                emitter,
            )

        future = HamiltonianState(
            emitter.event,
            tuple(-value for value in first.covector),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "frequency must be positive"):
            observer_to_emitter_frequency_shift_g(
                metric,
                future,
                tetrad.four_velocity,
                future,
                emitter,
            )

        displaced = replace(
            emitter,
            event=(emitter.event[0] + 1.0, *emitter.event[1:]),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            observer_to_emitter_frequency_shift_g(
                metric,
                first,
                tetrad.four_velocity,
                HamiltonianState(emitter.event, first.covector),
                displaced,
            )

    def test_frequency_shift_rejects_same_energy_lz_but_different_carter_constant(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.5)
        disk = self.disk(metric=metric)
        radius = 10.0
        phi = -math.atan2(metric.spin_a_m, radius)
        emitter = disk.emitter(radius, phi_ks_rad=phi)
        tetrad = kerr_bl_zamo_tetrad(
            metric,
            observer_radius_m=radius,
            phi_ks_rad=phi,
        )
        inverse = metric.sample(emitter.event).inverse
        x_m = emitter.event[1]

        def null_state(p_z: float) -> HamiltonianState:
            # At this equatorial +x event, fixing p_t=1 and L_z=x*p_y=0
            # leaves a one-parameter null family.  Varying p_z changes Carter K
            # while preserving the two Killing constants checked separately.
            p_t = 1.0
            p_y = 0.0 / x_m
            quadratic = inverse[1][1]
            linear = 2.0 * (
                inverse[0][1] * p_t
                + inverse[1][2] * p_y
                + inverse[1][3] * p_z
            )
            constant = (
                inverse[0][0] * p_t * p_t
                + 2.0 * inverse[0][2] * p_t * p_y
                + 2.0 * inverse[0][3] * p_t * p_z
                + inverse[2][2] * p_y * p_y
                + 2.0 * inverse[2][3] * p_y * p_z
                + inverse[3][3] * p_z * p_z
            )
            discriminant = linear * linear - 4.0 * quadratic * constant
            self.assertGreater(discriminant, 0.0)
            p_x = (
                -linear - math.sqrt(discriminant)
            ) / (2.0 * quadratic)
            return HamiltonianState(
                emitter.event,
                (p_t, p_x, p_y, p_z),
            )

        equatorial = null_state(0.0)
        inclined = null_state(0.2)
        self.assertEqual(
            stationary_axisymmetric_constants(equatorial),
            stationary_axisymmetric_constants(inclined),
        )
        with self.assertRaisesRegex(ValueError, "Carter constant"):
            observer_to_emitter_frequency_shift_g(
                metric,
                equatorial,
                tetrad.four_velocity,
                inclined,
                emitter,
            )

    def test_page_thorne_flux_has_si_units_and_stefan_boltzmann_temperature(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        mass_kg = 2.0e8 * SOLAR_MASS_KG
        accretion_rate = 3.0e22
        disk = self.disk(
            metric=metric,
            mass_kg=mass_kg,
            accretion_kg_s=accretion_rate,
        )
        radius = 12.0
        thermal = disk.thermal_state(radius)
        shape = page_thorne_flux_shape(radius, 0.7, PROGRADE)
        expected_flux = (
            LIGHT_SPEED_M_S**6
            * accretion_rate
            * shape
            / (
                4.0
                * math.pi
                * GRAVITATIONAL_CONSTANT_M3_KG_S2**2
                * mass_kg**2
            )
        )
        self.assertLess(relative_error(thermal.surface_flux_w_m2, expected_flux), 2e-14)
        self.assertLess(
            relative_error(
                STEFAN_BOLTZMANN_W_M2_K4
                * thermal.effective_temperature_k**4,
                thermal.surface_flux_w_m2,
            ),
            3e-14,
        )
        self.assertEqual(thermal.page_thorne_flux_shape, shape)
        self.assertAlmostEqual(
            thermal.colour_temperature_k,
            disk.colour_correction * thermal.effective_temperature_k,
            places=10,
        )

        twice_accretion = self.disk(
            metric=metric,
            mass_kg=mass_kg,
            accretion_kg_s=2.0 * accretion_rate,
        ).thermal_state(radius)
        twice_mass = self.disk(
            metric=metric,
            mass_kg=2.0 * mass_kg,
            accretion_kg_s=accretion_rate,
        ).thermal_state(radius)
        self.assertLess(
            relative_error(
                twice_accretion.surface_flux_w_m2,
                2.0 * thermal.surface_flux_w_m2,
            ),
            3e-14,
        )
        self.assertLess(
            relative_error(
                twice_mass.surface_flux_w_m2,
                0.25 * thermal.surface_flux_w_m2,
            ),
            3e-14,
        )

    def test_colour_corrected_planck_intensity_is_diluted_in_physical_units(self) -> None:
        disk = self.disk(colour_correction=1.8)
        radius = 10.0
        frequency = 5.0e14
        thermal = disk.thermal_state(radius)
        exponent = (
            PLANCK_CONSTANT_J_S
            * frequency
            / (BOLTZMANN_CONSTANT_J_K * thermal.colour_temperature_k)
        )
        expected = (
            2.0
            * PLANCK_CONSTANT_J_S
            * frequency**3
            / LIGHT_SPEED_M_S**2
            / math.expm1(exponent)
            / disk.colour_correction**4
        )
        intensity = disk.emitted_specific_intensity_nu(radius, frequency)
        self.assertLess(relative_error(intensity, expected), 3e-14)
        self.assertGreater(intensity, 0.0)
        self.assertEqual(
            disk.emitted_specific_intensity_nu(radius, 1.0e300),
            0.0,
        )

        unit_correction = self.disk(colour_correction=1.0).thermal_state(radius)
        self.assertAlmostEqual(
            unit_correction.effective_temperature_k,
            unit_correction.colour_temperature_k,
            places=14,
        )

    def test_planck_binary64_overflow_and_half_subnormal_boundaries(self) -> None:
        overflow_frequency = float.fromhex("0x1.43f5e7c6af360p+79")
        first = overflow_frequency
        for _ in range(4):
            first = math.nextafter(first, 0.0)
        finite_neighbor = math.nextafter(first, 0.0)
        self.assertTrue(
            math.isfinite(
                colour_corrected_planck_specific_intensity_nu(
                    1.0e300,
                    1.0,
                    finite_neighbor,
                )
            )
        )
        frequency = first
        for offset in range(-4, 5):
            with self.subTest(overflow_neighbor=offset), self.assertRaisesRegex(
                KerrDiskError,
                "intensity overflowed",
            ):
                colour_corrected_planck_specific_intensity_nu(
                    1.0e300,
                    1.0,
                    frequency,
                )
            frequency = math.nextafter(frequency, math.inf)

        rounded_subnormal = colour_corrected_planck_specific_intensity_nu(
            1.0e7,
            1.7,
            2.7243194629688948e20,
        )
        self.assertEqual(rounded_subnormal.hex(), math.ulp(0.0).hex())

    def test_planck_decimal_boundary_isolated_from_ambient_context(self) -> None:
        with localcontext() as polluted:
            polluted.rounding = ROUND_UP
            polluted.Emax = 100
            polluted.Emin = -100
            polluted.traps[Inexact] = True
            polluted.traps[Rounded] = True

            finite_upper = colour_corrected_planck_specific_intensity_nu(
                1.0e300,
                1.0,
                float.fromhex("0x1.43f5e7c6af35bp+79"),
            )
            self.assertEqual(
                finite_upper.hex(),
                "0x1.ffffffffffffdp+1023",
            )
            first_minimum_subnormal = (
                colour_corrected_planck_specific_intensity_nu(
                    1.0e7,
                    1.7,
                    float.fromhex("0x1.d8ccb5e268bf4p+67"),
                )
            )
            self.assertEqual(
                first_minimum_subnormal.hex(),
                math.ulp(0.0).hex(),
            )
            specified_subnormal = (
                colour_corrected_planck_specific_intensity_nu(
                    1.0e7,
                    1.7,
                    2.7243194629688948e20,
                )
            )
            self.assertEqual(
                specified_subnormal.hex(),
                math.ulp(0.0).hex(),
            )
            with self.assertRaisesRegex(KerrDiskError, "intensity overflowed"):
                colour_corrected_planck_specific_intensity_nu(
                    1.0e300,
                    1.0,
                    float.fromhex("0x1.43f5e7c6af360p+79"),
                )

    def test_zero_torque_edge_and_zero_accretion_are_exactly_dark(self) -> None:
        disk = self.disk()
        isco = disk.isco_radius_m
        at_edge = disk.thermal_state(isco)
        inside = disk.thermal_state(0.99 * isco)
        self.assertEqual(at_edge.page_thorne_flux_shape, 0.0)
        self.assertEqual(at_edge.surface_flux_w_m2, 0.0)
        self.assertEqual(at_edge.effective_temperature_k, 0.0)
        self.assertEqual(inside.surface_flux_w_m2, 0.0)
        self.assertEqual(disk.emitted_specific_intensity_nu(isco, 1.0e15), 0.0)

        no_accretion = self.disk(accretion_kg_s=0.0)
        dark = no_accretion.thermal_state(10.0)
        self.assertGreater(dark.page_thorne_flux_shape, 0.0)
        self.assertEqual(dark.surface_flux_w_m2, 0.0)
        self.assertEqual(no_accretion.emitted_specific_intensity_nu(10.0, 1.0e15), 0.0)

    def test_strict_validation_and_subextremal_domain(self) -> None:
        with self.assertRaises(TypeError):
            StationaryNovikovThorneDisk(  # type: ignore[arg-type]
                metric=MinkowskiMetric(),
                black_hole_mass_kg=SOLAR_MASS_KG,
                mass_accretion_rate_kg_s=1.0,
            )
        for mass in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(mass=mass):
                with self.assertRaises(ValueError):
                    self.disk(mass_kg=mass)  # type: ignore[arg-type]
        for accretion in (-1.0, math.inf, math.nan, True):
            with self.subTest(accretion=accretion):
                with self.assertRaises(ValueError):
                    self.disk(accretion_kg_s=accretion)  # type: ignore[arg-type]
        for correction in (0.0, 0.9, math.inf, math.nan, True):
            with self.subTest(correction=correction):
                with self.assertRaises(ValueError):
                    self.disk(colour_correction=correction)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.disk(orientation="clockwise")
        with self.assertRaises(ValueError):
            self.disk(metric=KerrKerrSchildMetric(spin_a_m=1.0))

        disk = self.disk()
        with self.assertRaises(ValueError):
            disk.emitter(0.99 * disk.isco_radius_m)
        for radius in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    disk.thermal_state(radius)  # type: ignore[arg-type]
        for frequency in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(frequency=frequency):
                with self.assertRaises(ValueError):
                    disk.emitted_specific_intensity_nu(10.0, frequency)  # type: ignore[arg-type]

        emitter = disk.emitter(10.0)
        with self.assertRaises(ValueError):
            KerrDiskEmitter(
                event=(math.nan, *emitter.event[1:]),
                four_velocity=emitter.four_velocity,
                kerr_mass_m=emitter.kerr_mass_m,
                kerr_spin_a_m=emitter.kerr_spin_a_m,
                radius_m=emitter.radius_m,
                radius_over_mass=emitter.radius_over_mass,
                phi_ks_rad=emitter.phi_ks_rad,
                orientation=emitter.orientation,
                angular_velocity_inverse_m=emitter.angular_velocity_inverse_m,
                specific_energy=emitter.specific_energy,
                specific_angular_momentum_m=emitter.specific_angular_momentum_m,
            )


if __name__ == "__main__":
    unittest.main()
