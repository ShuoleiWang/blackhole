from __future__ import annotations

from dataclasses import replace
import math
import unittest

from offline.disk_atmosphere import (
    DiskAtmosphereError,
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
    apply_angular_emission,
    equatorial_emission_angle_cosine,
)
from offline.geodesic import HamiltonianState, hamiltonian_null_residual
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_vector_to_ks_cartesian,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.spacetime import bilinear, matrix_vector


SOLAR_MASS_KG = 1.98847e30


def _normalized_spatial(
    metric: KerrKerrSchildMetric,
    event: tuple[float, float, float, float],
    vector: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    sample = metric.sample(event)
    norm_squared = bilinear(vector, sample.covariant, vector)
    assert norm_squared > 0.0
    inverse_norm = 1.0 / math.sqrt(norm_squared)
    return tuple(value * inverse_norm for value in vector)  # type: ignore[return-value]


class OfflineDiskAtmosphereTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        self.disk = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
        )
        self.emitter = self.disk.emitter(8.0, phi_ks_rad=0.4)

    def photon_at_mu(self, cosine: float, scale: float = 1.0) -> HamiltonianState:
        radius = self.emitter.radius_m
        phi = self.emitter.phi_ks_rad
        normal = _normalized_spatial(
            self.metric,
            self.emitter.event,
            kerr_bl_vector_to_ks_cartesian(
                (0.0, 0.0, -1.0 / radius, 0.0),
                mass_m=self.metric.mass_m,
                spin_a_m=self.metric.spin_a_m,
                radius_m=radius,
                theta_rad=0.5 * math.pi,
                phi_ks_rad=phi,
            ),
        )
        radial = _normalized_spatial(
            self.metric,
            self.emitter.event,
            kerr_bl_vector_to_ks_cartesian(
                (0.0, 1.0, 0.0, 0.0),
                mass_m=self.metric.mass_m,
                spin_a_m=self.metric.spin_a_m,
                radius_m=radius,
                theta_rad=0.5 * math.pi,
                phi_ks_rad=phi,
            ),
        )
        self.assertAlmostEqual(
            bilinear(normal, self.metric.sample(self.emitter.event).covariant, radial),
            0.0,
            places=13,
        )
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        past_directed = tuple(
            scale
            * (
                -self.emitter.four_velocity[index]
                + cosine * normal[index]
                + sine * radial[index]
            )
            for index in range(4)
        )
        covector = matrix_vector(
            self.metric.sample(self.emitter.event).covariant,
            past_directed,
        )
        state = HamiltonianState(self.emitter.event, covector)
        self.assertLess(hamiltonian_null_residual(self.metric, state), 2.0e-15)
        return state

    def test_covariant_emission_angle_recovers_local_direction_cosine(self) -> None:
        for cosine in (0.0, 0.1, 0.5, 0.9, 1.0):
            recovered = equatorial_emission_angle_cosine(
                self.metric,
                self.photon_at_mu(cosine),
                self.emitter,
            )
            self.assertAlmostEqual(recovered, cosine, places=12)

    def test_emission_angle_is_invariant_under_positive_affine_rescaling(self) -> None:
        reference = equatorial_emission_angle_cosine(
            self.metric,
            self.photon_at_mu(0.37),
            self.emitter,
        )
        for scale in (
            1.0e-300,
            1.0e-150,
            1.0e-9,
            1.0e9,
            1.0e150,
            1.0e300,
        ):
            recovered = equatorial_emission_angle_cosine(
                self.metric,
                self.photon_at_mu(0.37, scale),
                self.emitter,
            )
            self.assertAlmostEqual(recovered, reference, places=12)

    def test_angle_consumer_accepts_near_extremal_valid_emitter(self) -> None:
        cases = (
            (0.9999999999, 1.000000001, 1.7),
            (0.9999999999, 1.0000001, 0.0),
            (-0.9999999999, 1.0000001, 0.0),
        )
        for spin, radius_factor, phi in cases:
            with self.subTest(spin=spin, radius_factor=radius_factor, phi=phi):
                metric = KerrKerrSchildMetric(spin_a_m=spin)
                disk = StationaryNovikovThorneDisk(
                    metric=metric,
                    black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
                    mass_accretion_rate_kg_s=1.0e22,
                )
                emitter = disk.emitter(
                    disk.isco_radius_m * radius_factor,
                    phi_ks_rad=phi,
                )
                normal = _normalized_spatial(
                    metric,
                    emitter.event,
                    kerr_bl_vector_to_ks_cartesian(
                        (0.0, 0.0, -1.0 / emitter.radius_m, 0.0),
                        mass_m=metric.mass_m,
                        spin_a_m=metric.spin_a_m,
                        radius_m=emitter.radius_m,
                        theta_rad=0.5 * math.pi,
                        phi_ks_rad=emitter.phi_ks_rad,
                    ),
                )
                past_directed = tuple(
                    -emitter.four_velocity[index] + normal[index]
                    for index in range(4)
                )
                state = HamiltonianState(
                    emitter.event,
                    matrix_vector(
                        metric.sample(emitter.event).covariant,
                        past_directed,
                    ),
                )
                self.assertAlmostEqual(
                    equatorial_emission_angle_cosine(metric, state, emitter),
                    1.0,
                    places=8,
                )

    def test_linear_limb_darkening_preserves_one_face_flux(self) -> None:
        law = FluxConservingLinearLimbDarkening()
        self.assertEqual(law.coefficient, 1.5)
        self.assertAlmostEqual(law.intensity_multiplier(0.0), 0.5)
        self.assertAlmostEqual(law.intensity_multiplier(1.0), 1.25)
        intervals = 100_000
        step = 1.0 / intervals
        integral = math.fsum(
            2.0
            * (index + 0.5)
            * step
            * law.intensity_multiplier((index + 0.5) * step)
            * step
            for index in range(intervals)
        )
        self.assertAlmostEqual(integral, 1.0, places=10)
        self.assertLess(law.intensity_multiplier(0.0), 1.0)
        self.assertGreater(law.intensity_multiplier(1.0), 1.0)

    def test_isotropic_and_linear_laws_scale_nonnegative_intensity(self) -> None:
        self.assertEqual(
            apply_angular_emission(3.0, 0.2, IsotropicAngularEmission()),
            3.0,
        )
        law = FluxConservingLinearLimbDarkening(2.06)
        self.assertAlmostEqual(
            apply_angular_emission(3.0, 0.8, law),
            3.0 * law.intensity_multiplier(0.8),
        )
        self.assertEqual(apply_angular_emission(0.0, 0.8, law), 0.0)

    def test_strict_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            FluxConservingLinearLimbDarkening(-0.1)
        with self.assertRaises(TypeError):
            IsotropicAngularEmission("")
        with self.assertRaises(TypeError):
            FluxConservingLinearLimbDarkening(
                1.5,
                "claimed-full-atmosphere/v99",
            )
        with self.assertRaises(ValueError):
            apply_angular_emission(-1.0, 0.5, IsotropicAngularEmission())
        with self.assertRaises(ValueError):
            apply_angular_emission(1.0, 1.1, IsotropicAngularEmission())
        bad = HamiltonianState(
            tuple(value + (1.0 if index == 1 else 0.0) for index, value in enumerate(self.emitter.event)),
            self.photon_at_mu(0.5).covector,
        )
        with self.assertRaises(ValueError):
            equatorial_emission_angle_cosine(self.metric, bad, self.emitter)
        extreme = FluxConservingLinearLimbDarkening(9.0e307)
        self.assertAlmostEqual(extreme.intensity_multiplier(0.5), 0.75)
        subnormal = FluxConservingLinearLimbDarkening(math.ulp(0.0))
        self.assertEqual(subnormal.normalization, 1.0)
        self.assertEqual(subnormal.intensity_multiplier(0.5), 1.0)
        with self.assertRaises(DiskAtmosphereError):
            apply_angular_emission(
                1.5e308,
                1.0,
                FluxConservingLinearLimbDarkening(),
            )
        with self.assertRaisesRegex(ValueError, "specific energy"):
            replace(
                self.emitter,
                specific_energy=0.5 * self.emitter.specific_energy,
            )
        different_radius = self.disk.emitter(
            7.0,
            phi_ks_rad=self.emitter.phi_ks_rad,
        )
        with self.assertRaisesRegex(ValueError, "event"):
            replace(self.emitter, event=different_radius.event)
        with self.assertRaisesRegex(ValueError, "four-velocity"):
            replace(
                self.emitter,
                four_velocity=tuple(
                    1.001 * value for value in self.emitter.four_velocity
                ),
            )


if __name__ == "__main__":
    unittest.main()
