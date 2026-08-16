from __future__ import annotations

import hashlib
import json
import math
import unittest

from offline.geodesic import HamiltonianState, hamiltonian_null_residual
from offline.kerr import (
    KerrKerrSchildMetric,
    kerr_bl_vector_to_ks_cartesian,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    BacksidePhotonError,
    IMPLEMENTATION_ID,
    PRIMARY_SOURCE_HASH_SEMANTICS,
    PRIMARY_SOURCE_REFERENCE_SHA256,
    PRIMARY_SOURCE_URL,
    SCIENTIFIC_STATUS,
    KerrFiniteThicknessFaceEmitter,
)
from offline.novikov_thorne import PROGRADE, RETROGRADE, circular_orbit_scalars
from offline.spacetime import Vector4, bilinear, matrix_vector


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


class OfflineKerrFiniteThicknessEmitterTests(unittest.TestCase):
    def calibration(
        self,
        *,
        spin: float = 0.7,
        dotm: float = 0.03,
        orientation: str = PROGRADE,
        outer_radius: float = 100.0,
    ) -> StationaryKerrFiniteThicknessCalibration:
        return StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=dotm,
            orientation=orientation,  # type: ignore[arg-type]
            outer_radius_over_mass=outer_radius,
        )

    def emitter(
        self,
        *,
        spin: float = 0.7,
        signed_spin: float | None = None,
        mass_m: float = 2.0,
        dotm: float = 0.03,
        orientation: str = PROGRADE,
        face: str = UPPER,
        rho: float = 10.0,
        phi: float = 0.4,
    ) -> KerrFiniteThicknessFaceEmitter:
        calibration = self.calibration(
            spin=spin,
            dotm=dotm,
            orientation=orientation,
            outer_radius=max(100.0, rho * 2.0),
        )
        metric_spin = spin if signed_spin is None else signed_spin
        metric = KerrKerrSchildMetric(
            mass_m=mass_m,
            spin_a_m=metric_spin * mass_m,
        )
        return KerrFiniteThicknessFaceEmitter(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,  # type: ignore[arg-type]
            phi_ks_rad=phi,
        )

    def photon_at_mu(
        self,
        emitter: KerrFiniteThicknessFaceEmitter,
        mu: float,
        *,
        scale: float = 1.0,
    ) -> HamiltonianState:
        self.assertGreaterEqual(mu, -1.0)
        self.assertLessEqual(mu, 1.0)
        metric = emitter.metric
        point = emitter.photosphere_point
        radius_m = point.radius_over_mass * metric.mass_m
        sample = metric.sample(emitter.event)

        # Construct a second local spatial basis vector from the public exact
        # BL Jacobian, then Gram-project it against u and the face normal.
        raw_tangent = kerr_bl_vector_to_ks_cartesian(
            (0.0, 1.0, 0.0, 0.0),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=radius_m,
            theta_rad=point.theta_rad,
            phi_ks_rad=emitter.phi_ks_rad,
        )
        tangent_u_projection = bilinear(
            raw_tangent,
            sample.covariant,
            emitter.four_velocity,
        )
        tangent = tuple(
            raw_tangent[index]
            + tangent_u_projection * emitter.four_velocity[index]
            for index in range(4)
        )
        tangent_n_projection = bilinear(
            tangent,  # type: ignore[arg-type]
            sample.covariant,
            emitter.outward_unit_normal,
        )
        tangent = tuple(
            tangent[index]
            - tangent_n_projection * emitter.outward_unit_normal[index]
            for index in range(4)
        )
        tangent_norm = bilinear(
            tangent,  # type: ignore[arg-type]
            sample.covariant,
            tangent,  # type: ignore[arg-type]
        )
        self.assertGreater(tangent_norm, 0.0)
        tangent = tuple(value / math.sqrt(tangent_norm) for value in tangent)

        tangent_weight = math.sqrt(max(0.0, 1.0 - mu * mu))
        past_directed_vector: Vector4 = tuple(  # type: ignore[assignment]
            -emitter.four_velocity[index]
            - mu * emitter.outward_unit_normal[index]
            + tangent_weight * tangent[index]
            for index in range(4)
        )
        covector = matrix_vector(sample.covariant, past_directed_vector)
        scaled_covector: Vector4 = tuple(  # type: ignore[assignment]
            scale * value for value in covector
        )
        state = HamiltonianState(emitter.event, scaled_covector)
        self.assertLess(hamiltonian_null_residual(metric, state), 2.0e-13)
        return state

    def test_scientific_status_and_source_hash_semantics_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(SCIENTIFIC_STATUS["primarySource"], PRIMARY_SOURCE_URL)
        self.assertFalse(SCIENTIFIC_STATUS["isOffEquatorialGeodesic"])
        self.assertFalse(SCIENTIFIC_STATUS["isHydrostaticVerticalStructureSolution"])
        self.assertFalse(SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"])
        self.assertFalse(SCIENTIFIC_STATUS["includesReturningRadiation"])
        self.assertFalse(SCIENTIFIC_STATUS["includesSolvedAtmosphere"])
        self.assertIn("not a hash", PRIMARY_SOURCE_HASH_SEMANTICS)

        emitter = self.emitter()
        descriptor = emitter.model_descriptor()
        canonical_source = json.dumps(
            descriptor["sourceReference"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_source).hexdigest(),
            PRIMARY_SOURCE_REFERENCE_SHA256,
        )
        self.assertEqual(
            descriptor["sourceReferenceHashSemantics"],
            PRIMARY_SOURCE_HASH_SEMANTICS,
        )
        canonical_descriptor = json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_descriptor).hexdigest(),
            emitter.model_descriptor_sha256,
        )
        self.assertEqual(descriptor["certifiedFrame"]["eventKs"], list(emitter.event))
        self.assertEqual(
            descriptor["certifiedFrame"]["fourVelocityKs"],
            list(emitter.four_velocity),
        )

    def test_actual_face_metric_normalizes_velocity_and_normal(self) -> None:
        emitter = self.emitter(spin=0.998, signed_spin=0.998, dotm=0.3, rho=4.0)
        sample = emitter.metric.sample(emitter.event)
        self.assertGreater(emitter.four_velocity[0], 0.0)
        self.assertAlmostEqual(
            bilinear(emitter.four_velocity, sample.covariant, emitter.four_velocity),
            -1.0,
            delta=2.0e-13,
        )
        self.assertAlmostEqual(
            bilinear(
                emitter.outward_unit_normal,
                sample.covariant,
                emitter.outward_unit_normal,
            ),
            1.0,
            delta=2.0e-13,
        )
        self.assertAlmostEqual(
            bilinear(
                emitter.four_velocity,
                sample.covariant,
                emitter.outward_unit_normal,
            ),
            0.0,
            delta=2.0e-13,
        )
        self.assertEqual(
            matrix_vector(sample.covariant, emitter.outward_unit_normal),
            emitter.outward_unit_normal_covector,
        )

        orbit = circular_orbit_scalars(4.0, 0.998, PROGRADE)
        self.assertAlmostEqual(
            emitter.angular_velocity_inverse_m,
            orbit.omega_m / emitter.metric.mass_m,
            places=15,
        )
        # The actual off-plane normalization must not silently reuse the
        # equatorial value 1/(E-Omega L).
        equatorial_u_t = 1.0 / (
            orbit.specific_energy
            - orbit.omega_m * orbit.specific_angular_momentum_m
        )
        self.assertGreater(
            abs(emitter.four_velocity_bl_time_component - equatorial_u_t),
            1.0e-5,
        )

    def test_ks_normal_covector_pulls_back_to_calibration_bl_normal(self) -> None:
        for signed_spin in (0.0, 0.998, -0.998):
            for face in (UPPER, LOWER):
                with self.subTest(signed_spin=signed_spin, face=face):
                    emitter = self.emitter(
                        spin=abs(signed_spin),
                        signed_spin=signed_spin,
                        dotm=0.2,
                        face=face,
                        rho=12.0,
                    )
                    metric = emitter.metric
                    point = emitter.photosphere_point
                    radius_m = point.radius_over_mass * metric.mass_m
                    radial_basis = kerr_bl_vector_to_ks_cartesian(
                        (0.0, 1.0, 0.0, 0.0),
                        mass_m=metric.mass_m,
                        spin_a_m=metric.spin_a_m,
                        radius_m=radius_m,
                        theta_rad=point.theta_rad,
                        phi_ks_rad=emitter.phi_ks_rad,
                    )
                    polar_basis_over_mass = kerr_bl_vector_to_ks_cartesian(
                        (0.0, 0.0, 1.0 / metric.mass_m, 0.0),
                        mass_m=metric.mass_m,
                        spin_a_m=metric.spin_a_m,
                        radius_m=radius_m,
                        theta_rad=point.theta_rad,
                        phi_ks_rad=emitter.phi_ks_rad,
                    )
                    expected_bl = emitter.calibration.unit_face_normal_covector_bl(
                        emitter.pseudo_cylindrical_radius_over_mass,
                        face,
                    )
                    radial_pullback = math.fsum(
                        emitter.outward_unit_normal_covector[index]
                        * radial_basis[index]
                        for index in range(4)
                    )
                    polar_pullback = math.fsum(
                        emitter.outward_unit_normal_covector[index]
                        * polar_basis_over_mass[index]
                        for index in range(4)
                    )
                    self.assertAlmostEqual(
                        radial_pullback,
                        expected_bl[1],
                        delta=2.0e-13,
                    )
                    self.assertAlmostEqual(
                        polar_pullback,
                        expected_bl[2],
                        delta=2.0e-13,
                    )

    def test_upper_lower_are_equatorial_reflections_with_bound_faces(self) -> None:
        common = dict(spin=0.998, signed_spin=-0.998, dotm=0.2, rho=5.0, phi=-0.7)
        upper = self.emitter(face=UPPER, **common)
        lower = self.emitter(face=LOWER, **common)
        self.assertEqual(upper.metric, lower.metric)
        self.assertEqual(upper.calibration, lower.calibration)
        self.assertEqual(upper.face, UPPER)
        self.assertEqual(lower.face, LOWER)
        for index in (0, 1, 2):
            self.assertAlmostEqual(upper.event[index], lower.event[index], places=13)
            self.assertAlmostEqual(
                upper.four_velocity[index],
                lower.four_velocity[index],
                places=13,
            )
            self.assertAlmostEqual(
                upper.outward_unit_normal[index],
                lower.outward_unit_normal[index],
                places=13,
            )
        self.assertAlmostEqual(upper.event[3], -lower.event[3], places=13)
        self.assertAlmostEqual(
            upper.four_velocity[3],
            -lower.four_velocity[3],
            places=13,
        )
        self.assertAlmostEqual(
            upper.outward_unit_normal[3],
            -lower.outward_unit_normal[3],
            places=13,
        )
        self.assertGreater(upper.outward_unit_normal[3], 0.0)
        self.assertLess(lower.outward_unit_normal[3], 0.0)
        self.assertNotEqual(
            upper.model_descriptor_sha256,
            lower.model_descriptor_sha256,
        )
        self.assertEqual(upper.model_descriptor()["surfaceEvent"]["face"], UPPER)
        self.assertEqual(lower.model_descriptor()["surfaceEvent"]["face"], LOWER)

    def test_negative_metric_spin_and_retrograde_orientation_are_bound(self) -> None:
        prograde = self.emitter(
            spin=0.7,
            signed_spin=-0.7,
            orientation=PROGRADE,
            rho=12.0,
        )
        retrograde = self.emitter(
            spin=0.7,
            signed_spin=-0.7,
            orientation=RETROGRADE,
            rho=12.0,
        )
        self.assertLess(prograde.angular_velocity_inverse_m, 0.0)
        self.assertGreater(retrograde.angular_velocity_inverse_m, 0.0)
        self.assertEqual(
            prograde.model_descriptor()["calibration"]["orientation"],
            PROGRADE,
        )
        self.assertEqual(
            retrograde.model_descriptor()["calibration"]["orientation"],
            RETROGRADE,
        )

    def test_small_dotm_converges_to_thin_disk_event_velocity_and_normal(self) -> None:
        metric = KerrKerrSchildMetric(mass_m=2.0, spin_a_m=1.4)
        rho = 10.0
        phi = 0.4
        thin_disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=1.0,
            mass_accretion_rate_kg_s=0.0,
        )
        thin = thin_disk.emitter(rho * metric.mass_m, phi_ks_rad=phi)
        errors: list[tuple[float, float, float]] = []
        for dotm in (1.0e-3, 1.0e-5, 1.0e-7):
            calibration = self.calibration(spin=0.7, dotm=dotm)
            thick = KerrFiniteThicknessFaceEmitter(
                metric=metric,
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=rho,
                face=UPPER,
                phi_ks_rad=phi,
            )
            event_error = max(
                abs(thick.event[index] - thin.event[index])
                for index in range(4)
            )
            velocity_error = max(
                abs(thick.four_velocity[index] - thin.four_velocity[index])
                for index in range(4)
            )
            normal_error = max(
                abs(thick.outward_unit_normal[index] - (1.0 if index == 3 else 0.0))
                for index in range(4)
            )
            errors.append((event_error, velocity_error, normal_error))
        for first, second in zip(errors, errors[1:]):
            for first_error, second_error in zip(first, second):
                self.assertLess(second_error, first_error)
        self.assertLess(errors[-1][0], 1.0e-5)
        self.assertLess(errors[-1][1], 1.0e-11)
        self.assertLess(errors[-1][2], 1.0e-6)

    def test_near_calibrated_extreme_spin_and_near_isco_remain_certified(self) -> None:
        calibration = self.calibration(spin=0.998, dotm=0.3, outer_radius=20.0)
        rho = calibration.isco_radius_over_mass * (1.0 + 1.0e-9)
        emitter = KerrFiniteThicknessFaceEmitter(
            metric=KerrKerrSchildMetric(spin_a_m=0.998),
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=UPPER,
        )
        sample = emitter.metric.sample(emitter.event)
        self.assertGreater(emitter.photosphere_point.signed_height_over_mass, 0.0)
        self.assertAlmostEqual(
            bilinear(emitter.four_velocity, sample.covariant, emitter.four_velocity),
            -1.0,
            delta=2.0e-9,
        )
        self.assertAlmostEqual(
            bilinear(
                emitter.outward_unit_normal,
                sample.covariant,
                emitter.outward_unit_normal,
            ),
            1.0,
            delta=2.0e-9,
        )

    def test_signed_photon_projection_recovers_frequency_and_outgoing_mu(self) -> None:
        emitter = self.emitter(spin=0.998, signed_spin=0.998, rho=4.0, dotm=0.2)
        state = self.photon_at_mu(emitter, 0.37, scale=3.5)
        projection = emitter.project_past_directed_photon(state)
        self.assertEqual(projection.face_classification, "outgoing")
        self.assertAlmostEqual(projection.local_frequency, 3.5, delta=2.0e-12)
        self.assertAlmostEqual(projection.outgoing_cosine, 0.37, delta=2.0e-12)
        self.assertLess(projection.null_residual, 2.0e-13)

    def test_projection_gates_and_mu_are_positive_affine_scale_invariant(self) -> None:
        emitter = self.emitter()
        reference_mu = None
        for scale in (1.0e-300, 1.0e-150, 1.0e-9, 1.0, 1.0e9, 1.0e150, 1.0e300):
            with self.subTest(scale=scale):
                state = self.photon_at_mu(emitter, 0.61, scale=scale)
                projection = emitter.project_past_directed_photon(state)
                self.assertEqual(projection.face_classification, "outgoing")
                self.assertLess(
                    _relative_error(projection.local_frequency, scale),
                    2.0e-13,
                )
                if reference_mu is None:
                    reference_mu = projection.outgoing_cosine
                self.assertAlmostEqual(
                    projection.outgoing_cosine,
                    reference_mu,
                    delta=2.0e-12,
                )

    def test_backside_mu_is_never_hidden_by_absolute_value(self) -> None:
        emitter = self.emitter()
        state = self.photon_at_mu(emitter, -0.42)
        with self.assertRaisesRegex(BacksidePhotonError, r"signed mu=-0\.4"):
            emitter.project_past_directed_photon(state)
        projection = emitter.project_past_directed_photon(
            state,
            backside_policy="classify",
        )
        self.assertEqual(projection.face_classification, "backside")
        self.assertAlmostEqual(projection.outgoing_cosine, -0.42, delta=2.0e-12)

    def test_photon_event_null_and_time_direction_gates_fail_closed(self) -> None:
        emitter = self.emitter()
        valid = self.photon_at_mu(emitter, 0.5)
        displaced_event = tuple(
            valid.event[index] + (1.0e-5 if index == 1 else 0.0)
            for index in range(4)
        )
        with self.assertRaisesRegex(ValueError, "authenticated face event"):
            emitter.project_past_directed_photon(
                HamiltonianState(displaced_event, valid.covector),  # type: ignore[arg-type]
            )

        sample = emitter.metric.sample(emitter.event)
        timelike_past_covector = matrix_vector(
            sample.covariant,
            tuple(-value for value in emitter.four_velocity),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "null residual"):
            emitter.project_past_directed_photon(
                HamiltonianState(emitter.event, timelike_past_covector),
            )

        future_covector = tuple(-value for value in valid.covector)
        with self.assertRaisesRegex(ValueError, "frequency must be positive"):
            emitter.project_past_directed_photon(
                HamiltonianState(emitter.event, future_covector),  # type: ignore[arg-type]
            )

    def test_metric_calibration_face_isco_and_scale_violations_fail_closed(self) -> None:
        calibration = self.calibration(spin=0.7)
        with self.assertRaisesRegex(ValueError, "calibration spin disagree"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=0.6),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=10.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "calibration spin disagree"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=1.0),
                calibration=self.calibration(spin=0.998),
                pseudo_cylindrical_radius_over_mass=10.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "strictly outside the ISCO"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=0.7),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=calibration.isco_radius_over_mass,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "strictly outside the ISCO"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=0.7),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=0.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "face must"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=0.7),
                calibration=calibration,
                pseudo_cylindrical_radius_over_mass=10.0,
                face="inside",  # type: ignore[arg-type]
            )
        zero_thickness = self.calibration(spin=0.7, dotm=0.0)
        with self.assertRaisesRegex(ValueError, "dotm=0"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(spin_a_m=0.7),
                calibration=zero_thickness,
                pseudo_cylindrical_radius_over_mass=10.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "coordinates overflowed"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(
                    mass_m=1.0e308,
                    spin_a_m=0.0,
                    singularity_guard_m=1.0,
                ),
                calibration=self.calibration(spin=0.0),
                pseudo_cylindrical_radius_over_mass=10.0,
                face=UPPER,
            )
        with self.assertRaisesRegex(ValueError, "scale/axis guard"):
            KerrFiniteThicknessFaceEmitter(
                metric=KerrKerrSchildMetric(
                    mass_m=1.0e-300,
                    spin_a_m=0.0,
                    singularity_guard_m=1.0e-9,
                ),
                calibration=self.calibration(spin=0.0),
                pseudo_cylindrical_radius_over_mass=10.0,
                face=UPPER,
            )


if __name__ == "__main__":
    unittest.main()
