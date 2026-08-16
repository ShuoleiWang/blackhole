from __future__ import annotations

from dataclasses import replace
import math
import unittest

from offline.kerr import KerrKerrSchildMetric
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_launch import (
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessLaunchError,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.spacetime import bilinear


class OfflineKerrFiniteThicknessLaunchTests(unittest.TestCase):
    def emitter(
        self,
        *,
        signed_spin: float = 0.7,
        face: str = UPPER,
        rho: float = 8.0,
        phi: float = 0.4,
    ) -> KerrFiniteThicknessFaceEmitter:
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=abs(signed_spin),
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=30.0,
        )
        return KerrFiniteThicknessFaceEmitter(
            metric=KerrKerrSchildMetric(spin_a_m=signed_spin),
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,
            phi_ks_rad=phi,
        )

    def test_scientific_boundary_is_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertFalse(SCIENTIFIC_STATUS["includesGeodesicTracing"])
        self.assertFalse(SCIENTIFIC_STATUS["includesReturningRadiationKernel"])
        self.assertFalse(SCIENTIFIC_STATUS["includesSolvedAtmosphere"])
        self.assertFalse(
            SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"]
        )

    def test_surface_frame_is_orthonormal_across_faces_and_spins(self) -> None:
        for signed_spin in (-0.998, 0.0, 0.998):
            for face in (UPPER, LOWER):
                with self.subTest(signed_spin=signed_spin, face=face):
                    emitter = self.emitter(signed_spin=signed_spin, face=face)
                    frame = KerrFiniteThicknessSurfaceFrame(emitter)
                    metric = emitter.metric.sample(emitter.event).covariant
                    basis = (
                        emitter.four_velocity,
                        frame.meridional_tangent,
                        frame.azimuthal_tangent,
                        emitter.outward_unit_normal,
                    )
                    for first in range(4):
                        for second in range(4):
                            expected = (
                                -1.0
                                if first == second == 0
                                else float(first == second)
                            )
                            self.assertAlmostEqual(
                                bilinear(basis[first], metric, basis[second]),
                                expected,
                                delta=2.0e-9,
                            )
                    self.assertLess(frame.maximum_gram_error, 2.0e-9)

    def test_future_launch_and_reversed_past_projection_agree(self) -> None:
        emitter = self.emitter(signed_spin=0.998, face=UPPER, rho=4.0)
        frame = KerrFiniteThicknessSurfaceFrame(emitter)
        for mu in (1.0e-4, 0.2, 0.73, 1.0):
            for azimuth in (0.0, 0.7, 2.9, 2.0 * math.pi + 0.7):
                with self.subTest(mu=mu, azimuth=azimuth):
                    launch = KerrFiniteThicknessEmissionLaunch(
                        frame,
                        mu,
                        azimuth,
                        local_frequency=3.5,
                    )
                    projection = emitter.project_past_directed_photon(
                        launch.reversed_past_state,
                    )
                    self.assertLess(launch.null_residual, 2.0e-10)
                    self.assertAlmostEqual(
                        projection.local_frequency,
                        3.5,
                        delta=2.0e-8,
                    )
                    self.assertAlmostEqual(
                        projection.outgoing_cosine,
                        mu,
                        delta=2.0e-9,
                    )

    def test_azimuth_is_canonical_and_periodic(self) -> None:
        frame = KerrFiniteThicknessSurfaceFrame(self.emitter())
        first = KerrFiniteThicknessEmissionLaunch(frame, 0.4, 0.75)
        wrapped = KerrFiniteThicknessEmissionLaunch(
            frame,
            0.4,
            0.75 + 8.0 * math.pi,
        )
        self.assertEqual(first.tangent_azimuth_rad, wrapped.tangent_azimuth_rad)
        self.assertEqual(first.future_state, wrapped.future_state)
        self.assertEqual(
            first.model_descriptor_sha256,
            wrapped.model_descriptor_sha256,
        )

    def test_local_frequency_is_positive_affine_scale(self) -> None:
        frame = KerrFiniteThicknessSurfaceFrame(self.emitter())
        reference = KerrFiniteThicknessEmissionLaunch(frame, 0.61, 1.2, 1.0)
        for scale in (1.0e-150, 1.0e-20, 1.0, 1.0e20, 1.0e150):
            with self.subTest(scale=scale):
                launch = KerrFiniteThicknessEmissionLaunch(
                    frame,
                    0.61,
                    1.2,
                    scale,
                )
                for actual, unit in zip(
                    launch.future_state.covector,
                    reference.future_state.covector,
                ):
                    self.assertAlmostEqual(actual / scale, unit, delta=2.0e-12)

    def test_upper_lower_reflection_preserves_local_launch_invariants(self) -> None:
        launches = []
        for face in (UPPER, LOWER):
            frame = KerrFiniteThicknessSurfaceFrame(
                self.emitter(face=face, phi=0.0)
            )
            launches.append(
                KerrFiniteThicknessEmissionLaunch(frame, 0.37, 0.0)
            )
        upper, lower = launches
        self.assertAlmostEqual(upper.null_residual, lower.null_residual, delta=2e-13)
        self.assertAlmostEqual(
            upper.frame.maximum_gram_error,
            lower.frame.maximum_gram_error,
            delta=2e-13,
        )
        self.assertAlmostEqual(
            upper.future_state.event[3],
            -lower.future_state.event[3],
            delta=2e-13,
        )

    def test_near_calibrated_extreme_isco_and_outer_face_remain_finite(self) -> None:
        for signed_spin in (-0.998, 0.998):
            calibration = StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.998,
                eddington_scaled_mass_accretion_rate=0.3,
                outer_radius_over_mass=40.0,
            )
            for rho in (
                calibration.isco_radius_over_mass * (1.0 + 1.0e-8),
                39.9,
            ):
                for face in (UPPER, LOWER):
                    with self.subTest(
                        signed_spin=signed_spin,
                        rho=rho,
                        face=face,
                    ):
                        emitter = KerrFiniteThicknessFaceEmitter(
                            metric=KerrKerrSchildMetric(spin_a_m=signed_spin),
                            calibration=calibration,
                            pseudo_cylindrical_radius_over_mass=rho,
                            face=face,
                            phi_ks_rad=1.2,
                        )
                        frame = KerrFiniteThicknessSurfaceFrame(emitter)
                        launch = KerrFiniteThicknessEmissionLaunch(
                            frame,
                            1.0e-3,
                            5.9,
                        )
                        self.assertLess(frame.maximum_gram_error, 2.0e-9)
                        self.assertLess(launch.null_residual, 2.0e-10)

    def test_fail_closed_inputs_and_identity(self) -> None:
        emitter = self.emitter()
        frame = KerrFiniteThicknessSurfaceFrame(emitter)
        for mu in (-0.1, 0.0, 1.1, math.nan):
            with self.subTest(mu=mu):
                with self.assertRaises(ValueError):
                    KerrFiniteThicknessEmissionLaunch(frame, mu, 0.0)
        for frequency in (0.0, -1.0, math.inf):
            with self.subTest(frequency=frequency):
                with self.assertRaises(ValueError):
                    KerrFiniteThicknessEmissionLaunch(
                        frame,
                        0.5,
                        0.0,
                        frequency,
                    )
        with self.assertRaises(ValueError):
            KerrFiniteThicknessEmissionLaunch(frame, 0.5, math.nan)
        with self.assertRaises(TypeError):
            KerrFiniteThicknessSurfaceFrame(object())  # type: ignore[arg-type]
        launch = KerrFiniteThicknessEmissionLaunch(frame, 0.5, 0.2)
        self.assertEqual(
            launch.model_descriptor()["frameDescriptorSha256"],
            frame.model_descriptor_sha256,
        )
        with self.assertRaises(TypeError):
            replace(launch, emission_angle_cosine=0.7)

    def test_low_level_frame_and_emitter_tampering_cannot_reuse_identity(self) -> None:
        emitter = self.emitter()
        frame = KerrFiniteThicknessSurfaceFrame(emitter)
        original_hash = frame.model_descriptor_sha256
        object.__setattr__(
            frame,
            "meridional_tangent",
            tuple(-value for value in frame.meridional_tangent),
        )
        self.assertEqual(frame.model_descriptor_sha256, original_hash)
        with self.assertRaises(KerrFiniteThicknessLaunchError):
            KerrFiniteThicknessEmissionLaunch(frame, 0.5, 0.0)

        emitter = self.emitter()
        object.__setattr__(
            emitter,
            "four_velocity",
            tuple(2.0 * value for value in emitter.four_velocity),
        )
        with self.assertRaises(KerrFiniteThicknessLaunchError):
            KerrFiniteThicknessSurfaceFrame(emitter)


if __name__ == "__main__":
    unittest.main()
