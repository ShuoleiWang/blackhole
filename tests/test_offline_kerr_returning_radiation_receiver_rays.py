from __future__ import annotations

from dataclasses import replace
import math
import unittest

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    UPPER_SURFACE_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_returning_radiation_rays import (
    trace_kerr_returning_radiation_direction,
)
from offline.kerr_returning_radiation_receiver_rays import (
    IMPLEMENTATION_ID,
    PAST_WORLDTUBE_NO_SOURCE,
    SCIENTIFIC_STATUS,
    KerrReturningRadiationReceiverRayError,
    trace_kerr_returning_radiation_receiver_direction,
    verify_kerr_returning_radiation_receiver_direction,
)


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


class PolicyBypassFloat(float):
    def __gt__(self, other):
        return False


class OfflineKerrReturningRadiationReceiverRayTests(unittest.TestCase):
    @staticmethod
    def forge_result(original, **changes):
        forged = object.__new__(type(original))
        for name in type(original).__dataclass_fields__:
            object.__setattr__(
                forged,
                name,
                changes.get(name, getattr(original, name)),
            )
        return forged

    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=30.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=200.0,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )

        source = KerrFiniteThicknessFaceEmitter(
            metric=cls.metric,
            calibration=cls.calibration,
            pseudo_cylindrical_radius_over_mass=8.0,
            face=UPPER,
        )
        launch = KerrFiniteThicknessEmissionLaunch(
            KerrFiniteThicknessSurfaceFrame(source),
            0.02,
            0.5 * math.pi,
        )
        cls.forward = trace_kerr_returning_radiation_direction(
            launch,
            cls.surface,
            termination=cls.termination,
            ray_options=cls.ray_options,
            surface_options=cls.surface_options,
        )
        assert cls.forward.receiver is not None
        cls.reciprocal_frame = KerrFiniteThicknessSurfaceFrame(
            cls.forward.receiver
        )
        covector = cls.forward.ray.terminal_state.covector
        receiver = cls.reciprocal_frame.emitter
        frequency = -math.fsum(
            receiver.four_velocity[index] * covector[index]
            for index in range(4)
        )
        cls.reciprocal_mu = -math.fsum(
            receiver.outward_unit_normal[index] * covector[index]
            for index in range(4)
        ) / frequency
        meridional = math.fsum(
            cls.reciprocal_frame.meridional_tangent[index] * covector[index]
            for index in range(4)
        ) / frequency
        azimuthal = math.fsum(
            cls.reciprocal_frame.azimuthal_tangent[index] * covector[index]
            for index in range(4)
        ) / frequency
        cls.reciprocal_psi = math.atan2(azimuthal, meridional) % (
            2.0 * math.pi
        )
        cls.reciprocal = cls.trace(
            cls.reciprocal_frame,
            cls.reciprocal_mu,
            cls.reciprocal_psi,
        )

    @classmethod
    def frame(cls, rho: float, face: str = UPPER):
        return KerrFiniteThicknessSurfaceFrame(
            KerrFiniteThicknessFaceEmitter(
                metric=cls.metric,
                calibration=cls.calibration,
                pseudo_cylindrical_radius_over_mass=rho,
                face=face,
            )
        )

    @classmethod
    def trace(cls, frame, mu: float, psi: float, **kwargs):
        return trace_kerr_returning_radiation_receiver_direction(
            frame,
            cls.surface,
            float(mu),
            float(psi),
            termination=cls.termination,
            ray_options=cls.ray_options,
            surface_options=cls.surface_options,
            **kwargs,
        )

    def test_scientific_boundary_is_directional_not_kernel_or_flux(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(
            SCIENTIFIC_STATUS["isReceiverCentredBackwardSkyPrimitive"]
        )
        for key in (
            "includesReceiverSolidAngleIntegration",
            "includesAreaJacobian",
            "outputsReturningRadiationKernelK",
            "isReceiverCentredIncidentFlux",
            "includesReturningRadiationStressWorkFS",
            "isCompleteKerrbb",
            "includesSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("mu_i*g^4", SCIENTIFIC_STATUS["directionalIntegrand"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["outputsReturningRadiationKernelK"] = True

    def test_same_face_source_reciprocity_g_inverse_and_d20_integrand(self) -> None:
        result = self.reciprocal
        self.assertEqual(result.outcome, "source-upper")
        self.assertEqual(result.source_face, UPPER)
        self.assertEqual(result.source_surface_id, UPPER_SURFACE_ID)
        self.assertAlmostEqual(
            result.source_radius_over_mass,
            8.0,
            delta=2.0e-8,
        )
        self.assertAlmostEqual(
            result.source_emission_cosine,
            0.02,
            delta=2.0e-8,
        )
        self.assertTrue(
            math.isclose(
                result.source_to_receiver_frequency_ratio,
                self.forward.emitter_to_receiver_frequency_ratio,
                rel_tol=5.0e-9,
            )
        )
        self.assertTrue(
            math.isclose(
                result.source_local_frequency,
                1.0 / self.forward.emitter_to_receiver_frequency_ratio,
                rel_tol=5.0e-9,
            )
        )
        expected_d20 = 0.5 + 0.75 * result.source_emission_cosine
        expected = (
            result.receiver_incidence_cosine
            * result.source_to_receiver_frequency_ratio**4
            * expected_d20
        )
        self.assertTrue(
            math.isclose(
                result.d20_angular_multiplier,
                expected_d20,
                rel_tol=2.0e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                result.receiver_directional_integrand,
                expected,
                rel_tol=2.0e-15,
            )
        )

    def test_real_cross_face_source_is_front_face_outgoing(self) -> None:
        result = self.trace(
            self.frame(4.0, UPPER),
            0.02,
            math.pi / 6.0,
        )
        self.assertEqual(result.outcome, "source-lower")
        self.assertEqual(result.source_face, LOWER)
        self.assertEqual(result.source_surface_id, LOWER_SURFACE_ID)
        self.assertAlmostEqual(
            result.source_radius_over_mass,
            4.945399033522557,
            delta=3.0e-5,
        )
        self.assertGreater(result.source_emission_cosine, 0.0)
        self.assertGreater(result.receiver_directional_integrand, 0.0)
        projection = result.source.project_past_directed_photon(
            result.ray.terminal_state,
            null_residual_limit=1.0e-6,
            event_tolerance_m=2.0e-8 * self.metric.mass_m,
            backside_policy="classify",
        )
        self.assertEqual(projection.face_classification, "outgoing")

    def test_receiver_is_incoming_backside_while_source_is_outgoing_front(self) -> None:
        result = self.reciprocal
        receiver_projection = result.receiver_frame.emitter.project_past_directed_photon(
            result.receiver_past_state,
            null_residual_limit=2.0e-10,
            backside_policy="classify",
        )
        source_projection = result.source.project_past_directed_photon(
            result.ray.terminal_state,
            null_residual_limit=1.0e-6,
            event_tolerance_m=2.0e-8 * self.metric.mass_m,
            backside_policy="classify",
        )
        self.assertEqual(receiver_projection.face_classification, "backside")
        self.assertLess(receiver_projection.outgoing_cosine, 0.0)
        self.assertEqual(source_projection.face_classification, "outgoing")
        self.assertGreater(source_projection.outgoing_cosine, 0.0)

    def test_no_disk_source_uses_past_worldtube_semantics_only(self) -> None:
        result = self.trace(self.frame(8.0), 1.0, 0.0)
        self.assertEqual(result.outcome, PAST_WORLDTUBE_NO_SOURCE)
        self.assertNotIn(result.outcome, ("captured", "escaped"))
        self.assertIsNone(result.source)
        self.assertIsNone(result.source_face)
        self.assertIsNotNone(result.past_worldtube_target_id)
        self.assertIsNotNone(result.past_worldtube_radius_m)
        self.assertEqual(result.receiver_directional_integrand, 0.0)
        descriptor = result.model_descriptor()
        self.assertEqual(
            descriptor["pastWorldtube"]["semanticOutcome"],
            PAST_WORLDTUBE_NO_SOURCE,
        )

    def test_initial_contact_and_each_n_2n_topology_are_bound(self) -> None:
        result = self.reciprocal
        for ray, options in (
            (result.ray, result.surface_options),
            (result.coarse_ray, result.coarse_surface_options),
        ):
            trace = ray.multi_surface_trace
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertTrue(trace.topology_converged)
            self.assertEqual(
                trace.base_subdivisions_per_step,
                options.subdivisions_per_segment,
            )
            self.assertEqual(
                trace.verification_subdivisions_per_step,
                2 * options.subdivisions_per_segment,
            )
            self.assertIsNotNone(trace.initial_contact)
            self.assertEqual(trace.initial_contact.surface_id, UPPER_SURFACE_ID)
            self.assertEqual(trace.initial_contact.side, 1)
            self.assertGreater(
                trace.crossings[-1].crossing.ray_affine_length,
                8.0 * options.affine_tolerance,
            )
        descriptor = result.model_descriptor()
        self.assertIs(
            descriptor["initialContact"]["epsilonEventDisplacement"],
            False,
        )
        self.assertTrue(result.convergence.complete_topology_agrees)

    def test_audited_grazing_fine_coarse_difference_fails_closed(self) -> None:
        coarse_ray = RayTraceOptions(
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
            initial_step=0.2,
            minimum_step=1.0e-8,
            maximum_step=2.0,
            maximum_affine_length=200.0,
            null_residual_limit=1.0e-6,
            metric_interpolation_error_limit=1.0e-6,
            event_value_tolerance=6.0e-8,
            event_affine_tolerance=6.0e-9,
        )
        coarse_surface = SurfaceEventOptions(
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
            null_residual_limit=1.0e-6,
            metric_interpolation_error_limit=1.0e-6,
            surface_value_tolerance=6.0e-8,
            affine_tolerance=6.0e-9,
            subdivisions_per_segment=2,
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationReceiverRayError,
            "fine/coarse receiver-ray terminal state disagrees",
        ):
            self.trace(
                self.frame(4.0),
                1.0e-4,
                math.pi / 24.0,
                coarse_ray_options=coarse_ray,
                coarse_surface_options=coarse_surface,
            )

    def test_public_replay_rejects_result_and_descriptor_tampering(self) -> None:
        result = self.reciprocal
        verify_kerr_returning_radiation_receiver_direction(result)
        result.revalidate()
        for field, value in (
            ("outcome", AlwaysEqualStr(PAST_WORLDTUBE_NO_SOURCE)),
            ("source_emission_cosine", AlwaysEqualFloat(0.9)),
            ("source_to_receiver_frequency_ratio", AlwaysEqualFloat(99.0)),
            ("receiver_directional_integrand", AlwaysEqualFloat(99.0)),
            ("source_face", AlwaysEqualStr(LOWER)),
        ):
            with self.subTest(field=field):
                forged = self.forge_result(result, **{field: value})
                with self.assertRaisesRegex(
                    KerrReturningRadiationReceiverRayError,
                    "live fields disagree",
                ):
                    verify_kerr_returning_radiation_receiver_direction(forged)
        stale_descriptor = self.forge_result(
            result,
            _descriptor_json=result._descriptor_json + " ",
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationReceiverRayError,
            "descriptor or SHA-256",
        ):
            verify_kerr_returning_radiation_receiver_direction(stale_descriptor)

    def test_exact_input_types_and_policy_bypass_subclasses_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact float"):
            trace_kerr_returning_radiation_receiver_direction(
                self.reciprocal_frame,
                self.surface,
                1,
                self.reciprocal_psi,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
            )
        forged_ray_options = replace(
            self.ray_options,
            absolute_tolerance=PolicyBypassFloat(1.0),
        )
        with self.assertRaises(KerrReturningRadiationReceiverRayError):
            trace_kerr_returning_radiation_receiver_direction(
                self.reciprocal_frame,
                self.surface,
                float(self.reciprocal_mu),
                float(self.reciprocal_psi),
                termination=self.termination,
                ray_options=forged_ray_options,
                surface_options=self.surface_options,
            )

    def test_option_schema_rejects_bool_int_and_float_subclass_before_policy(
        self,
    ) -> None:
        bad_ray_options = (
            replace(self.ray_options, maximum_step=True),
            replace(self.ray_options, maximum_step=1),
            replace(
                self.ray_options,
                maximum_step=AlwaysEqualFloat(1.0),
            ),
        )
        for options in bad_ray_options:
            with self.subTest(ray_type=type(options.maximum_step).__name__):
                with self.assertRaisesRegex(
                    KerrReturningRadiationReceiverRayError,
                    "ray_options.maximum_step has non-exact field type",
                ):
                    trace_kerr_returning_radiation_receiver_direction(
                        self.reciprocal_frame,
                        self.surface,
                        float(self.reciprocal_mu),
                        float(self.reciprocal_psi),
                        termination=self.termination,
                        ray_options=options,
                        surface_options=self.surface_options,
                    )

        bad_surface_options = (
            replace(self.surface_options, surface_value_tolerance=True),
            replace(self.surface_options, surface_value_tolerance=1),
            replace(
                self.surface_options,
                surface_value_tolerance=PolicyBypassFloat(1.0e-9),
            ),
        )
        for options in bad_surface_options:
            with self.subTest(
                surface_type=type(options.surface_value_tolerance).__name__
            ):
                with self.assertRaisesRegex(
                    KerrReturningRadiationReceiverRayError,
                    "surface_options.surface_value_tolerance has non-exact field type",
                ):
                    trace_kerr_returning_radiation_receiver_direction(
                        self.reciprocal_frame,
                        self.surface,
                        float(self.reciprocal_mu),
                        float(self.reciprocal_psi),
                        termination=self.termination,
                        ray_options=self.ray_options,
                        surface_options=options,
                    )


if __name__ == "__main__":
    unittest.main()
