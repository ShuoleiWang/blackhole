from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import pickle
import threading
import unittest
from unittest.mock import patch

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
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    KerrReturningRadiationRayError,
    _consume_issued_kerr_returning_radiation_direction,
    _trace_issued_kerr_returning_radiation_direction,
    trace_kerr_returning_radiation_direction,
    verify_kerr_returning_radiation_direction,
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


class OfflineKerrReturningRadiationRayTests(unittest.TestCase):
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

    @classmethod
    def launch(
        cls,
        *,
        rho: float,
        face: str,
        mu: float,
        azimuth: float,
        local_frequency: float = 1.0,
        metric: KerrKerrSchildMetric | None = None,
        calibration: StationaryKerrFiniteThicknessCalibration | None = None,
    ) -> KerrFiniteThicknessEmissionLaunch:
        selected_metric = cls.metric if metric is None else metric
        selected_calibration = (
            cls.calibration if calibration is None else calibration
        )
        emitter = KerrFiniteThicknessFaceEmitter(
            metric=selected_metric,
            calibration=selected_calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,
        )
        return KerrFiniteThicknessEmissionLaunch(
            KerrFiniteThicknessSurfaceFrame(emitter),
            mu,
            azimuth,
            local_frequency,
        )

    @classmethod
    def trace(cls, launch: KerrFiniteThicknessEmissionLaunch):
        return trace_kerr_returning_radiation_direction(
            launch,
            cls.surface,
            termination=cls.termination,
            ray_options=cls.ray_options,
            surface_options=cls.surface_options,
        )

    def test_scientific_boundary_and_coefficient_semantics_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(SCIENTIFIC_STATUS["isIndependentRayTransportPrimitive"])
        self.assertFalse(SCIENTIFIC_STATUS["isIndependentRayKernel"])
        self.assertTrue(
            SCIENTIFIC_STATUS["requiresPublicRevalidationBeforeConsumption"]
        )
        for key in (
            "isCompleteReturningRadiationKernel",
            "isCompleteKerrbb",
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("finite", SCIENTIFIC_STATUS["surfaceCompleteness"])
        self.assertIn("complete KERRBB", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isCompleteKerrbb"] = True

    def test_real_kerr_return_has_receiver_g4_and_excludes_initial_contact(self) -> None:
        result = self.trace(
            self.launch(
                rho=8.0,
                face=UPPER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            )
        )
        self.assertEqual(result.fate, "return-upper")
        self.assertEqual(result.receiver_face, UPPER)
        self.assertEqual(result.receiver_surface_id, UPPER_SURFACE_ID)
        self.assertIsNotNone(result.emitter_to_receiver_frequency_ratio)
        self.assertIsNotNone(result.receiver_incidence_cosine)
        self.assertGreater(result.receiver_incidence_cosine, 0.0)
        self.assertLessEqual(result.receiver_incidence_cosine, 1.0)
        expected_g4 = result.emitter_to_receiver_frequency_ratio**4
        self.assertTrue(
            math.isclose(result.bolometric_g4_factor, expected_g4, rel_tol=2e-15)
        )
        self.assertTrue(
            math.isclose(
                result.receiver_incidence_weighted_g4,
                result.receiver_incidence_cosine * expected_g4,
                rel_tol=2e-15,
            )
        )
        terminal = result.ray.multi_surface_trace.crossings[-1]
        self.assertGreater(
            terminal.crossing.ray_affine_length,
            8.0 * self.surface_options.affine_tolerance,
        )
        descriptor = result.model_descriptor()
        self.assertIs(
            descriptor["initialContact"]["epsilonEventDisplacement"],
            False,
        )
        self.assertEqual(
            descriptor["initialContact"]["exactStateTopologySide"],
            "outward-positive",
        )
        self.assertTrue(descriptor["wholeRayConvergence"]["actual"]["converged"])
        self.assertTrue(result.convergence.complete_topology_agrees)
        self.assertIsNotNone(
            result.convergence.frequency_ratio_relative_difference
        )
        self.assertIsNotNone(result.convergence.signed_receiver_cosine_difference)
        self.assertIsNotNone(result.convergence.g4_relative_difference)
        self.assertNotEqual(result.ray_options, result.coarse_ray_options)
        self.assertNotEqual(result.surface_options, result.coarse_surface_options)

    def test_audited_fine_return_coarse_escape_counterexample_fails_closed(self) -> None:
        launch = self.launch(
            rho=8.0,
            face=UPPER,
            mu=0.02,
            azimuth=0.9973983529591495,
        )
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
            KerrReturningRadiationRayError,
            "fine=return-upper.*coarse=escaped",
        ):
            trace_kerr_returning_radiation_direction(
                launch,
                self.surface,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                coarse_ray_options=coarse_ray,
                coarse_surface_options=coarse_surface,
            )

    def test_public_revalidator_replays_both_rays_and_rejects_live_tampering(self) -> None:
        result = self.trace(
            self.launch(
                rho=8.0,
                face=UPPER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            )
        )
        verify_kerr_returning_radiation_direction(result)
        result.revalidate()
        tampered_values = {
            "fate": AlwaysEqualStr("escaped"),
            "emitter_to_receiver_frequency_ratio": AlwaysEqualFloat(99.0),
            "bolometric_g4_factor": AlwaysEqualFloat(99.0),
        }
        for field, value in tampered_values.items():
            with self.subTest(field=field):
                forged = self.forge_result(result, **{field: value})
                self.assertEqual(
                    forged.model_descriptor_sha256,
                    result.model_descriptor_sha256,
                )
                with self.assertRaisesRegex(
                    KerrReturningRadiationRayError,
                    "live fields disagree",
                ):
                    verify_kerr_returning_radiation_direction(forged)
                if field == "fate":
                    with self.assertRaisesRegex(
                        KerrReturningRadiationRayError,
                        "non-exact type",
                    ):
                        forged.photon_fate_indicators()

        nested_covector = list(result.ray.terminal_state.covector)
        nested_covector[0] = AlwaysEqualFloat(nested_covector[0])
        nested_state = replace(
            result.ray.terminal_state,
            covector=tuple(nested_covector),
        )
        nested_ray = replace(result.ray, terminal_state=nested_state)
        nested_forged = self.forge_result(result, ray=nested_ray)
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "non-exact field type",
        ):
            verify_kerr_returning_radiation_direction(nested_forged)

        descriptor_forged = self.forge_result(
            result,
            _descriptor_json=AlwaysEqualStr(result._descriptor_json),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "descriptor is malformed",
        ):
            verify_kerr_returning_radiation_direction(descriptor_forged)

    def test_private_issued_capability_is_exact_fresh_and_single_use(self) -> None:
        primitive, token = _trace_issued_kerr_returning_radiation_direction(
            self.launch(
                rho=8.0,
                face=UPPER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            ),
            self.surface,
            termination=self.termination,
            ray_options=self.ray_options,
            surface_options=self.surface_options,
        )
        with self.assertRaises(TypeError):
            type(token)()
        with self.assertRaises(TypeError):
            type("ForgedToken", (type(token),), {})
        with self.assertRaises(TypeError):
            copy.copy(token)
        with self.assertRaises(TypeError):
            pickle.dumps(token)

        payload = _consume_issued_kerr_returning_radiation_direction(
            primitive,
            token,
        )
        self.assertEqual(payload.fate, primitive.fate)
        self.assertEqual(
            payload.primitive_descriptor_sha256,
            primitive.model_descriptor_sha256,
        )
        self.assertEqual(payload.receiver_face, primitive.receiver_face)
        self.assertEqual(
            payload.receiver_radius_over_mass.hex(),
            primitive.receiver_radius_over_mass.hex(),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "forged, stale, or already consumed",
        ):
            _consume_issued_kerr_returning_radiation_direction(primitive, token)
        with self.assertRaises(TypeError):
            _consume_issued_kerr_returning_radiation_direction(
                primitive,
                object(),
            )

    def test_private_issued_capability_burns_on_wrong_result_mutation_and_pid(
        self,
    ) -> None:
        arguments = (
            self.launch(
                rho=8.0,
                face=UPPER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            ),
            self.surface,
        )
        keywords = {
            "termination": self.termination,
            "ray_options": self.ray_options,
            "surface_options": self.surface_options,
        }

        primitive, token = _trace_issued_kerr_returning_radiation_direction(
            *arguments,
            **keywords,
        )
        wrong = copy.deepcopy(primitive)
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "does not own this exact result",
        ):
            _consume_issued_kerr_returning_radiation_direction(wrong, token)
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "forged, stale, or already consumed",
        ):
            _consume_issued_kerr_returning_radiation_direction(primitive, token)

        primitive, token = _trace_issued_kerr_returning_radiation_direction(
            *arguments,
            **keywords,
        )
        object.__setattr__(primitive, "fate", "escaped")
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "changed after canonical tracing",
        ):
            _consume_issued_kerr_returning_radiation_direction(primitive, token)

        primitive, token = _trace_issued_kerr_returning_radiation_direction(
            *arguments,
            **keywords,
        )
        with patch(
            "offline.kerr_returning_radiation_rays.os.getpid",
            return_value=-1,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationRayError,
                "different process",
            ):
                _consume_issued_kerr_returning_radiation_direction(
                    primitive,
                    token,
                )

    def test_private_issued_capability_has_one_concurrent_winner(self) -> None:
        primitive, token = _trace_issued_kerr_returning_radiation_direction(
            self.launch(
                rho=8.0,
                face=UPPER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            ),
            self.surface,
            termination=self.termination,
            ray_options=self.ray_options,
            surface_options=self.surface_options,
        )
        barrier = threading.Barrier(8)

        def consume() -> str:
            barrier.wait()
            try:
                _consume_issued_kerr_returning_radiation_direction(
                    primitive,
                    token,
                )
                return "consumed"
            except KerrReturningRadiationRayError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = tuple(executor.map(lambda _index: consume(), range(8)))
        self.assertEqual(outcomes.count("consumed"), 1)
        self.assertEqual(outcomes.count("rejected"), 7)

    def test_upper_lower_real_kerr_reflection_returns_to_matching_faces(self) -> None:
        results = []
        for face in (UPPER, LOWER):
            results.append(
                self.trace(
                    self.launch(
                        rho=8.0,
                        face=face,
                        mu=0.02,
                        azimuth=0.5 * math.pi,
                    )
                )
            )
        upper, lower = results
        self.assertEqual(upper.fate, "return-upper")
        self.assertEqual(lower.fate, "return-lower")
        self.assertEqual(lower.receiver_surface_id, LOWER_SURFACE_ID)
        for upper_value, lower_value in (
            (
                upper.receiver_radius_over_mass,
                lower.receiver_radius_over_mass,
            ),
            (
                upper.emitter_to_receiver_frequency_ratio,
                lower.emitter_to_receiver_frequency_ratio,
            ),
            (
                upper.receiver_incidence_cosine,
                lower.receiver_incidence_cosine,
            ),
            (upper.ray.affine_length, lower.ray.affine_length),
        ):
            self.assertTrue(
                math.isclose(upper_value, lower_value, rel_tol=3.0e-13)
            )

    def test_schwarzschild_low_spin_reflection_is_an_exact_calibration(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.0)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=30.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        results = []
        for face in (UPPER, LOWER):
            launch = self.launch(
                rho=8.0,
                face=face,
                mu=0.02,
                azimuth=0.5 * math.pi,
                metric=metric,
                calibration=calibration,
            )
            results.append(
                trace_kerr_returning_radiation_direction(
                    launch,
                    surface,
                    termination=termination,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                )
            )
        upper, lower = results
        self.assertEqual((upper.fate, lower.fate), ("return-upper", "return-lower"))
        self.assertEqual(
            upper.receiver_radius_over_mass,
            lower.receiver_radius_over_mass,
        )
        self.assertEqual(
            upper.emitter_to_receiver_frequency_ratio,
            lower.emitter_to_receiver_frequency_ratio,
        )
        self.assertTrue(
            math.isclose(
                upper.receiver_incidence_cosine,
                lower.receiver_incidence_cosine,
                rel_tol=8.0e-15,
            )
        )

    def test_real_kerr_capture_escape_and_plunge_are_distinct_zero_contributors(self) -> None:
        cases = (
            (
                "captured",
                self.launch(rho=3.5, face=UPPER, mu=0.5, azimuth=3.4),
            ),
            (
                "escaped",
                self.launch(rho=8.0, face=UPPER, mu=1.0, azimuth=0.0),
            ),
            (
                "plunge-sink",
                self.launch(rho=8.0, face=UPPER, mu=0.02, azimuth=math.pi),
            ),
        )
        for expected_fate, launch in cases:
            with self.subTest(fate=expected_fate):
                result = self.trace(launch)
                self.assertEqual(result.fate, expected_fate)
                self.assertIsNone(result.receiver)
                self.assertIsNone(result.emitter_to_receiver_frequency_ratio)
                self.assertIsNone(result.bolometric_g4_factor)
                self.assertEqual(result.receiver_incidence_weighted_g4, 0.0)
                if expected_fate in ("captured", "escaped"):
                    self.assertIsNotNone(
                        result.convergence.worldtube_radius_difference_m
                    )
                else:
                    self.assertIsNone(
                        result.convergence.worldtube_radius_difference_m
                    )

    def test_every_successful_direction_has_one_and_only_one_fate(self) -> None:
        launches = (
            self.launch(
                rho=8.0,
                face=LOWER,
                mu=0.02,
                azimuth=0.5 * math.pi,
            ),
            self.launch(rho=3.5, face=LOWER, mu=0.5, azimuth=3.4),
            self.launch(rho=8.0, face=LOWER, mu=1.0, azimuth=0.0),
            self.launch(rho=8.0, face=LOWER, mu=0.02, azimuth=math.pi),
        )
        for launch in launches:
            result = self.trace(launch)
            indicators = result.photon_fate_indicators()
            self.assertEqual(sum(indicators.values()), 1)
            self.assertEqual(indicators[result.fate], 1)
            with self.assertRaises(TypeError):
                indicators["escaped"] = 1

    def test_positive_affine_launch_scale_does_not_change_physics(self) -> None:
        results = []
        for local_frequency in (0.125, 9.0):
            results.append(
                self.trace(
                    self.launch(
                        rho=8.0,
                        face=UPPER,
                        mu=0.02,
                        azimuth=0.5 * math.pi,
                        local_frequency=local_frequency,
                    )
                )
            )
        low, high = results
        self.assertEqual(low.fate, high.fate)
        self.assertEqual(low.ray, high.ray)
        self.assertEqual(
            low.normalized_launch.model_descriptor_sha256,
            high.normalized_launch.model_descriptor_sha256,
        )
        self.assertEqual(
            low.emitter_to_receiver_frequency_ratio,
            high.emitter_to_receiver_frequency_ratio,
        )
        self.assertEqual(
            low.receiver_incidence_cosine,
            high.receiver_incidence_cosine,
        )
        self.assertNotEqual(
            low.launch.model_descriptor_sha256,
            high.launch.model_descriptor_sha256,
        )

    def test_tampered_launch_surface_and_termination_provenance_fail_closed(self) -> None:
        launch = self.launch(
            rho=8.0,
            face=UPPER,
            mu=0.02,
            azimuth=0.5 * math.pi,
        )
        object.__setattr__(launch, "emission_angle_cosine", 0.3)
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "launch\\.",
        ):
            self.trace(launch)

        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=30.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        clean_launch = self.launch(
            rho=8.0,
            face=UPPER,
            mu=0.02,
            azimuth=0.5 * math.pi,
            metric=metric,
            calibration=calibration,
        )
        object.__setattr__(metric, "source_id", "forged-kerr-provider")
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "surface\\.metric\\.source_id|identity",
        ):
            trace_kerr_returning_radiation_direction(
                clean_launch,
                surface,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
            )

        forged_termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        object.__setattr__(forged_termination, "spin_a_m", -0.7)
        with self.assertRaisesRegex(ValueError, "signed spins"):
            trace_kerr_returning_radiation_direction(
                self.launch(
                    rho=8.0,
                    face=UPPER,
                    mu=0.02,
                    azimuth=0.5 * math.pi,
                ),
                self.surface,
                termination=forged_termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
            )

    def test_accuracy_policy_and_unresolved_budget_fail_closed(self) -> None:
        launch = self.launch(
            rho=8.0,
            face=UPPER,
            mu=1.0,
            azimuth=0.0,
        )
        with self.assertRaisesRegex(ValueError, "accuracy policy"):
            trace_kerr_returning_radiation_direction(
                launch,
                self.surface,
                termination=self.termination,
                ray_options=RayTraceOptions(relative_tolerance=2.0e-7),
                surface_options=self.surface_options,
            )
        with self.assertRaisesRegex(
            KerrReturningRadiationRayError,
            "did not reach a certified fate",
        ):
            trace_kerr_returning_radiation_direction(
                launch,
                self.surface,
                termination=self.termination,
                ray_options=RayTraceOptions(
                    initial_step=0.01,
                    maximum_step=0.05,
                    maximum_affine_length=0.1,
                ),
                surface_options=self.surface_options,
            )

    def test_option_float_subclasses_cannot_bypass_fine_or_coarse_policy(self) -> None:
        launch = self.launch(
            rho=8.0,
            face=UPPER,
            mu=1.0,
            azimuth=0.0,
        )
        permissive_coarse_ray = RayTraceOptions(
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
            initial_step=0.05,
            maximum_step=0.6,
            maximum_affine_length=200.0,
            null_residual_limit=1.0e-6,
            metric_interpolation_error_limit=1.0e-6,
            event_value_tolerance=6.0e-8,
            event_affine_tolerance=6.0e-9,
        )
        permissive_coarse_surface = SurfaceEventOptions(
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
            null_residual_limit=1.0e-6,
            metric_interpolation_error_limit=1.0e-6,
            surface_value_tolerance=6.0e-8,
            affine_tolerance=6.0e-9,
            subdivisions_per_segment=2,
        )
        bypass = PolicyBypassFloat(1.0e-2)
        cases = (
            {
                "ray_options": replace(
                    self.ray_options,
                    absolute_tolerance=bypass,
                    relative_tolerance=bypass,
                ),
                "surface_options": self.surface_options,
            },
            {
                "ray_options": self.ray_options,
                "surface_options": replace(
                    self.surface_options,
                    absolute_tolerance=bypass,
                    relative_tolerance=bypass,
                ),
            },
            {
                "ray_options": self.ray_options,
                "surface_options": self.surface_options,
                "coarse_ray_options": replace(
                    permissive_coarse_ray,
                    absolute_tolerance=bypass,
                    relative_tolerance=bypass,
                ),
                "coarse_surface_options": permissive_coarse_surface,
            },
            {
                "ray_options": self.ray_options,
                "surface_options": self.surface_options,
                "coarse_ray_options": permissive_coarse_ray,
                "coarse_surface_options": replace(
                    permissive_coarse_surface,
                    absolute_tolerance=bypass,
                    relative_tolerance=bypass,
                ),
            },
        )
        for index, options in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaisesRegex(
                    KerrReturningRadiationRayError,
                    "non-exact field type",
                ):
                    trace_kerr_returning_radiation_direction(
                        launch,
                        self.surface,
                        termination=self.termination,
                        **options,
                    )


if __name__ == "__main__":
    unittest.main()
