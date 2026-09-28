from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from unittest import mock

from offline.adaptive_frame import AdaptivePixelOptions
from offline.geodesic import (
    RayTraceOptions,
    SurfaceEventOptions,
    trace_refined_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_frame import KerrFiniteThicknessRaySampler
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    UPPER_SURFACE_ID,
    KerrFiniteThicknessMultiSurface,
)
import offline.kerr_returning_radiation_finite_thickness_frame as frame_module
from offline.kerr_returning_radiation_finite_thickness_frame import (
    BINARY64_RADIUS_FLOOR_ULPS,
    IMPLEMENTATION_ID,
    MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER,
    SCIENTIFIC_STATUS,
    KerrReturningRadiationFiniteThicknessFrameError,
    KerrReturningRadiationFiniteThicknessRaySampler,
    integrate_returning_thermal_spectral_pixel,
)
import offline.kerr_returning_radiation_finite_thickness_transfer as transfer_module
from offline.kerr_returning_radiation_frame_context import (
    KerrReturningRadiationFrameContextVerificationError,
    authenticate_returning_thermal_emission,
)
import offline.kerr_returning_radiation_kernel as kernel_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_thermal_profile import (
    solve_certified_kerr_returning_radiation_thermal_profile,
)
from offline.kerr_returning_radiation_thermal_spectrum import (
    build_certified_returning_radiation_thermal_spectrum_provider,
)


SOLAR_MASS_KG = 1.98847e30


class AlwaysEqualFloat(float):
    def __eq__(self, other):
        return True


class _CodeCallCounter:
    """Count exact Python code-object calls without rebinding production APIs."""

    def __init__(self, *callables):
        self._codes = {entry.__code__ for entry in callables}
        self._counts = {code: 0 for code in self._codes}
        self._previous = None

    def __enter__(self):
        self._previous = sys.getprofile()

        def profile(frame, event, argument):
            if event == "call" and frame.f_code in self._counts:
                self._counts[frame.f_code] += 1
            if self._previous is not None:
                self._previous(frame, event, argument)

        sys.setprofile(profile)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        sys.setprofile(self._previous)

    def count(self, entry) -> int:
        return self._counts[entry.__code__]


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _low_return_classifier(*arguments):
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    receiver_face = UPPER if tangent_azimuth < math.pi else LOWER
    ratio = 0.2
    return kernel_module._DirectionTransport(
        "return-upper" if receiver_face == UPPER else "return-lower",
        receiver_face,
        source_radius,
        ratio,
        ratio * ratio,
        _identity_for(*arguments[6:]),
        receiver_face,
        source_radius,
    )


def _identity_classifier(*arguments):
    return kernel_module._DirectionTransport(
        "escaped",
        None,
        None,
        None,
        0.0,
        _identity_for(*arguments[6:]),
    )


class KerrReturningRadiationFiniteThicknessFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(cls.metric, cls.calibration)
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
            subdivisions_per_segment=4,
        )
        cls.disk = StationaryNovikovThorneDisk(
            metric=cls.metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
            colour_correction=1.7,
        )
        probe = cls.make_base.__func__(cls, observer_theta_rad=1.1)
        initial = kerr_zamo_camera_ray(
            cls.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        refinement = trace_refined_null_geodesic(
            cls.metric,
            initial,
            termination=cls.termination,
            multi_interior_surface=cls.surface,
            surface_options=cls.surface_options,
            fine_options=cls.ray_options,
            record_coarse_path=True,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
        )
        cls.cached_initial = initial
        cls.cached_refinement = refinement
        fine_ray, coarse_ray = probe._validate_refinement(refinement, initial)
        fine_base = probe._transfer(
            fine_ray,
            initial,
            (5.0e14,),
            ray_options=probe.fine_options,
            surface_options=probe.surface_options,
        )
        coarse_base = probe._transfer(
            coarse_ray,
            initial,
            (5.0e14,),
            ray_options=probe._coarse_ray_options,
            surface_options=probe._coarse_surface_options,
        )
        fine_rho = fine_base.pseudo_cylindrical_radius_over_mass
        coarse_rho = coarse_base.pseudo_cylindrical_radius_over_mass
        if fine_rho == coarse_rho:
            raise AssertionError("real fine/coarse probe needs distinct radii")
        cls.real_seam = 0.5 * (fine_rho + coarse_rho)
        cls.real_near_edge = math.nextafter(
            min(fine_rho, coarse_rho),
            -math.inf,
        )

        inner = float(cls.calibration.isco_radius_over_mass)
        cls.regular_edges = (inner, 10.0, 25.0)
        cls.seam_edges = (inner, cls.real_seam, 25.0)
        cls.near_edge_edges = (inner, cls.real_near_edge, 25.0)
        cls.low = cls.build_authority.__func__(
            cls,
            cls.regular_edges,
            _low_return_classifier,
        )
        cls.identity = cls.build_authority.__func__(
            cls,
            cls.regular_edges,
            _identity_classifier,
        )
        cls.seam = cls.build_authority.__func__(
            cls,
            cls.seam_edges,
            _low_return_classifier,
        )
        cls.near_edge = cls.build_authority.__func__(
            cls,
            cls.near_edge_edges,
            _low_return_classifier,
        )

    @classmethod
    def make_base(
        cls,
        *,
        observer_theta_rad: float,
        escaped=None,
    ) -> KerrFiniteThicknessRaySampler:
        return KerrFiniteThicknessRaySampler(
            metric=cls.metric,
            observer_radius_m=30.0,
            termination=cls.termination,
            surface=cls.surface,
            disk=cls.disk,
            escaped_observer_spectrum=(
                DarkEscapedObserverSpectrum() if escaped is None else escaped
            ),
            fine_options=cls.ray_options,
            surface_options=cls.surface_options,
            observer_theta_rad=observer_theta_rad,
            coarse_tolerance_multiplier=8.0,
            terminal_event_tolerance_m=2.0e-4,
            terminal_covector_tolerance=2.0e-4,
            specific_intensity_relative_tolerance=1.0e-3,
        )

    @classmethod
    def build_authority(cls, edges, classifier):
        kernel_policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
            symmetry_absolute_tolerance=1.0e-10,
            symmetry_relative_tolerance=1.0e-10,
            maximum_direction_evaluations=20_000,
            maximum_whole_ray_traces=80_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        with mock.patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=classifier,
        ):
            source = integrate_kerr_returning_radiation_energy_kernel(
                cls.surface,
                termination=cls.termination,
                annulus_edges_over_mass=edges,
                ray_options=cls.ray_options,
                surface_options=cls.surface_options,
                policy=kernel_policy,
                area_policy=area_policy,
            )
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                source,
                disk=cls.disk,
            )
            provider = build_certified_returning_radiation_thermal_spectrum_provider(
                profile
            )
            authority = authenticate_returning_thermal_emission(
                cls.surface,
                provider,
            )
        return {
            "source": source,
            "profile": profile,
            "provider": provider,
            "authority": authority,
        }

    def make_sampler(
        self,
        source,
        *,
        theta=1.1,
        escaped=None,
    ) -> KerrReturningRadiationFiniteThicknessRaySampler:
        return KerrReturningRadiationFiniteThicknessRaySampler(
            self.make_base(observer_theta_rad=theta, escaped=escaped),
            source["authority"],
        )

    def test_descriptor_binds_base_authority_and_honest_scientific_scope(self) -> None:
        sampler = self.make_sampler(self.low)
        descriptor = sampler.descriptor()
        self.assertEqual(descriptor["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(
            descriptor["authority"]["descriptorSha256"],
            self.low["authority"].snapshot.model_descriptor_sha256,
        )
        self.assertEqual(
            descriptor["authority"]["profileDescriptorSha256"],
            self.low["profile"].model_descriptor_sha256,
        )
        self.assertEqual(
            descriptor["baseSampler"]["descriptor"]["implementationId"],
            "kerr-finite-thickness-spectral-ray-sampler/v1",
        )
        policy = descriptor["annulusEdgeClearancePolicy"]
        self.assertEqual(policy["actual"]["binary64RadiusFloorUlps"], 32)
        self.assertEqual(
            policy["maxima"]["multiplier"],
            MAXIMUM_ANNULUS_EDGE_CLEARANCE_MULTIPLIER,
        )
        self.assertEqual(BINARY64_RADIUS_FLOOR_ULPS, 32)
        self.assertEqual(policy["singleAnnulus"], "not-applicable-no-internal-edge")
        for key in (
            "isContinuumRadialReturningRadiationSolution",
            "hasIndependentPhysicsOracle",
            "isCompleteKerrbb",
            "includesReturningRadiationStressWorkFS",
            "includesScatteringOrSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        json.dumps(descriptor, allow_nan=False, sort_keys=True)
        with self.assertRaisesRegex(ValueError, "policy maximum"):
            KerrReturningRadiationFiniteThicknessRaySampler(
                sampler.base_sampler,
                self.low["authority"],
                annulus_edge_clearance_multiplier=65.0,
            )

    def test_real_upper_lower_capture_and_escape_all_pass_full_audits(self) -> None:
        upper = self.make_sampler(self.low).sample(
            0.5,
            -0.5,
            (3.0e14, 5.0e14),
        )
        lower = self.make_sampler(self.low, theta=math.pi - 1.1).sample(
            0.5,
            0.5,
            (3.0e14, 5.0e14),
        )
        background = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=2.5,
            reference_frequency_hz=5.0e14,
            spectral_index=-1.0,
        )
        captured = self.make_sampler(self.low, theta=0.2, escaped=background).sample(
            0.0,
            0.0,
            (2.5e14, 5.0e14),
        )
        escaped = self.make_sampler(self.low, theta=0.4, escaped=background).sample(
            3.0,
            3.0,
            (2.5e14, 5.0e14),
        )
        for sample, source in (
            (upper, "disk"),
            (lower, "disk"),
            (captured, "captured-boundary"),
            (escaped, "escaped-boundary"),
        ):
            with self.subTest(source=source):
                self.assertEqual(sample.visible_source, source)
                self.assertTrue(sample.ray_converged)
                self.assertTrue(sample.convergence_audit.ray_gate_passed)
                self.assertTrue(sample.convergence_audit.source_gate_passed)
                self.assertTrue(sample.convergence_audit.transfer_gate_passed)
        self.assertIn(UPPER_SURFACE_ID, upper.topology_signature)
        self.assertIn(LOWER_SURFACE_ID, lower.topology_signature)
        self.assertTrue(
            all(
                value == 0.0 and math.copysign(1.0, value) > 0.0
                for value in captured.specific_intensities_nu
            )
        )
        for actual, expected in zip(
            escaped.specific_intensities_nu,
            (5.0, 2.5),
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertIsNotNone(escaped.escape_direction)

    def test_real_same_bin_passes_and_real_fine_coarse_seam_fails(self) -> None:
        regular = self.make_sampler(self.low).sample(
            0.5,
            -0.5,
            (5.0e14,),
        )
        self.assertEqual(regular.visible_source, "disk")
        self.assertGreater(regular.convergence_audit.disk_radius_difference_m, 0.0)
        with self.assertRaisesRegex(
            KerrReturningRadiationFiniteThicknessFrameError,
            "different thermal annuli",
        ):
            self.make_sampler(self.seam).sample(
                0.5,
                -0.5,
                (5.0e14,),
            )
        with self.assertRaisesRegex(
            KerrReturningRadiationFiniteThicknessFrameError,
            "too close to an internal thermal annulus edge",
        ):
            # Both real traces lie on the right of this edge, so the same-bin
            # gate passes; the independent clearance gate must still reject.
            self.make_sampler(self.near_edge).sample(
                0.5,
                -0.5,
                (5.0e14,),
            )

    def test_identity_and_low_return_change_only_authenticated_thermal_emission(
        self,
    ) -> None:
        low = self.make_sampler(self.low).sample(0.5, -0.5, (3.0e14, 5.0e14))
        identity = self.make_sampler(self.identity).sample(
            0.5,
            -0.5,
            (3.0e14, 5.0e14),
        )
        self.assertEqual(low.topology_signature, identity.topology_signature)
        self.assertEqual(low.frequency_shift_g.hex(), identity.frequency_shift_g.hex())
        self.assertNotEqual(low.specific_intensities_nu, identity.specific_intensities_nu)
        self.assertGreater(
            math.fsum(low.specific_intensities_nu),
            math.fsum(identity.specific_intensities_nu),
        )
        self.assertTrue(
            all(value == 0.0 for value in self.identity["profile"].incident_returning_flux_w_m2)
        )
        self.assertTrue(
            any(value > 0.0 for value in self.low["profile"].incident_returning_flux_w_m2)
        )

    def test_disk_sample_uses_two_base_replays_two_batches_and_two_live_gates(
        self,
    ) -> None:
        sampler = self.make_sampler(self.low)
        authority_type = type(self.low["authority"])
        original_live = authority_type.require_live
        original_batch = transfer_module._AUTHORITY_BATCH_ENTRY
        original_base = transfer_module._BASE_TRANSFER_ENTRY

        def live(bound):
            return original_live(bound)

        calls = _CodeCallCounter(original_base, original_batch)
        with calls, mock.patch.object(
            authority_type,
            "require_live",
            autospec=True,
            side_effect=live,
        ) as live_gate:
            sample = sampler.sample(0.5, -0.5, (5.0e14,))
        self.assertTrue(sample.ray_converged)
        self.assertEqual(live_gate.call_count, 2)
        self.assertEqual(calls.count(original_base), 2)
        self.assertEqual(calls.count(original_batch), 2)

    def test_capture_and_escape_never_batch_disk_emission(self) -> None:
        batch_entry = transfer_module._AUTHORITY_BATCH_ENTRY
        background = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=2.5,
            reference_frequency_hz=5.0e14,
            spectral_index=-1.0,
        )
        calls = _CodeCallCounter(batch_entry)
        with calls:
            self.make_sampler(self.low, theta=0.2).sample(0.0, 0.0, (5.0e14,))
            self.make_sampler(self.low, theta=0.4, escaped=background).sample(
                3.0,
                3.0,
                (5.0e14,),
            )
        self.assertEqual(calls.count(batch_entry), 0)

    def test_mutated_trusted_transfer_wrapper_fails_before_sample_escape(self) -> None:
        sampler = self.make_sampler(self.low)
        original = (
            transfer_module
            ._transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted
        )

        def mutate(*arguments, **keywords):
            result = original(*arguments, **keywords)
            object.__setattr__(result, "_descriptor_sha256", "0" * 64)
            return result

        with mock.patch.object(
            transfer_module,
            "_transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted",
            side_effect=mutate,
        ) as wrapper, self.assertRaisesRegex(
            KerrReturningRadiationFiniteThicknessFrameError,
            "callable changed",
        ):
            sampler.sample(0.5, -0.5, (5.0e14,))
        wrapper.assert_not_called()

    def test_rebound_base_batch_d20_and_result_builder_fail_before_use(self) -> None:
        sampler = self.make_sampler(self.low)
        authority_type = type(self.low["authority"])

        class ForgedD20:
            def intensity_multiplier(self, emission_angle_cosine):
                del emission_angle_cosine
                return 100.0

        cases = (
            (
                transfer_module,
                "transfer_kerr_finite_thickness_spectrum",
                mock.Mock(side_effect=AssertionError("rebound base ran")),
                "base-transfer callable changed",
            ),
            (
                authority_type,
                "emitted_specific_intensity_nu_batch",
                mock.Mock(return_value=(1.0,)),
                "thermal-batch callable changed",
            ),
            (
                transfer_module,
                "FluxConservingLinearLimbDarkening",
                ForgedD20,
                "D20 angular-law callable changed",
            ),
            (
                transfer_module,
                "_build_result_from_base",
                mock.Mock(side_effect=AssertionError("rebound builder ran")),
                "result builder changed",
            ),
        )
        for owner, name, replacement, message in cases:
            with self.subTest(binding=name), mock.patch.object(
                owner,
                name,
                replacement,
            ), self.assertRaisesRegex(RuntimeError, message):
                sampler.sample(0.5, -0.5, (5.0e14,))
            if isinstance(replacement, mock.Mock):
                replacement.assert_not_called()

    def test_multiplier_mutation_and_sampler_subclass_fail_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):
            class ForgedSampler(KerrReturningRadiationFiniteThicknessRaySampler):
                pass

        sampler = self.make_sampler(self.near_edge)
        original = sampler.annulus_edge_clearance_multiplier
        mutations = (
            0.0,
            -0.0,
            -1.0,
            AlwaysEqualFloat(original),
            math.nextafter(original, math.inf),
            65.0,
        )
        for changed in mutations:
            with self.subTest(changed=repr(changed)):
                object.__setattr__(
                    sampler,
                    "annulus_edge_clearance_multiplier",
                    changed,
                )
                try:
                    with self.assertRaisesRegex(
                        KerrReturningRadiationFiniteThicknessFrameError,
                        "multiplier changed",
                    ):
                        sampler.sample(0.5, -0.5, (5.0e14,))
                finally:
                    object.__setattr__(
                        sampler,
                        "annulus_edge_clearance_multiplier",
                        original,
                    )
        with self.assertRaisesRegex(
            KerrReturningRadiationFiniteThicknessFrameError,
            "too close to an internal thermal annulus edge",
        ):
            sampler.sample(0.5, -0.5, (5.0e14,))

    def test_adaptive_minimum_pixel_has_only_two_live_gates(self) -> None:
        sampler = self.make_sampler(self.low)
        authority_type = type(self.low["authority"])
        original_live = authority_type.require_live
        hot_entry = frame_module._SAMPLER_TRUSTED_ENTRY

        def live(bound):
            return original_live(bound)

        calls = _CodeCallCounter(hot_entry)
        with calls, mock.patch.object(
            authority_type,
            "require_live",
            autospec=True,
            side_effect=live,
        ) as live_gate, mock.patch.object(
            frame_module,
            "kerr_zamo_camera_ray",
            return_value=self.cached_initial,
        ), mock.patch.object(
            frame_module.base_frame_module,
            "trace_refined_null_geodesic",
            return_value=self.cached_refinement,
        ):
            pixel = integrate_returning_thermal_spectral_pixel(
                sampler,
                (5.0e14,),
                x_min=-1.0e-3,
                x_max=1.0e-3,
                y_min=-1.0e-3,
                y_max=1.0e-3,
                options=AdaptivePixelOptions(
                    minimum_depth=0,
                    maximum_depth=0,
                    maximum_ray_evaluations=32,
                    # This test certifies the whole-pixel authority/call path,
                    # not a zero-error quadrature claim.  The cached genuine
                    # fine/coarse witness retains its (small) ray error, so
                    # give the one-bin adaptive audit a conservative absolute
                    # budget below the declared radiance guard ceiling.
                    radiance_absolute_tolerances=(2.0,),
                    radiance_relative_tolerance=1.0e-12,
                    radiance_guard_ceilings=(2.0,),
                    weighted_log_g_tolerance=0.0,
                ),
            )
        self.assertTrue(pixel.converged)
        self.assertTrue(pixel.all_ray_gates_passed)
        self.assertTrue(pixel.all_source_gates_passed)
        self.assertTrue(pixel.all_transfer_gates_passed)
        self.assertEqual(pixel.sample_count, calls.count(hot_entry))
        self.assertGreater(pixel.sample_count, 1)
        self.assertEqual(live_gate.call_count, 2)

    def test_pixel_end_gate_catches_live_authority_mutation(self) -> None:
        sampler = self.make_sampler(self.low)
        provider = self.low["provider"]
        original_sha = object.__getattribute__(provider, "_descriptor_sha256")

        def mutate_then_abort(*arguments, **keywords):
            del arguments, keywords
            object.__setattr__(provider, "_descriptor_sha256", "0" * 64)
            raise AssertionError("pixel body aborted after live mutation")

        try:
            with mock.patch.object(
                frame_module.base_frame_module,
                "trace_refined_null_geodesic",
                side_effect=mutate_then_abort,
            ), self.assertRaises(
                KerrReturningRadiationFrameContextVerificationError
            ):
                integrate_returning_thermal_spectral_pixel(
                    sampler,
                    (5.0e14,),
                    x_min=-1.0e-3,
                    x_max=1.0e-3,
                    y_min=-1.0e-3,
                    y_max=1.0e-3,
                    options=AdaptivePixelOptions(
                        minimum_depth=0,
                        maximum_depth=0,
                        maximum_ray_evaluations=32,
                        radiance_absolute_tolerances=(0.0,),
                        radiance_relative_tolerance=0.0,
                        radiance_guard_ceilings=(2.0,),
                        weighted_log_g_tolerance=0.0,
                    ),
                )
        finally:
            object.__setattr__(provider, "_descriptor_sha256", original_sha)
        self.assertIs(self.low["authority"].require_live(), sampler._authenticated_snapshot)


if __name__ == "__main__":
    unittest.main()
