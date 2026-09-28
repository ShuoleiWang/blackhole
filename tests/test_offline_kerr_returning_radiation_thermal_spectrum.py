from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from decimal import Inexact, ROUND_UP, Rounded, localcontext

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import (
    BOLTZMANN_CONSTANT_J_K,
    LIGHT_SPEED_M_S,
    StationaryNovikovThorneDisk,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as kernel_module
import offline.kerr_returning_radiation_kernel_cached as cached_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_thermal_profile import (
    CACHED_KERR_FORWARD_CLASSIFICATION,
    REPLAYED_KERR_FORWARD_CLASSIFICATION,
    AxisymmetricReturningRadiationThermalProfile,
    solve_axisymmetric_returning_radiation_thermal_profile,
    solve_certified_kerr_returning_radiation_thermal_profile,
)
import offline.kerr_returning_radiation_thermal_spectrum as spectrum_module
from offline.kerr_returning_radiation_thermal_spectrum import (
    FREQUENCY_FRAME,
    IMPLEMENTATION_ID,
    PLANCK_IMPLEMENTATION_ID,
    QUERY_BOUNDARY_CONVENTION,
    SCIENTIFIC_STATUS,
    SPECIFIC_INTENSITY_UNITS,
    CertifiedReturningRadiationThermalSpectrumProvider,
    CertifiedReturningRadiationThermalSpectrumSample,
    KerrReturningRadiationThermalSpectrumError,
    KerrReturningRadiationThermalSpectrumVerificationError,
    build_certified_returning_radiation_thermal_spectrum_provider,
    colour_corrected_planck_specific_intensity_nu,
    verify_certified_returning_radiation_thermal_spectrum_provider,
    verify_certified_returning_radiation_thermal_spectrum_sample,
)
from offline.novikov_thorne import PROGRADE, kerr_isco_radius_m
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


class _FloatSubclass(float):
    pass


class _ProfileSubclass(AxisymmetricReturningRadiationThermalProfile):
    pass


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _zero_or_return_classifier(*arguments):
    surface = arguments[0]
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    if surface.calibration.dimensionless_spin < 0.45:
        return kernel_module._DirectionTransport(
            "escaped",
            None,
            None,
            None,
            0.0,
            _identity_for(*arguments[6:]),
        )
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


class ReturningRadiationThermalSpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._patcher = patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=_zero_or_return_classifier,
        )
        cls._patcher.start()
        cls.zero_source, cls.zero_disk = cls._build_source_and_disk(0.4)
        cls.return_source, cls.return_disk = cls._build_source_and_disk(0.5)
        cls._temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        cls.cache_root = Path(cls._temporary.name)
        cls.cached_source = (
            cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
                cls.return_source.surface,
                termination=cls.return_source.termination,
                annulus_edges_over_mass=cls.return_source.annulus_edges_over_mass,
                cache_root=cls.cache_root / "spectrum",
                ray_options=cls.return_source.ray_options,
                surface_options=cls.return_source.surface_options,
                coarse_ray_options=cls.return_source.coarse_ray_options,
                coarse_surface_options=cls.return_source.coarse_surface_options,
                policy=cls.return_source.policy,
                area_policy=cls.return_source.area_policy,
                directions_per_task=16,
                jobs=1,
            )
        )
        cls.zero_profile = solve_certified_kerr_returning_radiation_thermal_profile(
            cls.zero_source,
            disk=cls.zero_disk,
        )
        cls.return_profile = solve_certified_kerr_returning_radiation_thermal_profile(
            cls.return_source,
            disk=cls.return_disk,
        )
        cls.cached_profile = solve_certified_kerr_returning_radiation_thermal_profile(
            cls.cached_source,
            disk=cls.return_disk,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()
        cls._temporary.cleanup()

    @staticmethod
    def _build_source_and_disk(spin: float):
        metric = KerrKerrSchildMetric(mass_m=1.0, spin_a_m=spin)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=12.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=20.0,
            offset_m=0.02,
        )
        ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=100.0,
        )
        surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )
        kernel_policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
            symmetry_absolute_tolerance=1.0e-10,
            symmetry_relative_tolerance=1.0e-10,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=8,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=128,
        )
        edges = (
            float(calibration.isco_radius_over_mass),
            float(calibration.outer_radius_over_mass),
        )
        source = integrate_kerr_returning_radiation_energy_kernel(
            surface,
            termination=termination,
            annulus_edges_over_mass=edges,
            ray_options=ray_options,
            surface_options=surface_options,
            policy=kernel_policy,
            area_policy=area_policy,
        )
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=PROGRADE,
            colour_correction=1.7,
        )
        return source, disk

    @staticmethod
    def _forge(value, **changes):
        forged = object.__new__(type(value))
        for name in type(value).__dataclass_fields__:
            object.__setattr__(
                forged,
                name,
                changes.get(name, object.__getattribute__(value, name)),
            )
        return forged

    def test_scientific_scope_units_and_descriptor_binding_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(
            SCIENTIFIC_STATUS["requiredProfileClassification"],
            REPLAYED_KERR_FORWARD_CLASSIFICATION,
        )
        self.assertTrue(SCIENTIFIC_STATUS["profileGeometryMustBeCertified"])
        self.assertTrue(SCIENTIFIC_STATUS["piecewiseConstantAnnulusTemperature"])
        self.assertTrue(
            SCIENTIFIC_STATUS["renderLoopRequiresPriorProviderRevalidation"]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS["batchBuildsPerFrequencyEvidenceSamples"]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS["sharesPlanckAndNovikovThorneConstantsWithKerrDisk"]
        )
        for key in (
            "hasIndependentPlanckOracle",
            "appliesD20AngularLaw",
            "appliesTransferG3",
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesScattering",
            "includesSolvedAtmosphere",
            "includesPolarization",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)

        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        descriptor = provider.model_descriptor()
        self.assertEqual(descriptor["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(
            descriptor["binding"]["sourceProfileDescriptorSha256"],
            self.zero_profile.model_descriptor_sha256,
        )
        self.assertEqual(
            descriptor["binding"]["novikovThorneDiskDescriptorSha256"],
            self.zero_profile.novikov_thorne_disk_descriptor_sha256,
        )
        self.assertEqual(
            descriptor["emission"]["specificIntensityUnits"],
            SPECIFIC_INTENSITY_UNITS,
        )
        self.assertEqual(descriptor["emission"]["frequencyFrame"], FREQUENCY_FRAME)
        self.assertEqual(
            descriptor["annuli"]["queryBoundaryConvention"],
            QUERY_BOUNDARY_CONVENTION,
        )
        self.assertFalse(descriptor["emission"]["includesD20"])
        self.assertFalse(descriptor["emission"]["includesTransferG3"])
        hot_path = descriptor["emission"]["batchHotPath"]
        self.assertTrue(hot_path["callerMustRevalidateProviderBeforeRenderLoop"])
        self.assertTrue(hot_path["radiusAndAnnulusResolvedOncePerBatch"])
        self.assertFalse(hot_path["buildsPerFrequencyEvidenceSamples"])
        self.assertFalse(
            descriptor["emission"]["planck"]["independentOracle"]
        )
        self.assertEqual(
            descriptor["emission"]["planck"]["implementationId"],
            PLANCK_IMPLEMENTATION_ID,
        )
        self.assertEqual(len(provider.model_descriptor_sha256), 64)

    def test_cached_profile_provider_revalidates_without_ray_or_hot_path_scan(self) -> None:
        with (
            patch.object(
                kernel_module,
                "_trace_direction",
                side_effect=AssertionError("cached provider retraced a ray"),
            ) as tracer,
            patch.object(
                spectrum_module,
                "verify_axisymmetric_returning_radiation_thermal_profile",
                wraps=(
                    spectrum_module
                    .verify_axisymmetric_returning_radiation_thermal_profile
                ),
            ) as profile_verifier,
        ):
            provider = build_certified_returning_radiation_thermal_spectrum_provider(
                self.cached_profile
            )
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(profile_verifier.call_count, 1)
        self.assertEqual(
            provider.model_descriptor()["binding"][
                "profileProvenanceClassification"
            ],
            CACHED_KERR_FORWARD_CLASSIFICATION,
        )
        with patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=AssertionError("provider revalidation retraced a ray"),
        ):
            provider.revalidate()
        radius = provider.annulus_edges_over_mass[0]
        with patch.object(
            spectrum_module,
            "verify_certified_returning_radiation_thermal_spectrum_provider",
            side_effect=AssertionError("hot batch rescanned the cache"),
        ):
            values = provider.emitted_specific_intensity_nu_batch(
                radius,
                (1.0e14, 2.0e14, 3.0e14),
            )
        self.assertEqual(len(values), 3)
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["appliesTransferG3"] = True

    def test_provider_rejects_stale_live_axis_before_ray_or_cache_replay(self) -> None:
        projection = self.return_source.coarsen_annuli(
            self.return_source.annulus_edges_over_mass
        )
        profiles = (
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.return_source,
                disk=self.return_disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                projection,
                disk=self.return_disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.cached_source,
                disk=self.return_disk,
            ),
        )
        for profile in profiles:
            kernel = object.__getattribute__(profile, "_kernel")
            rows = object.__getattribute__(
                kernel,
                "receiver_emitter_coefficients",
            )
            self.assertTrue(any(value != 0.0 for row in rows for value in row))
            count = len(rows)
            object.__setattr__(
                kernel,
                "receiver_emitter_coefficients",
                tuple(tuple(0.0 for _ in range(count)) for _ in range(count)),
            )
            with (
                self.subTest(classification=profile.provenance_classification),
                patch.object(
                    kernel_module,
                    "_trace_direction",
                    side_effect=AssertionError(
                        "provider must reject stale live axis before ray replay"
                    ),
                ) as ray_trace,
                patch.object(
                    cached_module,
                    "_rebuild_verified_cached_execution",
                    side_effect=AssertionError(
                        "provider must reject stale live axis before cache scan"
                    ),
                ) as cache_replay,
                self.assertRaises(
                    KerrReturningRadiationThermalSpectrumVerificationError
                ),
            ):
                build_certified_returning_radiation_thermal_spectrum_provider(
                    profile
                )
            ray_trace.assert_not_called()
            cache_replay.assert_not_called()

    def test_identity_kernel_matches_stationary_nt_emission_bit_for_bit(self) -> None:
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        radius = self.zero_profile.annulus_representative_radii_over_mass[0]
        self.assertEqual(
            self.zero_profile.outgoing_effective_temperature_k,
            self.zero_profile.intrinsic_effective_temperature_k,
        )
        for frequency in (
            math.ulp(0.0),
            1.0e-100,
            1.0,
            1.0e10,
            1.0e15,
            1.0e25,
            1.0e100,
            1.0e308,
        ):
            with self.subTest(frequency=frequency):
                actual = provider.emitted_specific_intensity_nu(radius, frequency)
                expected = self.zero_disk.emitted_specific_intensity_nu(
                    radius * self.zero_disk.metric.mass_m,
                    frequency,
                )
                self.assertEqual(actual.hex(), expected.hex())

        sample = provider.sample(radius, 1.0e15)
        self.assertEqual(sample.specific_intensity_units, SPECIFIC_INTENSITY_UNITS)
        self.assertEqual(sample.frequency_frame, FREQUENCY_FRAME)
        self.assertEqual(
            sample.model_descriptor()["providerDescriptorSha256"],
            provider.model_descriptor_sha256,
        )
        provider.revalidate()
        sample.revalidate()
        verify_certified_returning_radiation_thermal_spectrum_provider(provider)
        verify_certified_returning_radiation_thermal_spectrum_sample(sample)

    def test_batch_hot_path_matches_scalar_and_builds_no_evidence_records(self) -> None:
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        provider.revalidate()
        radius = self.zero_profile.annulus_representative_radii_over_mass[0]
        frequencies = (1.0, 1.0e10, 1.0e15, 1.0e25, 1.0e100)
        samples = tuple(provider.sample(radius, value) for value in frequencies)
        expected = tuple(
            sample.isotropic_specific_intensity_nu for sample in samples
        )
        shared_planck = spectrum_module._shared_validated_planck_intensity
        with (
            patch.object(
                CertifiedReturningRadiationThermalSpectrumProvider,
                "sample",
                side_effect=AssertionError("hot path built a sample"),
            ),
            patch.object(
                spectrum_module,
                "_canonical_json",
                side_effect=AssertionError("hot path built a descriptor"),
            ),
            patch.object(
                spectrum_module.hashlib,
                "sha256",
                side_effect=AssertionError("hot path built a hash"),
            ),
            patch.object(
                spectrum_module,
                "_shared_validated_planck_intensity",
                wraps=shared_planck,
            ) as planck_calls,
        ):
            scalar = provider.emitted_specific_intensity_nu(
                radius,
                frequencies[2],
            )
            batch = provider.emitted_specific_intensity_nu_batch(
                radius,
                frequencies,
            )
        self.assertEqual(scalar.hex(), expected[2].hex())
        self.assertEqual(
            tuple(value.hex() for value in batch),
            tuple(value.hex() for value in expected),
        )
        self.assertEqual(planck_calls.call_count, len(frequencies) + 1)

        invalid_grids = (
            [],
            (),
            (1.0, 0.0),
            (1.0, _FloatSubclass(2.0)),
        )
        for grid in invalid_grids:
            with self.subTest(grid=grid), self.assertRaises((TypeError, ValueError)):
                provider.emitted_specific_intensity_nu_batch(radius, grid)

        tampered = self._forge(
            provider,
            outgoing_effective_temperature_k=(
                provider.outgoing_effective_temperature_k[0] * 1.01,
            ),
        )
        with self.assertRaises(
            KerrReturningRadiationThermalSpectrumVerificationError
        ):
            tampered.revalidate()

    def test_known_returning_flux_ratio_sets_fourth_root_temperature_and_spectrum(
        self,
    ) -> None:
        profile = self.return_profile
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            profile
        )
        intrinsic_flux = profile.intrinsic_flux_w_m2[0]
        outgoing_flux = profile.outgoing_flux_w_m2[0]
        flux_ratio = outgoing_flux / intrinsic_flux
        kernel_coefficient = profile._kernel.receiver_emitter_coefficients[0][0]
        expected_flux_ratio = 1.0 / (1.0 - kernel_coefficient)
        self.assertTrue(
            math.isclose(flux_ratio, expected_flux_ratio, rel_tol=3.0e-13)
        )
        expected_temperature_ratio = math.exp(0.25 * math.log(flux_ratio))
        actual_temperature_ratio = (
            profile.outgoing_effective_temperature_k[0]
            / profile.intrinsic_effective_temperature_k[0]
        )
        self.assertTrue(
            math.isclose(
                actual_temperature_ratio,
                expected_temperature_ratio,
                rel_tol=3.0e-15,
            )
        )
        self.assertGreater(flux_ratio, 1.0)

        radius = profile.annulus_representative_radii_over_mass[0]
        frequency = 1.0e15
        returned = provider.emitted_specific_intensity_nu(radius, frequency)
        intrinsic = self.return_disk.emitted_specific_intensity_nu(
            radius * self.return_disk.metric.mass_m,
            frequency,
        )
        self.assertGreater(returned, intrinsic)
        direct = colour_corrected_planck_specific_intensity_nu(
            profile.outgoing_effective_temperature_k[0],
            self.return_disk.colour_correction,
            frequency,
        )
        self.assertEqual(returned.hex(), direct.hex())

    def test_piecewise_radius_boundaries_match_profile_semantics(self) -> None:
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        inner, outer = provider.annulus_edges_over_mass
        self.assertEqual(provider.sample(inner, 1.0e15).annulus_index, 0)
        self.assertEqual(provider.sample(outer, 1.0e15).annulus_index, 0)
        for radius in (math.nextafter(inner, -math.inf), math.nextafter(outer, math.inf)):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                provider.sample(radius, 1.0e15)

    def test_planck_evaluator_covers_rj_wien_zero_and_binary64_extremes(self) -> None:
        temperature = 1.0e7
        correction = 1.7
        frequency = 1.0
        actual = colour_corrected_planck_specific_intensity_nu(
            temperature,
            correction,
            frequency,
        )
        rayleigh_jeans = (
            2.0
            * BOLTZMANN_CONSTANT_J_K
            * temperature
            * frequency
            * frequency
            / (LIGHT_SPEED_M_S * LIGHT_SPEED_M_S * correction**3)
        )
        self.assertTrue(math.isclose(actual, rayleigh_jeans, rel_tol=8.0e-14))
        self.assertGreater(
            colour_corrected_planck_specific_intensity_nu(
                temperature,
                correction,
                1.0e15,
            ),
            0.0,
        )
        self.assertEqual(
            colour_corrected_planck_specific_intensity_nu(
                temperature,
                correction,
                1.0e308,
            ),
            0.0,
        )
        zero = colour_corrected_planck_specific_intensity_nu(0.0, 1.0, 1.0)
        self.assertEqual(zero, 0.0)
        self.assertEqual(math.copysign(1.0, zero), 1.0)
        maximum_float = float.fromhex("0x1.fffffffffffffp+1023")
        for extreme_frequency in (math.ulp(0.0), maximum_float):
            with self.subTest(frequency=extreme_frequency):
                value = colour_corrected_planck_specific_intensity_nu(
                    temperature,
                    correction,
                    extreme_frequency,
                )
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
        half_subnormal_case = colour_corrected_planck_specific_intensity_nu(
            1.0e7,
            1.7,
            2.7243194629688948e20,
        )
        self.assertEqual(half_subnormal_case.hex(), math.ulp(0.0).hex())

    def test_invalid_and_overflowing_planck_inputs_fail_closed(self) -> None:
        invalid_calls = (
            (1, 1.0, 1.0),
            (1.0, 1, 1.0),
            (1.0, 1.0, 1),
            (-1.0, 1.0, 1.0),
            (1.0, 0.999, 1.0),
            (1.0, 1.0, 0.0),
            (math.nan, 1.0, 1.0),
            (1.0, math.inf, 1.0),
            (1.0, 1.0, math.nan),
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(
                (TypeError, ValueError)
            ):
                colour_corrected_planck_specific_intensity_nu(*arguments)
        with self.assertRaisesRegex(
            KerrReturningRadiationThermalSpectrumError,
            "colour temperature",
        ):
            colour_corrected_planck_specific_intensity_nu(1.0e308, 2.0, 1.0)
        with self.assertRaisesRegex(
            KerrReturningRadiationThermalSpectrumError,
            "intensity overflowed",
        ):
            colour_corrected_planck_specific_intensity_nu(1.0e300, 1.0, 1.0e200)
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
        neighbor = first
        for offset in range(-4, 5):
            with self.subTest(overflow_neighbor=offset), self.assertRaisesRegex(
                KerrReturningRadiationThermalSpectrumError,
                "intensity overflowed",
            ):
                colour_corrected_planck_specific_intensity_nu(
                    1.0e300,
                    1.0,
                    neighbor,
                )
            neighbor = math.nextafter(neighbor, math.inf)

    def test_shared_decimal_boundary_isolated_from_ambient_context(self) -> None:
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
            with self.assertRaisesRegex(
                KerrReturningRadiationThermalSpectrumError,
                "intensity overflowed",
            ):
                colour_corrected_planck_specific_intensity_nu(
                    1.0e300,
                    1.0,
                    float.fromhex("0x1.43f5e7c6af360p+79"),
                )

    def test_external_unverified_and_non_exact_profiles_are_rejected(self) -> None:
        disk = self.zero_disk
        isco = kerr_isco_radius_m(0.4, PROGRADE)
        edges = (isco, 12.0)
        kernel = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(0.5 * math.fsum(edges),),
            receiver_emitter_coefficients=((0.0,),),
            ray_kernel_producer_id="external-unit-test/v1",
        )
        profile = solve_axisymmetric_returning_radiation_thermal_profile(
            kernel,
            annulus_edges_over_mass=edges,
            annulus_proper_areas_over_mass_squared=(100.0,),
            disk=disk,
        )
        with self.assertRaisesRegex(ValueError, "geometry_provenance_certified"):
            build_certified_returning_radiation_thermal_spectrum_provider(profile)
        subclass = object.__new__(_ProfileSubclass)
        with self.assertRaises(TypeError):
            build_certified_returning_radiation_thermal_spectrum_provider(subclass)
        with self.assertRaises(TypeError):
            CertifiedReturningRadiationThermalSpectrumProvider()
        with self.assertRaises(TypeError):
            CertifiedReturningRadiationThermalSpectrumSample()

    def test_provider_profile_descriptor_hash_temperature_and_colour_tampering_fail(
        self,
    ) -> None:
        clean = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        forged_profile = self._forge(
            self.zero_profile,
            outgoing_effective_temperature_k=(
                self.zero_profile.outgoing_effective_temperature_k[0] * 1.01,
            ),
        )
        cases = (
            self._forge(
                clean,
                outgoing_effective_temperature_k=(
                    clean.outgoing_effective_temperature_k[0] * 1.01,
                ),
            ),
            self._forge(clean, colour_correction=1.8),
            self._forge(clean, source_profile_descriptor_sha256="0" * 64),
            self._forge(clean, _descriptor_sha256="0" * 64),
            self._forge(clean, _descriptor_json="{}"),
            self._forge(clean, _profile=forged_profile),
        )
        for index, forged in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(
                KerrReturningRadiationThermalSpectrumVerificationError
            ):
                forged.revalidate()

        wrong_type = self._forge(
            clean,
            colour_correction=_FloatSubclass(clean.colour_correction),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationThermalSpectrumVerificationError,
            "non-exact type",
        ):
            wrong_type.revalidate()

    def test_sample_radius_temperature_colour_descriptor_and_hash_tampering_fail(
        self,
    ) -> None:
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            self.zero_profile
        )
        radius = self.zero_profile.annulus_representative_radii_over_mass[0]
        clean = provider.sample(radius, 1.0e15)
        cases = (
            self._forge(clean, radius_over_mass=math.nextafter(radius, math.inf)),
            self._forge(
                clean,
                effective_temperature_k=clean.effective_temperature_k * 1.01,
            ),
            self._forge(clean, colour_correction=1.8),
            self._forge(clean, provider_descriptor_sha256="0" * 64),
            self._forge(clean, _descriptor_sha256="0" * 64),
            self._forge(clean, _descriptor_json="{}"),
        )
        for index, forged in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(
                KerrReturningRadiationThermalSpectrumVerificationError
            ):
                forged.revalidate()

        with self.assertRaises(TypeError):
            provider.sample(_FloatSubclass(radius), 1.0e15)
        with self.assertRaises(TypeError):
            provider.sample(radius, _FloatSubclass(1.0e15))


if __name__ == "__main__":
    unittest.main()
