from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import math
import sys
import unittest
from unittest import mock

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    RayTraceOptions,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_bl_zamo_tetrad,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import (
    STEFAN_BOLTZMANN_W_M2_K4,
    StationaryNovikovThorneDisk,
    colour_corrected_planck_specific_intensity_nu,
)
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
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)
import offline.kerr_returning_radiation_finite_thickness_transfer as transfer_module
from offline.kerr_returning_radiation_finite_thickness_transfer import (
    ANNULUS_QUERY_BOUNDARY_CONVENTION,
    IMPLEMENTATION_ID,
    MAXIMUM_FREQUENCY_BINS,
    SCIENTIFIC_STATUS,
    KerrReturningRadiationFiniteThicknessSpectrumResult,
    KerrReturningRadiationFiniteThicknessTransferVerificationError,
    transfer_kerr_returning_radiation_finite_thickness_spectrum,
)
from offline.kerr_returning_radiation_frame_context import (
    ValidatedReturningThermalAuthority,
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


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True


class AlwaysEqualInt(int):
    def __eq__(self, other):
        return True


class TupleSubclass(tuple):
    pass


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


def _zero_or_low_return_classifier(*arguments):
    surface = arguments[0]
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    if surface.calibration.dimensionless_spin < 0.65:
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


class KerrReturningRadiationFiniteThicknessTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._kernel_patcher = mock.patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=_zero_or_low_return_classifier,
        )
        cls._kernel_patcher.start()
        try:
            cls.returning = cls._build_system(0.7)
            cls.identity = cls._build_system(0.6)
            cls.returning_disk_ray = cls._trace(
                cls.returning,
                theta=1.1,
                screen_x=0.5,
                screen_y=-0.5,
            )
            cls.returning_capture_ray = cls._trace(
                cls.returning,
                theta=0.2,
                screen_x=0.0,
                screen_y=0.0,
            )
            cls.returning_escape_ray = cls._trace(
                cls.returning,
                theta=0.4,
                screen_x=3.0,
                screen_y=3.0,
            )
            cls.identity_disk_ray = cls._trace(
                cls.identity,
                theta=1.1,
                screen_x=0.5,
                screen_y=-0.5,
            )
        except BaseException:
            cls._kernel_patcher.stop()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls._kernel_patcher.stop()

    @classmethod
    def _build_system(cls, spin: float):
        metric = KerrKerrSchildMetric(spin_a_m=spin)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
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
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        source = integrate_kerr_returning_radiation_energy_kernel(
            surface,
            termination=termination,
            annulus_edges_over_mass=(
                float(calibration.isco_radius_over_mass),
                float(calibration.outer_radius_over_mass),
            ),
            ray_options=ray_options,
            surface_options=surface_options,
            policy=kernel_policy,
            area_policy=area_policy,
        )
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
            colour_correction=1.7,
        )
        profile = solve_certified_kerr_returning_radiation_thermal_profile(
            source,
            disk=disk,
        )
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            profile
        )
        authority = authenticate_returning_thermal_emission(surface, provider)
        return {
            "metric": metric,
            "calibration": calibration,
            "surface": surface,
            "termination": termination,
            "ray_options": ray_options,
            "surface_options": surface_options,
            "source": source,
            "disk": disk,
            "profile": profile,
            "provider": provider,
            "authority": authority,
        }

    @staticmethod
    def _trace(system, *, theta: float, screen_x: float, screen_y: float):
        initial = kerr_zamo_camera_ray(
            system["metric"],
            observer_radius_m=30.0,
            theta_rad=theta,
            screen_x=screen_x,
            screen_y=screen_y,
        )
        observer_velocity = kerr_bl_zamo_tetrad(
            system["metric"],
            observer_radius_m=30.0,
            theta_rad=theta,
        ).four_velocity
        ray = trace_null_geodesic(
            system["metric"],
            initial,
            termination=system["termination"],
            multi_interior_surface=system["surface"],
            surface_options=system["surface_options"],
            options=system["ray_options"],
        )
        return initial, observer_velocity, ray

    @staticmethod
    def _forge(original, **changes):
        forged = object.__new__(type(original))
        for item in fields(original):
            name = item.name
            object.__setattr__(
                forged,
                name,
                changes.get(name, object.__getattribute__(original, name)),
            )
        return forged

    def transfer(
        self,
        system,
        traced,
        *,
        frequencies=(3.0e14, 5.0e14),
        background=None,
        observer_velocity=None,
        authority=None,
    ):
        initial, stored_observer_velocity, ray = traced
        return transfer_kerr_returning_radiation_finite_thickness_spectrum(
            system["surface"],
            system["disk"],
            ray,
            initial,
            (
                stored_observer_velocity
                if observer_velocity is None
                else observer_velocity
            ),
            frequencies,
            termination=system["termination"],
            ray_options=system["ray_options"],
            surface_options=system["surface_options"],
            escaped_observer_spectrum=(
                DarkEscapedObserverSpectrum()
                if background is None
                else background
            ),
            authority=(system["authority"] if authority is None else authority),
        )

    def test_scope_constructor_and_public_boundary_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(SCIENTIFIC_STATUS["maximumFrequencyBins"], 4096)
        self.assertFalse(SCIENTIFIC_STATUS["trustedHotEntryIsPublic"])
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "recordsAnnulusIndexAndNearestInternalEdgeDistance"
            ]
        )
        self.assertIn(
            "piecewise-constant",
            SCIENTIFIC_STATUS["thermalRadialDiscretization"],
        )
        for key in (
            "hasIndependentPhysicsOracle",
            "isCompleteKerrbb",
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesScatteringOrSolvedAtmosphere",
            "includesPolarization",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertNotIn(
            "_transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted",
            transfer_module.__all__,
        )
        with self.assertRaises(TypeError):
            KerrReturningRadiationFiniteThicknessSpectrumResult()

    def test_piecewise_annulus_boundary_convention_is_explicit(self) -> None:
        snapshot = self.returning["authority"].snapshot
        inner, outer = snapshot.annulus_edges_over_mass
        middle = 0.5 * math.fsum((inner, outer))
        two_annuli = replace(
            snapshot,
            annulus_edges_over_mass=(inner, middle, outer),
        )
        self.assertEqual(
            transfer_module._annulus_binding(two_annuli, middle),
            (1, middle, outer, 0.0),
        )
        self.assertEqual(
            transfer_module._annulus_binding(two_annuli, outer),
            (1, middle, outer, outer - middle),
        )
        first_quarter = 0.5 * math.fsum((inner, middle))
        third_quarter = 0.5 * math.fsum((middle, outer))
        four_annuli = replace(
            snapshot,
            annulus_edges_over_mass=(
                inner,
                first_quarter,
                middle,
                third_quarter,
                outer,
            ),
        )
        for radius in (
            math.nextafter(first_quarter, inner),
            first_quarter,
            math.nextafter(first_quarter, middle),
            math.nextafter(middle, first_quarter),
            math.nextafter(middle, third_quarter),
            outer,
        ):
            expected = min(
                abs(radius - edge)
                for edge in four_annuli.annulus_edges_over_mass[1:-1]
            )
            actual = transfer_module._annulus_binding(four_annuli, radius)[3]
            self.assertEqual(actual.hex(), expected.hex())

        result = self.transfer(
            self.returning,
            self.returning_disk_ray,
            frequencies=(5.0e14,),
        )
        self.assertIn(
            "piecewise-constant",
            result.model_descriptor()["annulus"]["valueModel"],
        )

    def test_real_disk_ray_replays_base_once_batches_once_and_closes_chain(
        self,
    ) -> None:
        system = self.returning
        authority = system["authority"]
        authority_type = type(authority)
        original_base = transfer_module._BASE_TRANSFER_ENTRY
        original_require = authority_type.require_live
        original_batch = transfer_module._AUTHORITY_BATCH_ENTRY

        def require_once(bound):
            return original_require(bound)

        calls = _CodeCallCounter(original_base, original_batch)
        with calls, mock.patch.object(
            authority_type,
            "require_live",
            autospec=True,
            side_effect=require_once,
        ) as live_gate:
            result = self.transfer(system, self.returning_disk_ray)
        self.assertEqual(calls.count(original_base), 1)
        self.assertEqual(live_gate.call_count, 2)
        self.assertEqual(calls.count(original_batch), 1)

        base = result.base_result
        self.assertEqual(base.source_kind, "finite-thickness-disk")
        self.assertEqual(result.source_kind, "finite-thickness-disk")
        self.assertEqual(
            base.observer_frequencies_hz,
            (result.observer_frequencies_hz[0],),
        )
        self.assertEqual(
            result.geometry_witness_observer_frequency_hz.hex(),
            result.observer_frequencies_hz[0].hex(),
        )
        self.assertIn(
            "not used",
            result.model_descriptor()["geometryWitness"]["purpose"],
        )
        self.assertEqual(
            result.pseudo_cylindrical_radius_over_mass.hex(),
            base.pseudo_cylindrical_radius_over_mass.hex(),
        )
        self.assertEqual(result.face, base.face)
        self.assertEqual(result.frequency_shift_g.hex(), base.frequency_shift_g.hex())
        self.assertEqual(
            result.outgoing_emission_angle_cosine.hex(),
            base.photon_projection.outgoing_cosine.hex(),
        )
        expected_emitted_frequencies = tuple(
            frequency / result.frequency_shift_g
            for frequency in result.observer_frequencies_hz
        )
        self.assertEqual(result.emitted_frequencies_hz, expected_emitted_frequencies)
        expected_isotropic = original_batch(
            authority,
            result.pseudo_cylindrical_radius_over_mass,
            expected_emitted_frequencies,
        )
        self.assertEqual(
            result.isotropic_emitted_specific_intensities_nu,
            expected_isotropic,
        )
        expected_multiplier = FluxConservingLinearLimbDarkening().intensity_multiplier(
            result.outgoing_emission_angle_cosine
        )
        self.assertEqual(
            result.angular_emission_multiplier.hex(),
            expected_multiplier.hex(),
        )
        self.assertAlmostEqual(
            expected_multiplier,
            0.5 + 0.75 * result.outgoing_emission_angle_cosine,
            places=15,
        )
        expected_emitted = tuple(
            value * expected_multiplier for value in expected_isotropic
        )
        expected_observed = tuple(
            result.frequency_shift_g**3 * value for value in expected_emitted
        )
        self.assertEqual(result.emitted_specific_intensities_nu, expected_emitted)
        self.assertEqual(result.observed_specific_intensities_nu, expected_observed)

    def test_legacy_nt_geometry_witness_work_is_constant_in_bin_count(self) -> None:
        system = self.returning
        authority = system["authority"]
        authority_type = type(authority)
        original_nt = StationaryNovikovThorneDisk.emitted_specific_intensity_nu
        original_batch = transfer_module._AUTHORITY_BATCH_ENTRY

        def nt(bound, *args, **kwargs):
            return original_nt(bound, *args, **kwargs)

        grids = (
            tuple(1.0e14 + 1.0e11 * index for index in range(4)),
            tuple(1.0e14 + 1.0e11 * index for index in range(471)),
        )
        for frequencies in grids:
            calls = _CodeCallCounter(original_batch)
            with self.subTest(bin_count=len(frequencies)), calls, mock.patch.object(
                StationaryNovikovThorneDisk,
                "emitted_specific_intensity_nu",
                autospec=True,
                side_effect=nt,
            ) as nt_calls:
                result = self.transfer(
                    system,
                    self.returning_disk_ray,
                    frequencies=frequencies,
                )
            self.assertEqual(nt_calls.call_count, 2)
            self.assertEqual(calls.count(original_batch), 1)
            self.assertEqual(
                result.base_result.observer_frequencies_hz,
                (frequencies[0],),
            )
            self.assertEqual(
                len(result.observed_specific_intensities_nu),
                len(frequencies),
            )
        self.assertEqual(result.annulus_index, 0)
        self.assertIsNone(result.nearest_internal_annulus_edge_distance_over_mass)
        self.assertEqual(
            result.annulus_query_boundary_convention,
            ANNULUS_QUERY_BOUNDARY_CONVENTION,
        )
        snapshot = authority.snapshot
        self.assertEqual(
            result.authority_snapshot_descriptor_sha256,
            snapshot.model_descriptor_sha256,
        )
        self.assertEqual(
            result.profile_descriptor_sha256,
            snapshot.profile_descriptor_sha256,
        )
        self.assertEqual(
            result.underlying_kernel_descriptor_sha256,
            snapshot.underlying_kernel_descriptor_sha256,
        )
        descriptor = result.model_descriptor()
        self.assertFalse(descriptor["algebra"]["usesBolometricG4"])
        self.assertFalse(descriptor["scientificBoundary"]["implicitFactorOfTwo"])
        self.assertFalse(descriptor["scientificBoundary"]["addsIncidentFluxAgain"])

    def test_real_capture_is_positive_zero_and_escape_is_recomputed(self) -> None:
        batch_entry = transfer_module._AUTHORITY_BATCH_ENTRY
        calls = _CodeCallCounter(batch_entry)
        with calls:
            captured = self.transfer(
                self.returning,
                self.returning_capture_ray,
            )
            background = PowerLawEscapedObserverSpectrum(
                reference_specific_intensity_nu=5.0,
                reference_frequency_hz=2.5e14,
                spectral_index=-1.0,
            )
            escaped = self.transfer(
                self.returning,
                self.returning_escape_ray,
                frequencies=(2.5e14, 5.0e14),
                background=background,
            )
        self.assertEqual(calls.count(batch_entry), 0)
        self.assertEqual(captured.source_kind, "captured-boundary")
        self.assertEqual(captured.observed_specific_intensities_nu, (0.0, 0.0))
        self.assertTrue(
            all(
                math.copysign(1.0, value) == 1.0
                for value in captured.observed_specific_intensities_nu
            )
        )
        self.assertEqual(escaped.source_kind, "escaped-boundary")
        self.assertEqual(
            escaped.observed_specific_intensities_nu,
            tuple(
                background(
                    escaped.base_result.ray.terminal_state,
                    frequency,
                    escaped.base_result.ray.terminal_target_id,
                )
                for frequency in escaped.observer_frequencies_hz
            ),
        )
        for result in (captured, escaped):
            self.assertIsNone(result.annulus_index)
            self.assertIsNone(result.face)
            self.assertIsNone(result.frequency_shift_g)
            self.assertIsNone(result.emitted_frequencies_hz)
            self.assertIsNone(result.isotropic_emitted_specific_intensities_nu)

        with mock.patch.object(
            transfer_module,
            "transfer_kerr_finite_thickness_spectrum",
            side_effect=AssertionError("rebound base entry must not run"),
        ) as base_wrapper, self.assertRaisesRegex(
            KerrReturningRadiationFiniteThicknessTransferVerificationError,
            "base-transfer callable changed",
        ):
            self.transfer(
                self.returning,
                self.returning_escape_ray,
                frequencies=(2.5e14, 5.0e14),
                background=background,
            )
        base_wrapper.assert_not_called()

    def test_private_trusted_hot_entry_does_not_repeat_live_gate(self) -> None:
        system = self.returning
        authority = system["authority"]
        snapshot = authority.require_live()
        initial, observer_velocity, ray = self.returning_disk_ray
        authority_type = type(authority)
        original_base = transfer_module._BASE_TRANSFER_ENTRY
        original_batch = transfer_module._AUTHORITY_BATCH_ENTRY
        calls = _CodeCallCounter(original_base, original_batch)

        with calls, mock.patch.object(
            authority_type,
            "require_live",
            side_effect=AssertionError("trusted per-ray entry cannot repeat live I/O"),
        ) as live_gate:
            result = (
                transfer_module
                ._transfer_kerr_returning_radiation_finite_thickness_spectrum_trusted(
                    system["surface"],
                    system["disk"],
                    ray,
                    initial,
                    observer_velocity,
                    (3.0e14, 5.0e14),
                    termination=system["termination"],
                    ray_options=system["ray_options"],
                    surface_options=system["surface_options"],
                    escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
                    authority=authority,
                    authenticated_snapshot=snapshot,
                )
            )
        live_gate.assert_not_called()
        self.assertEqual(calls.count(original_base), 1)
        self.assertEqual(calls.count(original_batch), 1)
        self.assertEqual(result.source_kind, "finite-thickness-disk")

    def test_identity_return_kernel_recovers_discrete_nt_baseline(self) -> None:
        system = self.identity
        profile = system["profile"]
        self.assertEqual(profile.incident_returning_flux_w_m2, (0.0,))
        self.assertEqual(profile.outgoing_flux_w_m2, profile.intrinsic_flux_w_m2)
        result = self.transfer(system, self.identity_disk_ray)
        self.assertEqual(result.source_kind, "finite-thickness-disk")
        representative_radius = profile.annulus_representative_radii_over_mass[0]
        baseline_isotropic = tuple(
            system["disk"].emitted_specific_intensity_nu(
                representative_radius * system["metric"].mass_m,
                frequency,
            )
            for frequency in result.emitted_frequencies_hz
        )
        self.assertEqual(
            result.isotropic_emitted_specific_intensities_nu,
            baseline_isotropic,
        )
        baseline_emitted = tuple(
            value * result.angular_emission_multiplier
            for value in baseline_isotropic
        )
        baseline_observed = tuple(
            (
                result.frequency_shift_g
                * result.frequency_shift_g
                * result.frequency_shift_g
            )
            * value
            for value in baseline_emitted
        )
        self.assertEqual(result.emitted_specific_intensities_nu, baseline_emitted)
        self.assertEqual(result.observed_specific_intensities_nu, baseline_observed)

    def test_fout_temperature_is_consumed_once_without_fin_or_two_face_factor(
        self,
    ) -> None:
        system = self.returning
        profile = system["profile"]
        self.assertGreater(
            profile.outgoing_flux_w_m2[0],
            profile.intrinsic_flux_w_m2[0],
        )
        result = self.transfer(system, self.returning_disk_ray)
        snapshot = system["authority"].snapshot
        flux = snapshot.outgoing_flux_w_m2[0]
        temperature = snapshot.outgoing_effective_temperature_k[0]
        self.assertLessEqual(
            abs(STEFAN_BOLTZMANN_W_M2_K4 * temperature**4 - flux)
            / flux,
            2.0e-12,
        )
        expected_once = tuple(
            colour_corrected_planck_specific_intensity_nu(
                temperature,
                snapshot.colour_correction,
                frequency,
            )
            for frequency in result.emitted_frequencies_hz
        )
        self.assertEqual(
            result.isotropic_emitted_specific_intensities_nu,
            expected_once,
        )
        doubled_temperature = (2.0 * flux / STEFAN_BOLTZMANN_W_M2_K4) ** 0.25
        incorrectly_doubled = tuple(
            colour_corrected_planck_specific_intensity_nu(
                doubled_temperature,
                snapshot.colour_correction,
                frequency,
            )
            for frequency in result.emitted_frequencies_hz
        )
        self.assertNotEqual(expected_once, incorrectly_doubled)

    def test_result_revalidation_replays_once_and_tamper_fails_exactly(self) -> None:
        result = self.transfer(self.returning, self.returning_disk_ray)
        original_base = transfer_module._BASE_TRANSFER_ENTRY
        original_batch = transfer_module._AUTHORITY_BATCH_ENTRY
        calls = _CodeCallCounter(original_base, original_batch)

        with calls:
            result.revalidate()
        self.assertEqual(calls.count(original_base), 1)
        self.assertEqual(calls.count(original_batch), 1)

        changed_observed = (
            AlwaysEqualFloat(result.observed_specific_intensities_nu[0]),
            *result.observed_specific_intensities_nu[1:],
        )
        changed_base = self._forge(
            result.base_result,
            observer_frequencies_hz=(4.0e14, 5.0e14),
        )
        attacks = (
            {"observed_specific_intensities_nu": changed_observed},
            {"source_kind": AlwaysEqualStr(result.source_kind)},
            {"annulus_index": AlwaysEqualInt(result.annulus_index)},
            {"base_result": changed_base},
            {"_descriptor_sha256": AlwaysEqualStr(result.model_descriptor_sha256)},
        )
        for changes in attacks:
            with self.subTest(fields=tuple(changes)), self.assertRaises(
                KerrReturningRadiationFiniteThicknessTransferVerificationError
            ):
                self._forge(result, **changes).revalidate()

    def test_exact_input_limits_extremes_and_authority_ownership(self) -> None:
        authority_type = type(self.returning["authority"])
        invalid_frequency_grids = (
            [3.0e14],
            (),
            TupleSubclass((3.0e14,)),
            (3,),
            (AlwaysEqualFloat(3.0e14),),
            (math.inf,),
            (5.0e14, 3.0e14),
            (3.0e14, 3.0e14),
            tuple(1.0 for _ in range(MAXIMUM_FREQUENCY_BINS + 1)),
        )
        with mock.patch.object(
            authority_type,
            "require_live",
            side_effect=AssertionError("invalid inputs must fail before live I/O"),
        ) as live_gate:
            for frequencies in invalid_frequency_grids:
                with self.subTest(kind=type(frequencies).__name__), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    self.transfer(
                        self.returning,
                        self.returning_disk_ray,
                        frequencies=frequencies,
                    )
            for observer_velocity in (
                list(self.returning_disk_ray[1]),
                TupleSubclass(self.returning_disk_ray[1]),
                (
                    AlwaysEqualFloat(self.returning_disk_ray[1][0]),
                    *self.returning_disk_ray[1][1:],
                ),
            ):
                with self.subTest(velocity_type=type(observer_velocity).__name__), self.assertRaises(
                    TypeError
                ):
                    self.transfer(
                        self.returning,
                        self.returning_disk_ray,
                        observer_velocity=observer_velocity,
                    )
        live_gate.assert_not_called()

        extreme = self.transfer(
            self.returning,
            self.returning_capture_ray,
            frequencies=(float.fromhex("0x1.0p-1022"), 1.0e300),
        )
        self.assertEqual(extreme.observed_specific_intensities_nu, (0.0, 0.0))
        with self.assertRaises(
            KerrReturningRadiationFiniteThicknessTransferVerificationError
        ):
            self.transfer(
                self.returning,
                self.returning_disk_ray,
                authority=self.identity["authority"],
            )


if __name__ == "__main__":
    unittest.main()
