from __future__ import annotations

from dataclasses import fields, replace
import math
import unittest
from unittest import mock

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.geodesic import (
    ClassifiedMultiInteriorSurfaceCrossing,
    HamiltonianState,
    InteriorSurfaceDecision,
    RayTraceOptions,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
    hamiltonian_null_residual,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_bl_vector_to_ks_cartesian,
    kerr_bl_zamo_tetrad,
    kerr_ks_event_to_oblate,
    kerr_zamo_camera_ray,
)
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    PowerLawEscapedObserverSpectrum,
)
from offline.kerr_disk_transfer import transfer_kerr_disk_spectrum
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_surface import (
    LOWER_TARGET_ID,
    OPAQUE_OUTCOME,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_finite_thickness_transfer import (
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    _transfer_kerr_finite_thickness_spectrum_certified,
    transfer_kerr_finite_thickness_spectrum,
)
import offline.kerr_finite_thickness_replay_certificate as replay_module


SOLAR_MASS_KG = 1.98847e30


def _forge_frozen_dataclass(original, **changes):
    """Test-only constructor bypass for downstream trust-boundary audits."""

    forged = object.__new__(type(original))
    for name in type(original).__dataclass_fields__:
        object.__setattr__(
            forged,
            name,
            changes.get(name, getattr(original, name)),
        )
    return forged


def _solve_linear_4x4(rows, right_hand_side):
    augmented = [
        [float(value) for value in row] + [float(right_hand_side[index])]
        for index, row in enumerate(rows)
    ]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(augmented[row][column]))
        if augmented[pivot][column] == 0.0:
            raise AssertionError("test Jacobian is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][entry] - factor * augmented[column][entry]
                for entry in range(5)
            ]
    return tuple(augmented[row][4] for row in range(4))


def _flip_boyer_lindquist_radial_covector(metric, state):
    oblate = kerr_ks_event_to_oblate(metric, state.event)
    bases = tuple(
        kerr_bl_vector_to_ks_cartesian(
            tuple(float(index == component) for index in range(4)),
            mass_m=metric.mass_m,
            spin_a_m=metric.spin_a_m,
            radius_m=oblate.radius_m,
            theta_rad=oblate.theta_rad,
            phi_ks_rad=oblate.phi_ks_rad,
        )
        for component in range(4)
    )
    pulled_back = [
        math.fsum(
            state.covector[index] * bases[component][index]
            for index in range(4)
        )
        for component in range(4)
    ]
    pulled_back[1] = -pulled_back[1]
    return HamiltonianState(
        state.event,
        _solve_linear_4x4(bases, pulled_back),
    )


class OfflineKerrFiniteThicknessTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.disk = StationaryNovikovThorneDisk(
            metric=cls.metric,
            black_hole_mass_kg=1.0e8 * SOLAR_MASS_KG,
            mass_accretion_rate_kg_s=1.0e22,
        )
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

    def trace(
        self,
        *,
        theta: float,
        screen_x: float,
        screen_y: float,
        observer_radius: float = 30.0,
        surface: KerrFiniteThicknessMultiSurface | None = None,
        ray_options: RayTraceOptions | None = None,
    ):
        selected_surface = self.surface if surface is None else surface
        initial = kerr_zamo_camera_ray(
            selected_surface.metric,
            observer_radius_m=observer_radius,
            theta_rad=theta,
            screen_x=screen_x,
            screen_y=screen_y,
        )
        observer = kerr_bl_zamo_tetrad(
            selected_surface.metric,
            observer_radius_m=observer_radius,
            theta_rad=theta,
        )
        ray = trace_null_geodesic(
            selected_surface.metric,
            initial,
            termination=self.termination,
            multi_interior_surface=selected_surface,
            surface_options=self.surface_options,
            options=self.ray_options if ray_options is None else ray_options,
        )
        return initial, observer.four_velocity, ray

    def transfer(
        self,
        initial,
        observer_velocity,
        ray,
        *,
        surface: KerrFiniteThicknessMultiSurface | None = None,
        disk: StationaryNovikovThorneDisk | None = None,
        frequencies=(3.0e14, 5.0e14),
        background=None,
        termination=None,
        ray_options=None,
        surface_options=None,
        **transfer_options,
    ):
        return transfer_kerr_finite_thickness_spectrum(
            self.surface if surface is None else surface,
            self.disk if disk is None else disk,
            ray,
            initial,
            observer_velocity,
            frequencies,
            termination=(
                self.termination if termination is None else termination
            ),
            ray_options=(
                self.ray_options if ray_options is None else ray_options
            ),
            surface_options=(
                self.surface_options
                if surface_options is None
                else surface_options
            ),
            escaped_observer_spectrum=(
                DarkEscapedObserverSpectrum()
                if background is None
                else background
            ),
            **transfer_options,
        )

    def test_public_transfer_and_untrusted_result_each_replay_once(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        background = DarkEscapedObserverSpectrum()
        with mock.patch.object(
            replay_module,
            "trace_null_geodesic",
            wraps=replay_module.trace_null_geodesic,
        ) as replay:
            result = self.transfer(
                initial,
                observer_velocity,
                ray,
                background=background,
            )
            self.assertEqual(replay.call_count, 1)
            constructor_arguments = {
                field.name: getattr(result, field.name)
                for field in fields(result)
                if field.init
            }
            self.assertNotIn("_replay_certificate", constructor_arguments)
            direct = type(result)(**constructor_arguments)
            self.assertEqual(replay.call_count, 2)
            revalidated = replace(result)
            self.assertEqual(replay.call_count, 3)
            certified, _certificate = (
                _transfer_kerr_finite_thickness_spectrum_certified(
                    self.surface,
                    self.disk,
                    ray,
                    initial,
                    observer_velocity,
                    (3.0e14, 5.0e14),
                    termination=self.termination,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                    escaped_observer_spectrum=background,
                )
            )
            self.assertEqual(replay.call_count, 4)

        self.assertEqual(direct, result)
        self.assertEqual(revalidated, result)
        self.assertEqual(certified, result)
        self.assertEqual(
            direct.transfer_configuration_sha256,
            result.transfer_configuration_sha256,
        )
        self.assertEqual(
            revalidated.transfer_configuration_sha256,
            result.transfer_configuration_sha256,
        )
        self.assertEqual(
            revalidated.escape_spectrum_descriptor_sha256,
            result.escape_spectrum_descriptor_sha256,
        )
        self.assertEqual(
            certified.transfer_configuration_sha256,
            result.transfer_configuration_sha256,
        )

    def test_certified_revalidation_skips_only_geometry_replay(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        with mock.patch.object(
            replay_module,
            "trace_null_geodesic",
            wraps=replay_module.trace_null_geodesic,
        ) as replay:
            result, certificate = (
                _transfer_kerr_finite_thickness_spectrum_certified(
                    self.surface,
                    self.disk,
                    ray,
                    initial,
                    observer_velocity,
                    (3.0e14, 5.0e14),
                    termination=self.termination,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                    escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
                )
            )
            self.assertEqual(replay.call_count, 1)
            self.assertIsNotNone(result.pseudo_cylindrical_radius_over_mass)
            self.assertIsNotNone(result.frequency_shift_g)
            self.assertIsNotNone(result.photon_projection)
            mutations = (
                {"source_kind": "escaped-boundary"},
                {"terminal_surface_entry": None},
                {
                    "pseudo_cylindrical_radius_over_mass": (
                        1.001 * result.pseudo_cylindrical_radius_over_mass
                    )
                },
                {"frequency_shift_g": 1.001 * result.frequency_shift_g},
                {
                    "photon_projection": replace(
                        result.photon_projection,
                        outgoing_cosine=(
                            0.99 * result.photon_projection.outgoing_cosine
                        ),
                    )
                },
                {
                    "observed_specific_intensities_nu": tuple(
                        1.001 * value
                        for value in result.observed_specific_intensities_nu
                    )
                },
                {"transfer_configuration_sha256": "0" * 64},
            )
            for mutation in mutations:
                with self.subTest(mutation=tuple(mutation)), self.assertRaises(
                    (TypeError, ValueError),
                ):
                    replace(
                        result,
                        _replay_certificate=certificate,
                        **mutation,
                    )
                self.assertEqual(replay.call_count, 1)

    def test_scientific_status_is_explicitly_limited(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertEqual(
            SCIENTIFIC_STATUS["thermalReference"],
            (
                "equatorial Novikov-Thorne/Page-Thorne flux at matching "
                "pseudo-cylindrical rho"
            ),
        )
        self.assertIn(
            "independently caller-supplied",
            SCIENTIFIC_STATUS["heightFluxRateBinding"],
        )
        self.assertFalse(SCIENTIFIC_STATUS["includesFineCoarseWholeRayConvergence"])
        self.assertFalse(SCIENTIFIC_STATUS["isHydrostaticVerticalStructureSolution"])
        self.assertFalse(SCIENTIFIC_STATUS["isOffEquatorialGeodesicDisk"])
        self.assertFalse(SCIENTIFIC_STATUS["includesReturningRadiation"])
        self.assertFalse(SCIENTIFIC_STATUS["includesSolvedAtmosphere"])
        self.assertFalse(
            SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"]
        )
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["includesReturningRadiation"] = True

    def test_standard_disk_ray_survives_full_replay_authentication(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        self.assertEqual(ray.outcome, OPAQUE_OUTCOME)
        result = self.transfer(initial, observer_velocity, ray)
        self.assertEqual(result.source_kind, "finite-thickness-disk")
        self.assertIs(result.ray, ray)
        self.assertIs(result.ray_options, self.ray_options)
        self.assertIs(result.surface_options, self.surface_options)

    def test_real_upper_and_lower_rays_are_reflection_symmetric(self) -> None:
        results = []
        for theta, screen_y, expected_face in (
            (1.1, -0.5, UPPER),
            (math.pi - 1.1, 0.5, LOWER),
        ):
            initial, observer_velocity, ray = self.trace(
                theta=theta,
                screen_x=0.5,
                screen_y=screen_y,
            )
            result = self.transfer(initial, observer_velocity, ray)
            self.assertEqual(ray.outcome, OPAQUE_OUTCOME)
            self.assertEqual(result.source_kind, "finite-thickness-disk")
            self.assertEqual(result.face, expected_face)
            self.assertTrue(ray.multi_surface_trace.topology_converged)
            self.assertIs(
                result.terminal_surface_entry,
                ray.multi_surface_trace.crossings[-1],
            )
            self.assertGreater(result.frequency_shift_g, 0.0)
            self.assertGreater(result.photon_projection.outgoing_cosine, 0.0)
            self.assertTrue(
                all(
                    value > 0.0
                    for value in result.observed_specific_intensities_nu
                )
            )
            results.append(result)

        upper, lower = results
        self.assertAlmostEqual(
            upper.pseudo_cylindrical_radius_over_mass,
            lower.pseudo_cylindrical_radius_over_mass,
            places=12,
        )
        self.assertAlmostEqual(
            upper.frequency_shift_g,
            lower.frequency_shift_g,
            places=13,
        )
        self.assertAlmostEqual(
            upper.photon_projection.outgoing_cosine,
            lower.photon_projection.outgoing_cosine,
            places=13,
        )
        for upper_value, lower_value in zip(
            upper.observed_specific_intensities_nu,
            lower.observed_specific_intensities_nu,
        ):
            self.assertAlmostEqual(upper_value, lower_value, places=18)

    def test_g_mu_and_every_spectral_stage_recompute(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        result = self.transfer(initial, observer_velocity, ray)
        crossing = result.terminal_surface_entry.crossing
        common_scale = max(
            *(abs(value) for value in initial.covector),
            *(abs(value) for value in crossing.state.covector),
        )
        observer_frequency = math.fsum(
            observer_velocity[index] * initial.covector[index] / common_scale
            for index in range(4)
        )
        emitter_frequency = math.fsum(
            result.emitter.four_velocity[index]
            * crossing.state.covector[index]
            / common_scale
            for index in range(4)
        )
        self.assertAlmostEqual(
            result.frequency_shift_g,
            observer_frequency / emitter_frequency,
            places=14,
        )
        projection = result.emitter.project_past_directed_photon(
            crossing.state,
            backside_policy="reject",
            null_residual_limit=2.0e-7,
            event_tolerance_m=result.emitter_event_tolerance_m,
        )
        self.assertEqual(result.photon_projection, projection)

        law = FluxConservingLinearLimbDarkening()
        expected_multiplier = law.intensity_multiplier(projection.outgoing_cosine)
        self.assertAlmostEqual(
            result.angular_emission_multiplier,
            expected_multiplier,
            places=15,
        )
        for index, observer_frequency_hz in enumerate(
            result.observer_frequencies_hz
        ):
            emitted_frequency = observer_frequency_hz / result.frequency_shift_g
            isotropic = self.disk.emitted_specific_intensity_nu(
                result.equatorial_reference_radius_m,
                emitted_frequency,
            )
            emitted = isotropic * expected_multiplier
            observed = result.frequency_shift_g**3 * emitted
            self.assertAlmostEqual(
                result.emitted_frequencies_hz[index],
                emitted_frequency,
                places=4,
            )
            self.assertAlmostEqual(
                result.isotropic_emitted_specific_intensities_nu[index],
                isotropic,
                delta=2.0e-13 * isotropic,
            )
            self.assertAlmostEqual(
                result.emitted_specific_intensities_nu[index],
                emitted,
                delta=2.0e-13 * emitted,
            )
            self.assertAlmostEqual(
                result.observed_specific_intensities_nu[index],
                observed,
                delta=2.0e-13 * observed,
            )

    def test_small_height_converges_to_zero_thickness_oracle(self) -> None:
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=1.0e-6,
            outer_radius_over_mass=25.0,
        )
        surface = KerrFiniteThicknessMultiSurface(self.metric, calibration)
        initial = kerr_zamo_camera_ray(
            self.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        observer = kerr_bl_zamo_tetrad(
            self.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
        )
        thick_ray = trace_null_geodesic(
            self.metric,
            initial,
            termination=self.termination,
            multi_interior_surface=surface,
            surface_options=self.surface_options,
            options=self.ray_options,
        )
        thin_ray = trace_null_geodesic(
            self.metric,
            initial,
            termination=self.termination,
            options=self.ray_options,
        )
        thick = self.transfer(
            initial,
            observer.four_velocity,
            thick_ray,
            surface=surface,
            frequencies=(5.0e14,),
        )
        thin = transfer_kerr_disk_spectrum(
            self.disk,
            thin_ray,
            observer.four_velocity,
            (5.0e14,),
            outer_radius_m=25.0,
            escaped_observer_specific_intensity_nu=DarkEscapedObserverSpectrum(),
            surface_options=self.surface_options,
            frequency_null_residual_limit=2.0e-7,
            angular_emission_law=FluxConservingLinearLimbDarkening(),
        )
        comparisons = (
            (thick.equatorial_reference_radius_m, thin.disk_radius_m),
            (thick.frequency_shift_g, thin.frequency_shift_g),
            (
                thick.photon_projection.outgoing_cosine,
                thin.emission_angle_cosine,
            ),
            (
                thick.observed_specific_intensities_nu[0],
                thin.observed_specific_intensities_nu[0],
            ),
        )
        for actual, expected in comparisons:
            self.assertLess(
                abs(actual - expected) / max(abs(expected), 1.0e-300),
                8.0e-7,
            )

    def test_capture_is_black_and_escape_uses_closed_builtin_spectrum(self) -> None:
        background = PowerLawEscapedObserverSpectrum(
            reference_specific_intensity_nu=2.5,
            reference_frequency_hz=5.0e14,
            spectral_index=-1.0,
        )
        cases = (
            (0.2, 0.0, 0.0, "captured-boundary", (0.0, 0.0)),
            (0.4, 3.0, 3.0, "escaped-boundary", (5.0, 2.5)),
        )
        for theta, screen_x, screen_y, source_kind, expected in cases:
            initial, observer_velocity, ray = self.trace(
                theta=theta,
                screen_x=screen_x,
                screen_y=screen_y,
            )
            result = self.transfer(
                initial,
                observer_velocity,
                ray,
                frequencies=(2.5e14, 5.0e14),
                background=background,
            )
            with self.subTest(ray_outcome=ray.outcome):
                self.assertEqual(result.source_kind, source_kind)
                for actual, expected_value in zip(
                    result.observed_specific_intensities_nu,
                    expected,
                ):
                    self.assertAlmostEqual(actual, expected_value, places=12)
                self.assertIsNone(result.terminal_surface_entry)
                self.assertIsNone(result.emitter)

        class ForeignSpectrum:
            def __call__(self, _state, _frequency, _target):
                return 0.0

            def descriptor(self):
                return DarkEscapedObserverSpectrum().descriptor()

        initial, observer_velocity, ray = self.trace(
            theta=0.4,
            screen_x=3.0,
            screen_y=3.0,
        )
        with self.assertRaisesRegex(TypeError, "exact built-in"):
            self.transfer(
                initial,
                observer_velocity,
                ray,
                background=ForeignSpectrum(),
            )

    def test_boundary_outcome_target_and_worldtube_cannot_be_forged(self) -> None:
        disk_initial, disk_velocity, disk_ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        empty_disk_trace = replace(
            disk_ray.multi_surface_trace,
            crossings=(),
        )
        for outcome, target in (
            ("captured", self.termination.capture_target_id),
            ("escaped", self.termination.escape_target_id),
        ):
            forged = replace(
                disk_ray,
                outcome=outcome,
                terminal_target_id=target,
                multi_surface_trace=empty_disk_trace,
            )
            with self.subTest(outcome=outcome), self.assertRaisesRegex(
                ValueError,
                "authenticated worldtube",
            ):
                self.transfer(disk_initial, disk_velocity, forged)

        capture_initial, capture_velocity, captured = self.trace(
            theta=0.2,
            screen_x=0.0,
            screen_y=0.0,
        )
        self.assertEqual(captured.outcome, "captured")
        wrong_target = replace(captured, terminal_target_id="forged-capture")
        with self.assertRaisesRegex(ValueError, "target is not owned"):
            self.transfer(capture_initial, capture_velocity, wrong_target)
        capture_as_escape = replace(
            captured,
            outcome="escaped",
            terminal_target_id=self.termination.escape_target_id,
        )
        with self.assertRaisesRegex(ValueError, "authenticated worldtube"):
            self.transfer(capture_initial, capture_velocity, capture_as_escape)

        escape_initial, escape_velocity, escaped = self.trace(
            theta=0.4,
            screen_x=3.0,
            screen_y=3.0,
        )
        self.assertEqual(escaped.outcome, "escaped")
        escape_as_capture = replace(
            escaped,
            outcome="captured",
            terminal_target_id=self.termination.capture_target_id,
        )
        with self.assertRaisesRegex(ValueError, "authenticated worldtube"):
            self.transfer(escape_initial, escape_velocity, escape_as_capture)

        escaped_result = self.transfer(
            escape_initial,
            escape_velocity,
            escaped,
        )
        shifted_termination = replace(
            self.termination,
            escape_radius_m=self.termination.escape_radius_m + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "configuration provenance"):
            replace(escaped_result, termination=shifted_termination)

        disk_isco_radius = (
            self.calibration.isco_radius_over_mass * self.metric.mass_m
        )
        disk_outer_radius = (
            self.calibration.outer_radius_over_mass * self.metric.mass_m
        )
        swallowing_capture = replace(
            self.termination,
            capture_radius_m=disk_isco_radius,
        )
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            self.transfer(
                disk_initial,
                disk_velocity,
                disk_ray,
                termination=swallowing_capture,
            )
        premature_escape = replace(
            self.termination,
            escape_radius_m=disk_outer_radius,
        )
        with self.assertRaisesRegex(ValueError, "strictly outside"):
            self.transfer(
                disk_initial,
                disk_velocity,
                disk_ray,
                termination=premature_escape,
            )

        mislabeled_event = replace(
            self.termination,
            capture_target_id="analytic-kerr-event-horizon",
        )
        with self.assertRaisesRegex(
            ValueError,
            "event-horizon capture target must use the exact Kerr",
        ):
            self.transfer(
                disk_initial,
                disk_velocity,
                disk_ray,
                termination=mislabeled_event,
            )

        exact_horizon = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=self.termination.escape_radius_m,
            offset_m=0.0,
        )
        mislabeled_stretched = replace(
            exact_horizon,
            capture_target_id="analytic-kerr-stretched-horizon",
        )
        with self.assertRaisesRegex(
            ValueError,
            "stretched-horizon capture target must lie strictly outside",
        ):
            self.transfer(
                disk_initial,
                disk_velocity,
                disk_ray,
                termination=mislabeled_stretched,
            )

    def test_escape_worldtube_clears_maximum_oblate_photosphere_radius(
        self,
    ) -> None:
        initial, observer_velocity, near_outer_ray = self.trace(
            theta=1.1,
            screen_x=1.056,
            screen_y=-0.05,
        )
        self.assertEqual(near_outer_ray.outcome, OPAQUE_OUTCOME)
        hit = kerr_ks_event_to_oblate(
            self.metric,
            near_outer_ray.terminal_state.event,
        )
        outer_point = self.calibration.photosphere_point(
            self.calibration.outer_radius_over_mass,
            UPPER,
        )
        self.assertGreater(hit.radius_m, 25.01)
        self.assertLessEqual(
            hit.radius_m,
            outer_point.radius_over_mass * self.metric.mass_m,
        )
        premature = replace(self.termination, escape_radius_m=25.01)
        with self.assertRaisesRegex(ValueError, "maximum.*oblate radius"):
            self.transfer(
                initial,
                observer_velocity,
                near_outer_ray,
                termination=premature,
            )

    def test_transfer_tolerance_policy_is_bounded_and_hashed(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        excessive = (
            {"boundary_value_tolerance_m": 100.0 * self.metric.mass_m},
            {"recorded_path_absolute_tolerance": 1.0},
            {"recorded_path_relative_tolerance": 1.0},
            {"surface_value_tolerance": 1.0},
            {"conserved_quantity_tolerance": 1.0},
            {"null_residual_limit": 1.0},
            {"emitter_event_tolerance_m": self.metric.mass_m},
        )
        for options in excessive:
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError,
                "policy maximum",
            ):
                self.transfer(initial, observer_velocity, ray, **options)

        excessive_trace_options = (
            {
                "ray_options": replace(
                    self.ray_options,
                    event_value_tolerance=100.0 * self.metric.mass_m,
                )
            },
            {
                "ray_options": replace(
                    self.ray_options,
                    maximum_step=100.0 * self.metric.mass_m,
                )
            },
            {
                "surface_options": replace(
                    self.surface_options,
                    surface_value_tolerance=100.0,
                )
            },
            {
                "surface_options": replace(
                    self.surface_options,
                    subdivisions_per_segment=130,
                )
            },
        )
        for options in excessive_trace_options:
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError,
                "trace replay policy maximum",
            ):
                self.transfer(initial, observer_velocity, ray, **options)

        result = self.transfer(initial, observer_velocity, ray)
        with self.assertRaisesRegex(ValueError, "configuration provenance"):
            replace(result, null_residual_limit=3.0e-7)
        with self.assertRaisesRegex(ValueError, "configuration provenance"):
            replace(
                result,
                ray_options=replace(self.ray_options, initial_step=0.04),
            )

    def test_boundary_terminal_radial_momentum_branch_is_reintegrated(
        self,
    ) -> None:
        for theta, screen_x, screen_y, expected_outcome in (
            (0.2, 0.0, 0.0, "captured"),
            (0.4, 3.0, 3.0, "escaped"),
        ):
            initial, observer_velocity, ray = self.trace(
                theta=theta,
                screen_x=screen_x,
                screen_y=screen_y,
            )
            self.assertEqual(ray.outcome, expected_outcome)
            flipped_state = _flip_boyer_lindquist_radial_covector(
                self.metric,
                ray.terminal_state,
            )
            flipped_residual = hamiltonian_null_residual(
                self.metric,
                flipped_state,
            )
            self.assertLess(flipped_residual, 2.0e-7)
            flipped_last = replace(ray.segments[-1], end=flipped_state)
            forged_ray = replace(
                ray,
                terminal_state=flipped_state,
                maximum_null_residual=max(
                    ray.maximum_null_residual,
                    2.0 * flipped_residual,
                ),
                segments=ray.segments[:-1] + (flipped_last,),
            )
            with self.subTest(outcome=expected_outcome), self.assertRaisesRegex(
                ValueError,
                "terminal state is not bound",
            ):
                self.transfer(initial, observer_velocity, forged_ray)

    def test_full_replay_rejects_deleted_transparent_crossing(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=0.2,
            screen_x=2.0,
            screen_y=0.0,
        )
        self.assertEqual(ray.outcome, "escaped")
        self.assertGreater(len(ray.multi_surface_trace.crossings), 0)
        self.assertTrue(
            all(
                not entry.decision.terminates
                for entry in ray.multi_surface_trace.crossings
            )
        )
        deleted_trace = replace(ray.multi_surface_trace, crossings=())
        forged_ray = replace(ray, multi_surface_trace=deleted_trace)
        with self.assertRaisesRegex(ValueError, "deterministic first-visible replay"):
            self.transfer(initial, observer_velocity, forged_ray)

    def test_observer_inside_disk_is_rejected_before_backside_transfer(self) -> None:
        inside_options = replace(
            self.ray_options,
            initial_step=0.02,
            maximum_step=0.15,
        )
        initial, observer_velocity, ray = self.trace(
            theta=0.5 * math.pi,
            screen_x=0.0,
            screen_y=2.0,
            observer_radius=10.0,
            ray_options=inside_options,
        )
        terminal = ray.multi_surface_trace.crossings[-1]
        oblate = kerr_ks_event_to_oblate(self.metric, terminal.crossing.state.event)
        rho = oblate.radius_m * math.sin(oblate.theta_rad) / self.metric.mass_m
        emitter = KerrFiniteThicknessFaceEmitter(
            metric=self.metric,
            calibration=self.calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=UPPER,
            phi_ks_rad=oblate.phi_ks_rad,
            coordinate_time_m=oblate.coordinate_time_m,
        )
        classified = emitter.project_past_directed_photon(
            terminal.crossing.state,
            backside_policy="classify",
            null_residual_limit=2.0e-7,
            event_tolerance_m=1.0e-8,
        )
        self.assertEqual(classified.face_classification, "backside")
        self.assertLess(classified.outgoing_cosine, 0.0)
        with self.assertRaisesRegex(ValueError, "observer lies"):
            self.transfer(initial, observer_velocity, ray)

        capture_initial, capture_velocity, captured = self.trace(
            theta=0.5 * math.pi,
            screen_x=0.0,
            screen_y=0.0,
            observer_radius=10.0,
            ray_options=inside_options,
        )
        self.assertEqual(captured.outcome, "captured")
        self.assertEqual(captured.multi_surface_trace.crossings, ())
        with self.assertRaisesRegex(ValueError, "observer lies"):
            self.transfer(capture_initial, capture_velocity, captured)

    def test_foreign_metric_face_and_stored_spectral_fields_fail_closed(
        self,
    ) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        result = self.transfer(initial, observer_velocity, ray)

        equal_but_foreign_metric = KerrKerrSchildMetric(spin_a_m=0.7)
        foreign_disk = StationaryNovikovThorneDisk(
            metric=equal_but_foreign_metric,
            black_hole_mass_kg=self.disk.black_hole_mass_kg,
            mass_accretion_rate_kg_s=self.disk.mass_accretion_rate_kg_s,
        )
        with self.assertRaisesRegex(ValueError, "same metric"):
            self.transfer(
                initial,
                observer_velocity,
                ray,
                disk=foreign_disk,
            )

        foreign_face = KerrFiniteThicknessFaceEmitter(
            metric=self.metric,
            calibration=self.calibration,
            pseudo_cylindrical_radius_over_mass=(
                result.pseudo_cylindrical_radius_over_mass
            ),
            face=LOWER,
            phi_ks_rad=result.emitter.phi_ks_rad,
            coordinate_time_m=result.emitter.coordinate_time_m,
        )
        mutations = (
            {"emitter": foreign_face},
            {
                "pseudo_cylindrical_radius_over_mass": (
                    result.pseudo_cylindrical_radius_over_mass * 1.001
                )
            },
            {
                "equatorial_reference_radius_m": (
                    result.equatorial_reference_radius_m * 1.001
                )
            },
            {"frequency_shift_g": result.frequency_shift_g * 1.001},
            {
                "photon_projection": replace(
                    result.photon_projection,
                    outgoing_cosine=(
                        0.99 * result.photon_projection.outgoing_cosine
                    ),
                )
            },
            {
                "isotropic_emitted_specific_intensities_nu": tuple(
                    1.001 * value
                    for value in result.isotropic_emitted_specific_intensities_nu
                )
            },
            {
                "observed_specific_intensities_nu": tuple(
                    1.001 * value
                    for value in result.observed_specific_intensities_nu
                )
            },
            {"observed_specific_intensities_nu": (math.nan, math.nan)},
        )
        for mutation in mutations:
            with self.subTest(mutation=tuple(mutation)), self.assertRaises(
                (TypeError, ValueError)
            ):
                replace(result, **mutation)

    def test_unconverged_or_reclassified_trace_and_bad_inputs_fail_closed(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        trace = ray.multi_surface_trace
        unconverged_ray = replace(
            ray,
            multi_surface_trace=replace(trace, topology_converged=False),
        )
        with self.assertRaisesRegex(ValueError, "not converged"):
            self.transfer(initial, observer_velocity, unconverged_ray)

        terminal = trace.crossings[-1]
        reclassified = replace(
            terminal,
            decision=InteriorSurfaceDecision(
                "opaque-lower-photosphere",
                OPAQUE_OUTCOME,
                LOWER_TARGET_ID,
            ),
        )
        reclassified_ray = replace(
            ray,
            multi_surface_trace=replace(trace, crossings=(reclassified,)),
        )
        with self.assertRaisesRegex(ValueError, "classification"):
            self.transfer(initial, observer_velocity, reclassified_ray)

        no_trace = replace(ray, multi_surface_trace=None)
        with self.assertRaisesRegex(ValueError, "MultiInteriorSurfaceTrace"):
            self.transfer(initial, observer_velocity, no_trace)
        failed = replace(ray, failure_reason="synthetic failure")
        with self.assertRaisesRegex(ValueError, "failed ray"):
            self.transfer(initial, observer_velocity, failed)
        with self.assertRaises(ValueError):
            self.transfer(
                initial,
                observer_velocity,
                ray,
                frequencies=(math.nan,),
            )

    def test_forged_multi_surface_trace_header_is_revalidated(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        trace = ray.multi_surface_trace
        cases = (
            ({"base_subdivisions_per_step": 3}, "base subdivisions"),
            ({"verification_subdivisions_per_step": 7}, "verification"),
            ({"topology_converged": 1}, "exact bool"),
            ({"maximum_probe_event_difference": math.nan}, "probe convergence"),
            (
                {"maximum_probe_covector_relative_difference": -1.0},
                "probe convergence",
            ),
            ({"probe_reintegrations": -7}, "work diagnostics"),
            ({"surface_value_evaluations": True}, "work diagnostics"),
            (
                {"surface_ids": tuple(reversed(trace.surface_ids))},
                "trace ids",
            ),
        )
        for changes, message in cases:
            forged_trace = _forge_frozen_dataclass(trace, **changes)
            forged_ray = replace(ray, multi_surface_trace=forged_trace)
            with self.subTest(changes=changes), self.assertRaisesRegex(
                (TypeError, ValueError),
                message,
            ):
                self.transfer(initial, observer_velocity, forged_ray)

    def test_forged_negative_crossing_indices_and_affines_fail_before_indexing(
        self,
    ) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        trace = ray.multi_surface_trace
        terminal = trace.crossings[-1]

        def forged_crossing(**changes):
            # Deliberately bypass the frozen dataclass constructor to prove
            # that the transfer trust boundary repeats the complete ABI.
            forged = object.__new__(RecordedSurfaceCrossing)
            for name in RecordedSurfaceCrossing.__dataclass_fields__:
                object.__setattr__(
                    forged,
                    name,
                    changes.get(name, getattr(terminal.crossing, name)),
                )
            return forged

        cases = (
            ({"segment_index": -1}, "segment index"),
            ({"segment_affine_length": -1.0e-9}, "affine diagnostics"),
            ({"ray_affine_length": -1.0e-9}, "affine diagnostics"),
            ({"orientation": True}, "orientation"),
            ({"bracket_affine_width": -1.0e-9}, "root diagnostics"),
            ({"iterations": -1}, "root diagnostics"),
        )
        for changes, message in cases:
            forged_entry = ClassifiedMultiInteriorSurfaceCrossing(
                surface_id=terminal.surface_id,
                crossing=forged_crossing(**changes),
                decision=terminal.decision,
            )
            forged_trace = replace(trace, crossings=(forged_entry,))
            forged_ray = replace(ray, multi_surface_trace=forged_trace)
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                self.transfer(initial, observer_velocity, forged_ray)

    def test_trace_order_terminal_prefix_and_segment_provenance_are_replayed(
        self,
    ) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        trace = ray.multi_surface_trace
        terminal = trace.crossings[-1]

        transparent = InteriorSurfaceDecision("forged-transparent")
        first_transparent = ClassifiedMultiInteriorSurfaceCrossing(
            surface_id=terminal.surface_id,
            crossing=terminal.crossing,
            decision=transparent,
        )
        duplicate_transparent = ClassifiedMultiInteriorSurfaceCrossing(
            surface_id=terminal.surface_id,
            crossing=terminal.crossing,
            decision=transparent,
        )
        unordered_trace = _forge_frozen_dataclass(
            trace,
            crossings=(first_transparent, duplicate_transparent),
        )
        unordered_ray = replace(ray, multi_surface_trace=unordered_trace)
        with self.assertRaisesRegex(ValueError, "strictly observer-to-source"):
            self.transfer(initial, observer_velocity, unordered_ray)

        delta = min(
            1.0e-6,
            0.5 * terminal.crossing.segment_affine_length,
            0.5 * terminal.crossing.ray_affine_length,
        )
        self.assertGreater(delta, 0.0)
        earlier_crossing = _forge_frozen_dataclass(
            terminal.crossing,
            segment_affine_length=(
                terminal.crossing.segment_affine_length - delta
            ),
            ray_affine_length=terminal.crossing.ray_affine_length - delta,
        )
        terminal_before_real = ClassifiedMultiInteriorSurfaceCrossing(
            surface_id=terminal.surface_id,
            crossing=earlier_crossing,
            decision=terminal.decision,
        )
        terminal_prefix_trace = _forge_frozen_dataclass(
            trace,
            crossings=(terminal_before_real, terminal),
        )
        terminal_prefix_ray = replace(
            ray,
            multi_surface_trace=terminal_prefix_trace,
        )
        with self.assertRaisesRegex(ValueError, "only the final"):
            self.transfer(initial, observer_velocity, terminal_prefix_ray)

        claimed_segment = ray.segments[0]
        claimed_offset = 0.5 * claimed_segment.affine_length
        misplaced_crossing = RecordedSurfaceCrossing(
            state=terminal.crossing.state,
            ray_affine_length=claimed_offset,
            segment_index=0,
            segment_affine_length=claimed_offset,
            orientation=terminal.crossing.orientation,
            surface_value=terminal.crossing.surface_value,
            bracket_affine_width=terminal.crossing.bracket_affine_width,
            iterations=terminal.crossing.iterations,
        )
        misplaced_entry = ClassifiedMultiInteriorSurfaceCrossing(
            surface_id=terminal.surface_id,
            crossing=misplaced_crossing,
            decision=terminal.decision,
        )
        misplaced_trace = replace(trace, crossings=(misplaced_entry,))
        misplaced_ray = replace(ray, multi_surface_trace=misplaced_trace)
        with self.assertRaisesRegex(ValueError, "claimed recorded segment"):
            self.transfer(initial, observer_velocity, misplaced_ray)

    def test_terminal_radial_momentum_branch_is_reintegrated(self) -> None:
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        terminal = ray.multi_surface_trace.crossings[-1]
        flipped_state = _flip_boyer_lindquist_radial_covector(
            self.metric,
            terminal.crossing.state,
        )
        self.assertNotEqual(flipped_state.covector, terminal.crossing.state.covector)
        flipped_residual = hamiltonian_null_residual(self.metric, flipped_state)
        self.assertLess(flipped_residual, 2.0e-7)

        flipped_crossing = replace(
            terminal.crossing,
            state=flipped_state,
        )
        flipped_entry = replace(terminal, crossing=flipped_crossing)
        flipped_trace = replace(
            ray.multi_surface_trace,
            crossings=(flipped_entry,),
        )
        flipped_last_segment = replace(
            ray.segments[-1],
            end=flipped_state,
        )
        flipped_ray = replace(
            ray,
            terminal_state=flipped_state,
            maximum_null_residual=max(
                ray.maximum_null_residual,
                2.0 * flipped_residual,
            ),
            segments=ray.segments[:-1] + (flipped_last_segment,),
            multi_surface_trace=flipped_trace,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"terminal state.*Hamiltonian reintegration",
        ):
            self.transfer(initial, observer_velocity, flipped_ray)

    def test_calibration_subclass_cannot_impersonate_official_provenance(
        self,
    ) -> None:
        class ForgedCalibration(StationaryKerrFiniteThicknessCalibration):
            def photosphere_height_over_mass(self, value):
                return 0.5 * super().photosphere_height_over_mass(value)

        forged_calibration = ForgedCalibration(
            dimensionless_spin=self.calibration.dimensionless_spin,
            eddington_scaled_mass_accretion_rate=(
                self.calibration.eddington_scaled_mass_accretion_rate
            ),
            orientation=self.calibration.orientation,
            outer_radius_over_mass=self.calibration.outer_radius_over_mass,
            thinness_gate_maximum_h_over_rho=(
                self.calibration.thinness_gate_maximum_h_over_rho
            ),
        )
        field_names = tuple(
            StationaryKerrFiniteThicknessCalibration.__dataclass_fields__
        )
        self.assertEqual(
            tuple(getattr(forged_calibration, name) for name in field_names),
            tuple(getattr(self.calibration, name) for name in field_names),
        )
        self.assertIsNot(type(forged_calibration), type(self.calibration))
        forged_surface = KerrFiniteThicknessMultiSurface(
            self.metric,
            forged_calibration,
        )
        initial, observer_velocity, ray = self.trace(
            theta=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        with self.assertRaisesRegex(TypeError, "exact built-in"):
            self.transfer(
                initial,
                observer_velocity,
                ray,
                surface=forged_surface,
            )

        official = self.transfer(initial, observer_velocity, ray)
        with self.assertRaisesRegex(TypeError, "exact built-in"):
            replace(official, surface=forged_surface)


if __name__ == "__main__":
    unittest.main()
