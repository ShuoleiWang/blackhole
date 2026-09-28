from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric
from offline.kerr import KerrOblateTermination
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.kerr_returning_radiation_thermal_profile import (
    CACHED_KERR_FORWARD_CLASSIFICATION,
    EXTERNAL_UNVERIFIED_CLASSIFICATION,
    MAXIMUM_RELATIVE_EQUATION_RESIDUAL,
    REPLAYED_KERR_FORWARD_CLASSIFICATION,
    SCIENTIFIC_STATUS,
    AxisymmetricReturningRadiationThermalProfile,
    KerrReturningRadiationThermalProfileError,
    KerrReturningRadiationThermalProfileVerificationError,
    solve_axisymmetric_returning_radiation_thermal_profile,
    solve_certified_kerr_returning_radiation_thermal_profile,
    verify_axisymmetric_returning_radiation_thermal_profile,
    verify_axisymmetric_returning_radiation_thermal_profile_projection,
)
import offline.kerr_returning_radiation_kernel as kernel_module
import offline.kerr_returning_radiation_kernel_cached as cached_module
import offline.kerr_returning_radiation_thermal_profile as profile_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
from offline.novikov_thorne import PROGRADE, RETROGRADE, kerr_isco_radius_m
from offline.returning_radiation import (
    AxisymmetricReturningRadiationKernel,
    ReturningRadiationConvergenceError,
    ReturningRadiationFixedPointPolicy,
)


class _FloatSubclass(float):
    pass


class _DiskSubclass(StationaryNovikovThorneDisk):
    pass


class _KernelSubclass(AxisymmetricReturningRadiationKernel):
    pass


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _low_return_forward_classifier(*arguments):
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    receiver_face = UPPER if tangent_azimuth < math.pi else LOWER
    frequency_ratio = 0.2
    return kernel_module._DirectionTransport(
        "return-upper" if receiver_face == UPPER else "return-lower",
        receiver_face,
        source_radius,
        frequency_ratio,
        frequency_ratio * frequency_ratio,
        _identity_for(*arguments[6:]),
        receiver_face,
        source_radius,
    )


class ReturningRadiationThermalProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = KerrKerrSchildMetric(mass_m=1.0, spin_a_m=0.5)
        self.disk = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=PROGRADE,
            colour_correction=1.7,
        )
        self.isco = kerr_isco_radius_m(0.5, PROGRADE)
        self.edges = (self.isco, 8.0, 12.0)
        self.radii = (
            0.5 * math.fsum((self.edges[0], self.edges[1])),
            10.0,
        )
        self.areas = (100.0, 200.0)

    def kernel(
        self, coefficients: tuple[tuple[float, ...], ...]
    ) -> AxisymmetricReturningRadiationKernel:
        return AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=self.radii,
            receiver_emitter_coefficients=coefficients,
            ray_kernel_producer_id="unit-test-axisymmetric-kernel/v1",
        )

    def solve(
        self,
        coefficients: tuple[tuple[float, ...], ...],
        *,
        policy: ReturningRadiationFixedPointPolicy | None = None,
    ) -> AxisymmetricReturningRadiationThermalProfile:
        arguments = {
            "annulus_edges_over_mass": self.edges,
            "annulus_proper_areas_over_mass_squared": self.areas,
            "disk": self.disk,
        }
        if policy is not None:
            arguments["policy"] = policy
        return solve_axisymmetric_returning_radiation_thermal_profile(
            self.kernel(coefficients),
            **arguments,
        )

    def test_scientific_scope_and_one_face_semantics_are_explicit(self) -> None:
        self.assertIn("two-tier", SCIENTIFIC_STATUS["classification"])
        self.assertEqual(
            SCIENTIFIC_STATUS["externalEntryClassification"],
            EXTERNAL_UNVERIFIED_CLASSIFICATION,
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["certifiedEntryClassification"],
            REPLAYED_KERR_FORWARD_CLASSIFICATION,
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["equation"],
            "F_out=F_0+F_in; F_in=K F_out",
        )
        self.assertEqual(SCIENTIFIC_STATUS["absorptionAlbedo"], 0.0)
        self.assertTrue(SCIENTIFIC_STATUS["fixedColourCorrection"])
        self.assertFalse(SCIENTIFIC_STATUS["changesRenderedFrames"])
        self.assertFalse(
            SCIENTIFIC_STATUS["externalEntryPublishesProvenanceCertifiedWatts"]
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["maximumRelativeEquationResidual"],
            MAXIMUM_RELATIVE_EQUATION_RESIDUAL,
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "independentlyVerifiesDeclaredProperAreaProvenance"
            ]
        )
        for key in (
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesScattering",
            "includesSolvedAtmosphere",
            "includesPolarization",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["includesScattering"] = True

    def test_zero_kernel_recovers_nt_flux_and_unit_temperature_ratio(self) -> None:
        profile = self.solve(((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(profile.incident_returning_flux_w_m2, (0.0, 0.0))
        self.assertEqual(profile.outgoing_flux_w_m2, profile.intrinsic_flux_w_m2)
        self.assertEqual(profile.effective_temperature_ratio, (1.0, 1.0))
        self.assertEqual(
            profile.outgoing_effective_temperature_k,
            profile.intrinsic_effective_temperature_k,
        )
        self.assertFalse(profile.geometry_provenance_certified)
        self.assertEqual(
            profile.provenance_classification,
            EXTERNAL_UNVERIFIED_CLASSIFICATION,
        )
        self.assertIsNone(profile.one_face_intrinsic_power_w)
        self.assertIsNone(profile.one_face_incident_returning_power_w)
        self.assertIsNone(profile.one_face_outgoing_power_w)
        self.assertIsNone(profile.two_face_outgoing_power_w)
        self.assertLessEqual(
            profile.maximum_relative_equation_residual,
            profile.maximum_allowed_relative_equation_residual,
        )
        profile.revalidate()

    def test_profile_runs_fixed_point_solver_once(self) -> None:
        original = profile_module.solve_absorbed_returning_radiation
        with patch.object(
            profile_module,
            "solve_absorbed_returning_radiation",
            wraps=original,
        ) as solver:
            profile = self.solve(((0.05, 0.01), (0.02, 0.04)))
        self.assertEqual(solver.call_count, 1)
        profile.revalidate()

    def test_two_by_two_matches_analytic_inverse_for_actual_nt_source(self) -> None:
        coefficients = ((0.1, 0.2), (0.05, 0.1))
        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=1024,
            absolute_residual_tolerance=0.0,
            relative_residual_tolerance=1.0e-14,
        )
        profile = self.solve(coefficients, policy=policy)
        first, second = profile.intrinsic_flux_w_m2
        determinant = (1.0 - 0.1) * (1.0 - 0.1) - 0.2 * 0.05
        expected = (
            ((1.0 - 0.1) * first + 0.2 * second) / determinant,
            (0.05 * first + (1.0 - 0.1) * second) / determinant,
        )
        for actual, analytic in zip(profile.outgoing_flux_w_m2, expected):
            self.assertTrue(math.isclose(actual, analytic, rel_tol=3.0e-14))
        for f0, fout, ratio in zip(
            profile.intrinsic_flux_w_m2,
            profile.outgoing_flux_w_m2,
            profile.effective_temperature_ratio,
        ):
            self.assertTrue(
                math.isclose(ratio, (fout / f0) ** 0.25, rel_tol=3.0e-15)
            )

    def test_local_energy_balance_is_certified_but_external_areas_are_not(self) -> None:
        profile = self.solve(((0.12, 0.03), (0.04, 0.08)))
        for f0, fin, fout in zip(
            profile.intrinsic_flux_w_m2,
            profile.incident_returning_flux_w_m2,
            profile.outgoing_flux_w_m2,
        ):
            self.assertTrue(math.isclose(fout, f0 + fin, rel_tol=1.1e-12))
        self.assertIsNone(profile.one_face_power_balance_residual_w)
        self.assertIsNone(profile.one_face_power_balance_tolerance_w)
        self.assertIsNone(profile.model_descriptor()["result"]["oneFacePowerW"])

    def test_descriptor_binds_grid_kernel_nt_and_colour_assumptions(self) -> None:
        profile = self.solve(((0.05, 0.01), (0.02, 0.07)))
        descriptor = profile.model_descriptor()
        self.assertEqual(
            descriptor["classification"], EXTERNAL_UNVERIFIED_CLASSIFICATION
        )
        self.assertFalse(
            descriptor["binding"]["provenance"][
                "sameCodeSourceReplayVerified"
            ]
        )
        self.assertEqual(descriptor["annuli"]["edgesOverMass"], list(self.edges))
        self.assertEqual(
            descriptor["annuli"]["properAreasOverMassSquared"],
            list(self.areas),
        )
        binding = descriptor["binding"]
        self.assertEqual(binding["kernelDescriptorSha256"], profile.kernel_descriptor_sha256)
        disk = binding["novikovThorneDisk"]
        self.assertEqual(disk["massAccretionRateKgS"], 1.0e18)
        self.assertEqual(disk["orientation"], PROGRADE)
        self.assertEqual(disk["colourCorrection"], 1.7)
        self.assertEqual(disk["fluxFaceSemantics"], "one disk face")
        self.assertTrue(descriptor["assumptions"]["colourCorrectionHeldFixed"])
        self.assertFalse(
            descriptor["assumptions"]["includesReturningRadiationStressWorkFS"]
        )
        self.assertEqual(
            descriptor["result"]["maximumAllowedRelativeEquationResidual"],
            MAXIMUM_RELATIVE_EQUATION_RESIDUAL,
        )
        self.assertEqual(len(profile.model_descriptor_sha256), 64)
        self.assertEqual(
            json.loads(json.dumps(descriptor, sort_keys=True)), descriptor
        )

    def test_piecewise_constant_query_has_strict_radial_domain(self) -> None:
        profile = self.solve(((0.02, 0.0), (0.0, 0.03)))
        first = profile.at_radius_over_mass(self.isco)
        self.assertEqual(first.annulus_index, 0)
        self.assertEqual(
            profile.at_radius_over_mass(8.0).annulus_index,
            1,
        )
        self.assertEqual(
            profile.at_radius_over_mass(12.0).annulus_index,
            1,
        )
        self.assertEqual(first, profile.annulus(0))
        with self.assertRaisesRegex(ValueError, "inside the ISCO"):
            profile.at_radius_over_mass(self.isco - 1.0e-6)
        with self.assertRaisesRegex(ValueError, "outside"):
            profile.at_radius_over_mass(12.000001)
        with self.assertRaises(TypeError):
            profile.at_radius_over_mass(_FloatSubclass(7.0))
        with self.assertRaises(TypeError):
            profile.annulus(True)
        with self.assertRaises(IndexError):
            profile.annulus(2)

    def test_annulus_grid_must_cover_isco_and_contain_representatives(self) -> None:
        kernel = self.kernel(((0.0, 0.0), (0.0, 0.0)))
        invalid = (
            ((self.isco + 0.1, 8.0, 12.0), self.areas),
            ((self.isco - 0.1, 8.0, 12.0), self.areas),
            ((self.isco, self.radii[0], 12.0), self.areas),
            (self.edges, (100.0,)),
            (self.edges, (100.0, 0.0)),
        )
        for edges, areas in invalid:
            with self.subTest(edges=edges, areas=areas):
                with self.assertRaises((TypeError, ValueError)):
                    solve_axisymmetric_returning_radiation_thermal_profile(
                        kernel,
                        annulus_edges_over_mass=edges,
                        annulus_proper_areas_over_mass_squared=areas,
                        disk=self.disk,
                    )

    def test_zero_accretion_rate_is_rejected_because_ratio_is_undefined(self) -> None:
        disk = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=0.0,
        )
        with self.assertRaisesRegex(ValueError, "mass_accretion_rate"):
            solve_axisymmetric_returning_radiation_thermal_profile(
                self.kernel(((0.0, 0.0), (0.0, 0.0))),
                annulus_edges_over_mass=self.edges,
                annulus_proper_areas_over_mass_squared=self.areas,
                disk=disk,
            )

    def test_supercritical_kernel_fails_closed_through_shared_solver(self) -> None:
        policy = ReturningRadiationFixedPointPolicy(maximum_iterations=12)
        with self.assertRaises(ReturningRadiationConvergenceError):
            self.solve(((1.0, 0.0), (0.0, 0.0)), policy=policy)

    def test_loose_solver_policy_cannot_relax_final_relative_residual_gate(self) -> None:
        kernel = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(self.radii[0],),
            receiver_emitter_coefficients=((0.9,),),
            ray_kernel_producer_id="loose-policy-regression",
        )
        policy = ReturningRadiationFixedPointPolicy(
            maximum_iterations=1,
            absolute_residual_tolerance=0.0,
            relative_residual_tolerance=1.0,
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationThermalProfileError,
            "non-relaxable.*relative residual",
        ):
            solve_axisymmetric_returning_radiation_thermal_profile(
                kernel,
                annulus_edges_over_mass=(self.isco, 8.0),
                annulus_proper_areas_over_mass_squared=(100.0,),
                disk=self.disk,
                policy=policy,
            )

    def test_area_projection_is_conservative_and_not_claimed_as_resolve(self) -> None:
        profile = self.solve(((0.1, 0.02), (0.03, 0.1)))
        projection = profile.coarsen_annuli((self.isco, 12.0))
        self.assertEqual(projection.annulus_count, 1)
        self.assertEqual(projection.annulus_proper_areas_over_mass_squared, (300.0,))
        for fine, coarse in (
            (profile.intrinsic_flux_w_m2, projection.intrinsic_flux_w_m2),
            (
                profile.incident_returning_flux_w_m2,
                projection.incident_returning_flux_w_m2,
            ),
            (profile.outgoing_flux_w_m2, projection.outgoing_flux_w_m2),
        ):
            self.assertTrue(
                math.isclose(
                    math.fsum(area * value for area, value in zip(self.areas, fine)),
                    300.0 * coarse[0],
                    rel_tol=2.0e-16,
                )
            )
        self.assertIs(projection.is_resolved_coarse_fixed_point, False)
        self.assertFalse(projection.model_descriptor()["isResolvedCoarseFixedPoint"])
        self.assertEqual(
            projection.at_radius_over_mass(12.0).annulus_index,
            0,
        )
        projection.revalidate()
        verify_axisymmetric_returning_radiation_thermal_profile_projection(
            projection
        )
        for bad_edges in (
            (self.isco, 11.0, 12.0),
            (self.isco, 8.0),
            [self.isco, 12.0],
        ):
            with self.subTest(bad_edges=bad_edges):
                with self.assertRaises((TypeError, ValueError)):
                    profile.coarsen_annuli(bad_edges)

    def test_profile_and_projection_tampering_is_rejected(self) -> None:
        profile = self.solve(((0.08, 0.01), (0.02, 0.09)))
        object.__setattr__(
            profile,
            "outgoing_flux_w_m2",
            (profile.outgoing_flux_w_m2[0] * 1.000001, profile.outgoing_flux_w_m2[1]),
        )
        with self.assertRaises(KerrReturningRadiationThermalProfileVerificationError):
            verify_axisymmetric_returning_radiation_thermal_profile(profile)

        wrong_type = self.solve(((0.08, 0.01), (0.02, 0.09)))
        object.__setattr__(
            wrong_type,
            "intrinsic_flux_w_m2",
            (
                _FloatSubclass(wrong_type.intrinsic_flux_w_m2[0]),
                wrong_type.intrinsic_flux_w_m2[1],
            ),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationThermalProfileVerificationError,
            "non-exact type",
        ):
            wrong_type.revalidate()

        for name, forged_value in (
            ("annulus_edges_over_mass", (self.isco, 8.0, 999.0)),
            ("annulus_proper_areas_over_mass_squared", (1.0e300, 1.0)),
        ):
            external_geometry = self.solve(((0.08, 0.01), (0.02, 0.09)))
            object.__setattr__(external_geometry, name, forged_value)
            with self.subTest(name=name), self.assertRaises(
                KerrReturningRadiationThermalProfileVerificationError
            ):
                external_geometry.revalidate()

        clean = self.solve(((0.08, 0.01), (0.02, 0.09)))
        projection = clean.coarsen_annuli((self.isco, 12.0))
        object.__setattr__(projection, "source_profile_descriptor_sha256", "0" * 64)
        with self.assertRaises(KerrReturningRadiationThermalProfileVerificationError):
            projection.revalidate()

    def test_tampered_kernel_descriptor_is_rejected_before_solving(self) -> None:
        kernel = self.kernel(((0.01, 0.0), (0.0, 0.01)))
        object.__setattr__(
            kernel,
            "receiver_emitter_coefficients",
            ((0.5, 0.0), (0.0, 0.5)),
        )
        with self.assertRaises(KerrReturningRadiationThermalProfileVerificationError):
            solve_axisymmetric_returning_radiation_thermal_profile(
                kernel,
                annulus_edges_over_mass=self.edges,
                annulus_proper_areas_over_mass_squared=self.areas,
                disk=self.disk,
            )

    def test_arbitrary_areas_and_forged_producer_remain_external_unverified(self) -> None:
        forged = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=self.radii,
            receiver_emitter_coefficients=((0.0, 0.0), (0.0, 0.0)),
            ray_kernel_producer_id=(
                "kerr-finite-thickness-forward-finite-volume-returning-"
                "energy-kernel/v1:" + "a" * 64
            ),
        )
        first = solve_axisymmetric_returning_radiation_thermal_profile(
            forged,
            annulus_edges_over_mass=self.edges,
            annulus_proper_areas_over_mass_squared=(1.0, 1.0),
            disk=self.disk,
        )
        second = solve_axisymmetric_returning_radiation_thermal_profile(
            forged,
            annulus_edges_over_mass=self.edges,
            annulus_proper_areas_over_mass_squared=(1.0e30, 1.0e-20),
            disk=self.disk,
        )
        for profile in (first, second):
            self.assertFalse(profile.geometry_provenance_certified)
            self.assertIsNone(profile.one_face_outgoing_power_w)
            self.assertFalse(
                profile.model_descriptor()["binding"]["provenance"][
                    "geometryProvenanceCertified"
                ]
            )
        with self.assertRaisesRegex(TypeError, "producer string cannot certify"):
            solve_certified_kerr_returning_radiation_thermal_profile(
                forged,
                disk=self.disk,
            )

    def test_exact_type_boundaries_reject_subclasses_lists_bools_and_ints(self) -> None:
        kernel = self.kernel(((0.01, 0.0), (0.0, 0.01)))
        invalid_arguments = (
            {"annulus_edges_over_mass": list(self.edges)},
            {
                "annulus_edges_over_mass": (
                    _FloatSubclass(self.isco),
                    8.0,
                    12.0,
                )
            },
            {"annulus_edges_over_mass": (self.isco, 8, 12.0)},
            {"annulus_proper_areas_over_mass_squared": [100.0, 200.0]},
            {"annulus_proper_areas_over_mass_squared": (True, 200.0)},
        )
        for override in invalid_arguments:
            arguments = {
                "annulus_edges_over_mass": self.edges,
                "annulus_proper_areas_over_mass_squared": self.areas,
                "disk": self.disk,
            }
            arguments.update(override)
            with self.subTest(override=override):
                with self.assertRaises(TypeError):
                    solve_axisymmetric_returning_radiation_thermal_profile(
                        kernel, **arguments
                    )

        subclass_disk = _DiskSubclass(
            metric=self.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
        )
        with self.assertRaises(TypeError):
            solve_axisymmetric_returning_radiation_thermal_profile(
                kernel,
                annulus_edges_over_mass=self.edges,
                annulus_proper_areas_over_mass_squared=self.areas,
                disk=subclass_disk,
            )
        subclass_kernel = _KernelSubclass(
            annulus_radii_over_mass=self.radii,
            receiver_emitter_coefficients=((0.0, 0.0), (0.0, 0.0)),
            ray_kernel_producer_id="subclass",
        )
        with self.assertRaises(TypeError):
            solve_axisymmetric_returning_radiation_thermal_profile(
                subclass_kernel,
                annulus_edges_over_mass=self.edges,
                annulus_proper_areas_over_mass_squared=self.areas,
                disk=self.disk,
            )

    def test_orientation_is_bound_and_changes_isco_domain(self) -> None:
        retro_disk = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=RETROGRADE,
        )
        with self.assertRaisesRegex(ValueError, "disk ISCO"):
            solve_axisymmetric_returning_radiation_thermal_profile(
                self.kernel(((0.0, 0.0), (0.0, 0.0))),
                annulus_edges_over_mass=self.edges,
                annulus_proper_areas_over_mass_squared=self.areas,
                disk=retro_disk,
            )


class CertifiedReturningRadiationThermalProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(mass_m=1.0, spin_a_m=0.5)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.5,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=12.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=20.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=100.0,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )
        cls.kernel_policy = KerrReturningRadiationKernelPolicy(
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
        cls.area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=8,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=128,
        )
        cls.edges = (
            float(cls.calibration.isco_radius_over_mass),
            float(cls.calibration.outer_radius_over_mass),
        )
        cls._patcher = patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=_low_return_forward_classifier,
        )
        cls._patcher.start()
        cls.source = integrate_kerr_returning_radiation_energy_kernel(
            cls.surface,
            termination=cls.termination,
            annulus_edges_over_mass=cls.edges,
            ray_options=cls.ray_options,
            surface_options=cls.surface_options,
            policy=cls.kernel_policy,
            area_policy=cls.area_policy,
        )
        cls._temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        cls.cache_root = Path(cls._temporary.name)
        cls.cached_source = (
            cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
                cls.surface,
                termination=cls.termination,
                annulus_edges_over_mass=cls.edges,
                cache_root=cls.cache_root / "certified",
                ray_options=cls.ray_options,
                surface_options=cls.surface_options,
                policy=cls.kernel_policy,
                area_policy=cls.area_policy,
                directions_per_task=16,
                jobs=1,
            )
        )
        cls.disk = StationaryNovikovThorneDisk(
            metric=cls.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=PROGRADE,
            colour_correction=1.7,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()
        cls._temporary.cleanup()

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

    def test_certified_entry_replays_source_derives_geometry_and_publishes_watts(
        self,
    ) -> None:
        original_verifier = (
            kernel_module.verify_kerr_returning_radiation_energy_kernel
        )
        with patch.object(
            kernel_module,
            "verify_kerr_returning_radiation_energy_kernel",
            wraps=original_verifier,
        ) as verifier:
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=self.disk,
            )
        self.assertEqual(verifier.call_count, 1)
        self.assertTrue(profile.geometry_provenance_certified)
        self.assertEqual(
            profile.provenance_classification,
            REPLAYED_KERR_FORWARD_CLASSIFICATION,
        )
        self.assertEqual(profile.annulus_edges_over_mass, self.source.annulus_edges_over_mass)
        expected_areas = tuple(
            0.5 * math.fsum((upper, lower))
            for upper, lower in zip(
                self.source.upper_annulus_areas_over_mass_squared,
                self.source.lower_annulus_areas_over_mass_squared,
            )
        )
        self.assertEqual(
            profile.annulus_proper_areas_over_mass_squared,
            expected_areas,
        )
        self.assertIsNotNone(profile.one_face_intrinsic_power_w)
        self.assertGreater(profile.one_face_incident_returning_power_w, 0.0)
        self.assertGreater(
            profile.one_face_outgoing_power_w,
            profile.one_face_intrinsic_power_w,
        )
        self.assertEqual(
            profile.two_face_outgoing_power_w,
            2.0 * profile.one_face_outgoing_power_w,
        )
        power_residual = math.fsum(
            (
                profile.one_face_intrinsic_power_w,
                profile.one_face_incident_returning_power_w,
                -profile.one_face_outgoing_power_w,
            )
        )
        self.assertEqual(
            power_residual,
            profile.one_face_power_balance_residual_w,
        )
        power_roundoff = 32.0 * math.ulp(
            max(
                1.0,
                profile.one_face_intrinsic_power_w,
                profile.one_face_incident_returning_power_w,
                profile.one_face_outgoing_power_w,
            )
        )
        self.assertLessEqual(
            abs(power_residual),
            profile.one_face_power_balance_tolerance_w + power_roundoff,
        )
        descriptor = profile.model_descriptor()
        provenance = descriptor["binding"]["provenance"]
        self.assertTrue(provenance["sameCodeSourceReplayVerified"])
        self.assertTrue(provenance["automaticGeometryDerivation"])
        self.assertEqual(
            provenance["sourceKernelDescriptorSha256"],
            self.source.model_descriptor_sha256,
        )
        self.assertLessEqual(
            descriptor["result"]["maximumRelativeEquationResidual"],
            MAXIMUM_RELATIVE_EQUATION_RESIDUAL,
        )
        profile.revalidate()

    def test_production_cached_source_matches_direct_without_ray_replay(self) -> None:
        direct = solve_certified_kerr_returning_radiation_thermal_profile(
            self.source,
            disk=self.disk,
        )
        with (
            patch.object(
                kernel_module,
                "_trace_direction",
                side_effect=AssertionError("cached profile retraced a ray"),
            ) as tracer,
            patch.object(
                profile_module,
                "validate_and_reduce_cached_kerr_returning_radiation_energy_kernel",
                wraps=(
                    profile_module
                    .validate_and_reduce_cached_kerr_returning_radiation_energy_kernel
                ),
            ) as cached_verifier,
            patch.object(
                profile_module,
                "solve_absorbed_returning_radiation",
                wraps=profile_module.solve_absorbed_returning_radiation,
            ) as fixed_point_solver,
            patch.object(
                profile_module,
                "validate_returning_radiation_solution",
                wraps=profile_module.validate_returning_radiation_solution,
            ) as solution_validator,
        ):
            cached = solve_certified_kerr_returning_radiation_thermal_profile(
                self.cached_source,
                disk=self.disk,
            )
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(cached_verifier.call_count, 1)
        self.assertEqual(fixed_point_solver.call_count, 1)
        self.assertEqual(solution_validator.call_count, 1)
        self.assertEqual(
            cached.provenance_classification,
            CACHED_KERR_FORWARD_CLASSIFICATION,
        )
        self.assertEqual(
            self.cached_source.kernel.model_descriptor_sha256,
            self.source.model_descriptor_sha256,
        )
        self.assertEqual(
            cached.kernel_descriptor_sha256,
            direct.kernel_descriptor_sha256,
        )
        self.assertEqual(
            cached.fixed_point_solution_descriptor_sha256,
            direct.fixed_point_solution_descriptor_sha256,
        )
        for name in (
            "annulus_edges_over_mass",
            "annulus_proper_areas_over_mass_squared",
            "intrinsic_flux_w_m2",
            "incident_returning_flux_w_m2",
            "outgoing_flux_w_m2",
            "intrinsic_effective_temperature_k",
            "outgoing_effective_temperature_k",
        ):
            self.assertEqual(getattr(cached, name), getattr(direct, name))
        provenance = cached.model_descriptor()["binding"]["provenance"]
        self.assertTrue(provenance["cachedDirectionReplayVerified"])
        self.assertFalse(provenance["cachedReplayRetracedRays"])
        self.assertEqual(
            cached.source_authentication_descriptor_sha256,
            provenance["sourceAuthenticationDescriptorSha256"],
        )
        with (
            patch.object(
                kernel_module,
                "_trace_direction",
                side_effect=AssertionError(
                    "cached profile revalidation retraced a ray"
                ),
            ),
            patch.object(
                profile_module,
                "validate_and_reduce_cached_kerr_returning_radiation_energy_kernel",
                wraps=(
                    profile_module
                    .validate_and_reduce_cached_kerr_returning_radiation_energy_kernel
                ),
            ) as revalidation_cache_replay,
            patch.object(
                profile_module,
                "solve_absorbed_returning_radiation",
                wraps=profile_module.solve_absorbed_returning_radiation,
            ) as revalidation_fixed_point_solver,
            patch.object(
                profile_module,
                "validate_returning_radiation_solution",
                wraps=profile_module.validate_returning_radiation_solution,
            ) as revalidation_solution_validator,
        ):
            cached.revalidate()
        self.assertEqual(revalidation_cache_replay.call_count, 1)
        self.assertEqual(revalidation_fixed_point_solver.call_count, 1)
        self.assertEqual(revalidation_solution_validator.call_count, 1)

    def test_certified_live_axis_is_deeply_checked_before_any_source_replay(
        self,
    ) -> None:
        projection = self.source.coarsen_annuli(self.edges)
        profiles = (
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=self.disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                projection,
                disk=self.disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.cached_source,
                disk=self.disk,
            ),
        )
        for profile in profiles:
            kernel = object.__getattribute__(profile, "_kernel")
            count = len(kernel.receiver_emitter_coefficients)
            object.__setattr__(
                kernel,
                "receiver_emitter_coefficients",
                tuple(tuple(0.0 for _ in range(count)) for _ in range(count)),
            )
            with (
                self.subTest(classification=profile.provenance_classification),
                patch.object(
                    profile_module,
                    "verify_and_reduce_kerr_returning_radiation_energy_kernel",
                    side_effect=AssertionError(
                        "stale live axis must fail before direct replay"
                    ),
                ),
                patch.object(
                    profile_module,
                    "verify_and_reduce_kerr_returning_radiation_kernel_projection",
                    side_effect=AssertionError(
                        "stale live axis must fail before projection replay"
                    ),
                ),
                patch.object(
                    profile_module,
                    "validate_and_reduce_cached_kerr_returning_radiation_energy_kernel",
                    side_effect=AssertionError(
                        "stale live axis must fail before cached record scan"
                    ),
                ),
                patch.object(
                    kernel_module,
                    "_trace_direction",
                    side_effect=AssertionError(
                        "stale live axis must fail before tracing rays"
                    ),
                ),
                self.assertRaises(
                    KerrReturningRadiationThermalProfileVerificationError
                ),
            ):
                profile.revalidate()

    def test_self_consistent_foreign_axis_is_rejected_by_source_reduction(
        self,
    ) -> None:
        projection = self.source.coarsen_annuli(self.edges)
        cases = (
            ("direct", self.source),
            ("projection", projection),
            ("cached", self.cached_source),
        )
        for label, source in cases:
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                source,
                disk=self.disk,
            )
            live_kernel = object.__getattribute__(profile, "_kernel")
            count = len(live_kernel.receiver_emitter_coefficients)
            foreign_kernel = AxisymmetricReturningRadiationKernel(
                annulus_radii_over_mass=live_kernel.annulus_radii_over_mass,
                receiver_emitter_coefficients=tuple(
                    tuple(0.0 for _ in range(count)) for _ in range(count)
                ),
                ray_kernel_producer_id=live_kernel.ray_kernel_producer_id,
            )
            object.__setattr__(profile, "_kernel", foreign_kernel)
            with (
                self.subTest(source=label),
                patch.object(
                    profile_module,
                    "verify_and_reduce_kerr_returning_radiation_energy_kernel",
                    wraps=(
                        profile_module
                        .verify_and_reduce_kerr_returning_radiation_energy_kernel
                    ),
                ) as direct_replay,
                patch.object(
                    profile_module,
                    "verify_and_reduce_kerr_returning_radiation_kernel_projection",
                    wraps=(
                        profile_module
                        .verify_and_reduce_kerr_returning_radiation_kernel_projection
                    ),
                ) as projection_replay,
                patch.object(
                    profile_module,
                    "validate_and_reduce_cached_kerr_returning_radiation_energy_kernel",
                    wraps=(
                        profile_module
                        .validate_and_reduce_cached_kerr_returning_radiation_energy_kernel
                    ),
                ) as cached_replay,
                patch.object(
                    kernel_module,
                    "_trace_direction",
                    side_effect=_low_return_forward_classifier,
                ) as ray_trace,
                patch.object(
                    profile_module,
                    "solve_absorbed_returning_radiation",
                    side_effect=AssertionError(
                        "foreign axis must fail before fixed-point solve"
                    ),
                ) as fixed_point_solver,
                self.assertRaises(
                    KerrReturningRadiationThermalProfileVerificationError
                ),
            ):
                profile.revalidate()
            self.assertEqual(direct_replay.call_count, label == "direct")
            self.assertEqual(projection_replay.call_count, label == "projection")
            self.assertEqual(cached_replay.call_count, label == "cached")
            if label == "cached":
                ray_trace.assert_not_called()
            else:
                self.assertGreater(ray_trace.call_count, 0)
            fixed_point_solver.assert_not_called()

    def test_equal_foreign_live_owners_follow_value_identity_policy(self) -> None:
        profile = solve_certified_kerr_returning_radiation_thermal_profile(
            self.source,
            disk=self.disk,
        )
        live_kernel = object.__getattribute__(profile, "_kernel")
        live_disk = object.__getattribute__(profile, "_disk")
        live_policy = object.__getattribute__(profile, "_policy")
        foreign_metric = self._forge(live_disk.metric)
        foreign_disk = self._forge(live_disk, metric=foreign_metric)
        foreign_kernel = self._forge(live_kernel)
        foreign_policy = self._forge(live_policy)
        self.assertIsNot(foreign_kernel, live_kernel)
        self.assertIsNot(foreign_disk, live_disk)
        self.assertIsNot(foreign_policy, live_policy)
        object.__setattr__(profile, "_kernel", foreign_kernel)
        object.__setattr__(profile, "_disk", foreign_disk)
        object.__setattr__(profile, "_policy", foreign_policy)
        profile.revalidate()

    def test_cached_profile_identity_excludes_root_chunk_and_reuse(self) -> None:
        alternatives = []
        for name, chunk in (("identity-a", 3), ("identity-b", 16)):
            execution = (
                cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
                    self.surface,
                    termination=self.termination,
                    annulus_edges_over_mass=self.edges,
                    cache_root=self.cache_root / name,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                    policy=self.kernel_policy,
                    area_policy=self.area_policy,
                    directions_per_task=chunk,
                    jobs=1,
                )
            )
            alternatives.append(
                solve_certified_kerr_returning_radiation_thermal_profile(
                    execution,
                    disk=self.disk,
                )
            )
        reused_execution = (
            cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
                self.surface,
                termination=self.termination,
                annulus_edges_over_mass=self.edges,
                cache_root=self.cache_root / "identity-a",
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                policy=self.kernel_policy,
                area_policy=self.area_policy,
                directions_per_task=3,
                jobs=1,
            )
        )
        alternatives.append(
            solve_certified_kerr_returning_radiation_thermal_profile(
                reused_execution,
                disk=self.disk,
            )
        )
        self.assertGreater(reused_execution.execution_audit.reused_tasks, 0)
        self.assertEqual(
            alternatives[0].model_descriptor_sha256,
            alternatives[1].model_descriptor_sha256,
        )
        self.assertEqual(
            alternatives[0].model_descriptor_sha256,
            alternatives[2].model_descriptor_sha256,
        )
        descriptor_json = object.__getattribute__(alternatives[0], "_descriptor_json")
        self.assertNotIn(str(self.cache_root), descriptor_json)
        self.assertNotIn("directionsPerTask", descriptor_json)
        self.assertNotIn("reusedTasks", descriptor_json)

    def test_authenticated_projection_is_an_equivalent_certified_source(self) -> None:
        projection = self.source.coarsen_annuli(self.edges)
        original_verifier = (
            kernel_module.verify_kerr_returning_radiation_kernel_projection
        )
        original_full_verifier = (
            kernel_module.verify_kerr_returning_radiation_energy_kernel
        )
        with (
            patch.object(
                kernel_module,
                "verify_kerr_returning_radiation_kernel_projection",
                wraps=original_verifier,
            ) as verifier,
            patch.object(
                kernel_module,
                "verify_kerr_returning_radiation_energy_kernel",
                wraps=original_full_verifier,
            ) as full_verifier,
        ):
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                projection,
                disk=self.disk,
            )
        self.assertEqual(verifier.call_count, 1)
        self.assertEqual(full_verifier.call_count, 1)
        self.assertTrue(profile.geometry_provenance_certified)
        self.assertEqual(
            profile.model_descriptor()["binding"]["provenance"][
                "sourceKernelType"
            ],
            "KerrForwardReturningRadiationKernelProjection",
        )
        profile.revalidate()

    def test_zero_and_point_nine_spin_mismatches_fail_closed(self) -> None:
        for spin in (0.0, 0.9):
            disk = StationaryNovikovThorneDisk(
                metric=KerrKerrSchildMetric(mass_m=1.0, spin_a_m=spin),
                black_hole_mass_kg=1.0e31,
                mass_accretion_rate_kg_s=1.0e18,
            )
            with self.subTest(spin=spin), self.assertRaisesRegex(
                ValueError,
                "signed Kerr spin",
            ):
                solve_certified_kerr_returning_radiation_thermal_profile(
                    self.source,
                    disk=disk,
                )

    def test_metric_mass_and_orientation_mismatches_fail_closed(self) -> None:
        mass_mismatch = StationaryNovikovThorneDisk(
            metric=KerrKerrSchildMetric(mass_m=2.0, spin_a_m=1.0),
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
        )
        with self.assertRaisesRegex(ValueError, "metric mass"):
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=mass_mismatch,
            )
        orientation_mismatch = StationaryNovikovThorneDisk(
            metric=self.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=RETROGRADE,
        )
        with self.assertRaisesRegex(ValueError, "orientation"):
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=orientation_mismatch,
            )
        guard_mismatch = StationaryNovikovThorneDisk(
            metric=KerrKerrSchildMetric(
                mass_m=1.0,
                spin_a_m=0.5,
                singularity_guard_m=2.0e-9,
            ),
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
        )
        with self.assertRaisesRegex(ValueError, "singularity guard"):
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=guard_mismatch,
            )

    def test_certified_public_geometry_tampering_is_rejected(self) -> None:
        for name, forged_value in (
            ("annulus_edges_over_mass", (self.edges[0], 999.0)),
            ("annulus_proper_areas_over_mass_squared", (1.0e300,)),
        ):
            profile = solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=self.disk,
            )
            object.__setattr__(profile, name, forged_value)
            with self.subTest(name=name), self.assertRaisesRegex(
                KerrReturningRadiationThermalProfileVerificationError,
                f"profile.{name}",
            ):
                profile.revalidate()


if __name__ == "__main__":
    unittest.main()
